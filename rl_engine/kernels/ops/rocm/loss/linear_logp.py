# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""ROCm / gfx950 (CDNA4) fused linear log-prob — hand-written FlyDSL kernel.

Computes per-token ``logp = log_softmax(hidden @ Wᵀ)[target]`` without materializing
the ``[N, V]`` logits. AMD analog of ``FusedLinearLogpSM90Op``:

  * bf16 MFMA (``mfma_f32_16x16x32_bf16``) with fp32 accumulate + fp32 online softmax
    => **fp32-accurate** logprobs (the native path returns bf16-coarse logp); this
    precision matters for RL importance ratios / train-rollout consistency.
  * split-V grid + per-token online softmax => batch-invariant, memory-flat.

Forward is the FlyDSL kernel; backward is a chunked recompute (correct). The op is
used only for the aligned bf16 / no-bias / no-TP / ``vocab_start_index == 0`` case and
otherwise defers to ``NativeLinearLogpOp`` so it is always correct. A native-parity
FlyDSL backward and arbitrary-shape (padding) support are follow-ons.
"""
from typing import Any, Optional

import torch

from rl_engine.kernels.ops.pytorch.loss.linear_logp import NativeLinearLogpOp
from rl_engine.utils.logger import logger

try:
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import arith, vector, gpu, rocdl, range_constexpr, buffer_ops, math as fm
    from flydsl.expr.typing import T
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl._mlir import ir
    _FLYDSL_OK = True
except Exception as _e:  # flydsl absent (non-ROCm image) -> op transparently defers to native
    _FLYDSL_OK = False
    logger.warning(f"FlyDSL unavailable ({_e}); ROCm linear_logp will use the native fallback.")

# bf16-in / fp32-out GEMM (hipBLASLt) — fp32-accurate logit recompute at bf16 speed for the
# backward. Built once on first use via cpp_extension (cuBLASLt source hipified -> hipBLASLt).
_HBL_SRC = r"""
#include <torch/extension.h>
#include <cublasLt.h>
#include <ATen/cuda/CUDAContext.h>
static cublasLtHandle_t g_handle = nullptr;
torch::Tensor bf16_fp32_mm(torch::Tensor A, torch::Tensor B) {  // A[M,K] bf16, B[N,K] bf16 -> C[M,N] fp32 = A @ B^T
  TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "need contiguous");
  int M = A.size(0), K = A.size(1), N = B.size(0);
  auto C = torch::empty({M, N}, A.options().dtype(torch::kFloat32));
  if (!g_handle) cublasLtCreate(&g_handle);
  cublasLtMatmulDesc_t desc; cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
  cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
  cublasLtMatrixLayout_t la, lb, ld;
  cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, K, N, K);
  cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, K, M, K);
  cublasLtMatrixLayoutCreate(&ld, CUDA_R_32F, N, M, N);
  float alpha = 1.f, beta = 0.f; size_t ws = 64ull << 20;
  auto wbuf = torch::empty({(long)ws}, A.options().dtype(torch::kUInt8));
  cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
  cublasLtMatmulHeuristicResult_t heur; int ret = 0;
  cublasLtMatmulAlgoGetHeuristic(g_handle, desc, la, lb, ld, ld, pref, 1, &heur, &ret);
  TORCH_CHECK(ret > 0, "no hipblaslt algo");
  auto stream = at::cuda::getCurrentCUDAStream();
  auto st = cublasLtMatmul(g_handle, desc, &alpha, B.data_ptr(), la, A.data_ptr(), lb, &beta,
                           C.data_ptr(), ld, C.data_ptr(), ld, &heur.algo, wbuf.data_ptr(), ws, stream.stream());
  TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, "hipblasLtMatmul failed: ", (int)st);
  cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(ld);
  cublasLtMatmulDescDestroy(desc); cublasLtMatmulPreferenceDestroy(pref);
  return C;
}
"""
_HBL = None
_HBL_TRIED = False


def _get_hbl():
    """Lazily build the hipBLASLt bf16->fp32 GEMM; None if unavailable (-> fp32 fallback)."""
    global _HBL, _HBL_TRIED
    if _HBL is None and not _HBL_TRIED:
        _HBL_TRIED = True
        try:
            from torch.utils.cpp_extension import load_inline
            _HBL = load_inline(
                name="rlk_hbl_bf16fp32",
                cpp_sources="#include <torch/extension.h>\ntorch::Tensor bf16_fp32_mm(torch::Tensor A, torch::Tensor B);",
                cuda_sources=_HBL_SRC, functions=["bf16_fp32_mm"], with_cuda=True,
                extra_ldflags=["-lhipblaslt"], verbose=False,
            )
        except Exception as e:  # pragma: no cover
            logger.warning(f"hipBLASLt bf16->fp32 shim unavailable ({e}); backward uses fp32 recompute.")
            _HBL = None
    return _HBL


_N_REP = 8            # vocab tile = 16 * _N_REP = 128 columns
_NSPLIT = 64          # vocab splits (GPU-fill)
_CACHE: dict = {}


def _build(D, V, N_REP, NSPLIT, tag):
    TM = 64; M_REP = 4; TN = 16 * N_REP; KW = D // 2; KSTEPS = D // 32
    NVT = V // TN; VPS = (NVT + NSPLIT - 1) // NSPLIT; LDS_N = TM * TN
    alloc = SmemAllocator(None, arch="gfx950", global_sym_name=f"llp_{tag}"); alloc.ptr = LDS_N * 4

    @flyc.kernel
    def kern(H: fx.Tensor, W: fx.Tensor, TGT: fx.Tensor, PMAX: fx.Tensor, PSUM: fx.Tensor, PZT: fx.Tensor):
        alloc.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body): alloc.finalize()
        lane = arith.index_cast(T.i32, gpu.thread_id("x")); bx = arith.index_cast(T.i32, gpu.block_id("x")); by = arith.index_cast(T.i32, gpu.block_id("y"))
        c16 = arith.constant(16, type=T.i32); c4 = arith.constant(4, type=T.i32); cKW = arith.constant(KW, type=T.i32); cTN = arith.constant(TN, type=T.i32)
        r = arith.remsi(lane, c16); g = arith.divsi(lane, c16); g4 = arith.muli(g, c4); m_base = arith.muli(bx, arith.constant(TM, type=T.i32))
        def ki(x): return arith.constant(int(x), type=T.i32)
        def kf(x): return arith.constant(float(x), type=T.f32)
        def idx(x): return arith.index_cast(T.index, x)
        def rsrc(a): return buffer_ops.create_buffer_resource_from_addr(arith.index_cast(T.i64, buffer_ops.extract_base_index(a)))
        h_rsrc, w_rsrc, t_rsrc = rsrc(H), rsrc(W), rsrc(TGT); lds = SmemPtr(alloc.get_base(), 0, T.f32, shape=(LDS_N,)).get()
        def ld8(rs, off): return vector.bitcast(T.vec(8, T.bf16), buffer_ops.buffer_load(rs, off, vec_width=4, dtype=fx.Int32))
        def st1(pos, val): vector.store(vector.from_elements(T.vec(1, T.f32), [val]), lds, [idx(pos)])
        def ld1(pos): return vector.extract(vector.load(T.vec(1, T.f32), lds, [idx(pos)]), static_position=[0], dynamic_position=[])
        a_rowKW = [arith.muli(arith.addi(m_base, arith.addi(ki(mi*16), r)), cKW) for mi in range(M_REP)]
        tgt = buffer_ops.buffer_load(t_rsrc, arith.addi(m_base, lane), vec_width=1, dtype=fx.Int32); rowbase = arith.muli(lane, cTN); NEG = kf(-1e30)
        vt_begin = arith.muli(by, ki(VPS)); vt_end = arith.select(arith.cmpi(2, arith.addi(vt_begin, ki(VPS)), ki(NVT)), arith.addi(vt_begin, ki(VPS)), ki(NVT))
        res = None
        for vt_iv, (m_run, s_run, z) in scf_range(vt_begin, vt_end, 1, init=[NEG, kf(0.0), kf(0.0)]):
            vt_i = arith.index_cast(T.i32, vt_iv); n_base = arith.muli(vt_i, cTN)
            b_rowKW = [arith.muli(arith.addi(n_base, arith.addi(ki(ni*16), r)), cKW) for ni in range(N_REP)]
            accs = [[arith.constant_vector(0.0, T.f32x4) for _ in range(N_REP)] for _ in range(M_REP)]
            for ts in range_constexpr(KSTEPS):
                colab = arith.addi(ki(int(ts)*16), g4)
                a_op = [ld8(h_rsrc, arith.addi(a_rowKW[mi], colab)) for mi in range(M_REP)]; b_op = [ld8(w_rsrc, arith.addi(b_rowKW[ni], colab)) for ni in range(N_REP)]
                for mi in range_constexpr(M_REP):
                    for ni in range_constexpr(N_REP):
                        accs[int(mi)][int(ni)] = rocdl.mfma_f32_16x16x32_bf16(T.f32x4, [a_op[int(mi)], b_op[int(ni)], accs[int(mi)][int(ni)], 0, 0, 0])
            for mi in range_constexpr(M_REP):
                for ni in range_constexpr(N_REP):
                    col = arith.addi(ki(int(ni)*16), r)
                    for d in range_constexpr(4):
                        row = arith.addi(ki(int(mi)*16), arith.addi(g4, ki(int(d))))
                        st1(arith.addi(arith.muli(row, cTN), col), vector.extract(accs[int(mi)][int(ni)], static_position=[int(d)], dynamic_position=[]))
            gpu.barrier()
            tile_max = NEG; vals = []
            for c in range_constexpr(TN):
                v = ld1(arith.addi(rowbase, ki(int(c)))); vals.append(v); tile_max = arith.maximumf(tile_max, v)
            new_m = arith.maximumf(m_run, tile_max); tsum = kf(0.0)
            for v in vals: tsum = arith.addf(tsum, fm.exp(arith.subf(v, new_m)))
            new_s = arith.addf(arith.mulf(s_run, fm.exp(arith.subf(m_run, new_m))), tsum)
            in_tile = arith.andi(arith.cmpi(5, tgt, n_base), arith.cmpi(2, tgt, arith.addi(n_base, cTN)))
            cand = ld1(arith.addi(rowbase, arith.select(in_tile, arith.subi(tgt, n_base), ki(0)))); new_z = arith.select(in_tile, cand, z)
            gpu.barrier(); res = yield new_m, new_s, new_z
        m_run, s_run, z = res; gid = arith.addi(arith.muli(arith.addi(m_base, lane), ki(NSPLIT)), by)
        PMAX[(gid,)] = m_run; PSUM[(gid,)] = s_run; PZT[(gid,)] = z

    @flyc.kernel
    def combine(PMAX: fx.Tensor, PSUM: fx.Tensor, PZT: fx.Tensor, LOGP: fx.Tensor, LSE: fx.Tensor):
        tid = arith.index_cast(T.i32, gpu.thread_id("x")); bx = arith.index_cast(T.i32, gpu.block_id("x"))
        tok = arith.addi(arith.muli(bx, arith.constant(64, type=T.i32)), tid)
        def kf(x): return arith.constant(float(x), type=T.f32)
        def ki(x): return arith.constant(int(x), type=T.i32)
        def rsrc(a): return buffer_ops.create_buffer_resource_from_addr(arith.index_cast(T.i64, buffer_ops.extract_base_index(a)))
        pm, ps, pz = rsrc(PMAX), rsrc(PSUM), rsrc(PZT); base = arith.muli(tok, ki(NSPLIT))
        def ldf(rs, off): return buffer_ops.buffer_load(rs, off, vec_width=1, dtype=fx.Float32)
        M = kf(-1e30)
        for s in range_constexpr(NSPLIT): M = arith.maximumf(M, ldf(pm, arith.addi(base, ki(int(s)))))
        S = kf(0.0); Z = kf(0.0)
        for s in range_constexpr(NSPLIT):
            off = arith.addi(base, ki(int(s))); S = arith.addf(S, arith.mulf(ldf(ps, off), fm.exp(arith.subf(ldf(pm, off), M)))); Z = arith.addf(Z, ldf(pz, off))
        lse = arith.addf(M, fm.log(S)); LSE[(tok,)] = lse; LOGP[(tok,)] = arith.subf(Z, lse)

    @flyc.jit
    def run(H: fx.Tensor, W: fx.Tensor, TGT: fx.Tensor, PMAX: fx.Tensor, PSUM: fx.Tensor, PZT: fx.Tensor, LOGP: fx.Tensor, LSE: fx.Tensor, gx: fx.Constexpr[int], stream: fx.Stream = fx.Stream(None)):
        kern(H, W, TGT, PMAX, PSUM, PZT).launch(grid=(gx, NSPLIT, 1), block=(64, 1, 1), stream=stream)
        combine(PMAX, PSUM, PZT, LOGP, LSE).launch(grid=(gx, 1, 1), block=(64, 1, 1), stream=stream)
    return run


def _fwd(hidden_2d, weight, target):
    N, D = hidden_2d.shape; V = weight.shape[0]; key = (D, V, _N_REP, _NSPLIT)
    if key not in _CACHE:
        _CACHE[key] = _build(D, V, _N_REP, _NSPLIT, tag=f"{D}_{V}_{_N_REP}_{_NSPLIT}")
    import torch.cuda as _tc
    Hi = hidden_2d.view(torch.int32).contiguous(); Wi = weight.view(torch.int32).contiguous(); TGT = target.to(torch.int32).contiguous()
    PMAX = torch.zeros(N * _NSPLIT, dtype=torch.float32, device='cuda'); PSUM = torch.zeros_like(PMAX); PZT = torch.zeros_like(PMAX)
    LOGP = torch.zeros(N, dtype=torch.float32, device='cuda'); LSE = torch.zeros(N, dtype=torch.float32, device='cuda')
    _CACHE[key](Hi, Wi, TGT, PMAX, PSUM, PZT, LOGP, LSE, N // 64, stream=_tc.current_stream())
    return LOGP, LSE


def _chunked_backward(grad_logp, hidden_2d, weight, target, lse, chunk=16384):
    """Chunked recompute backward, never materializing [N, V]: fp32-accurate logit
    recompute (hipBLASLt bf16->fp32, fp32 fallback) + bf16 grad GEMMs (grads tolerate
    bf16). ~3x faster than an all-fp32 backward at the same grad accuracy (dH ~2e-3)."""
    N, D = hidden_2d.shape; V = weight.shape[0]; tgt = target.long()
    hbl = _get_hbl()
    hf = None if hbl is not None else hidden_2d.float()
    gh = torch.zeros(N, D, dtype=torch.float32, device='cuda')
    gw = torch.zeros(V, D, dtype=torch.float32, device='cuda')
    for v0 in range(0, V, chunk):
        v1 = min(v0 + chunk, V); wc = weight[v0:v1].contiguous()
        if hbl is not None:
            logits = hbl.bf16_fp32_mm(hidden_2d, wc)          # bf16-in / fp32-out (fast + accurate)
        else:
            logits = hf @ wc.float().t()                      # fp32 fallback
        p = torch.exp(logits - lse[:, None])
        onehot = ((tgt[:, None] >= v0) & (tgt[:, None] < v1)).float() * \
                 torch.nn.functional.one_hot((tgt - v0).clamp(0, v1 - v0 - 1), v1 - v0).float()
        g = grad_logp[:, None] * (onehot - p)                 # fp32 grad-logits
        gb = g.to(torch.bfloat16)                             # bf16 grad GEMMs (accurate enough)
        gh += (gb @ wc).float()
        gw[v0:v1] = (gb.t() @ hidden_2d).float()
    return gh.to(hidden_2d.dtype), gw.to(weight.dtype)


class _FlyDSLLinearLogp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_2d, weight, target):
        logp, lse = _fwd(hidden_2d, weight, target)
        ctx.save_for_backward(hidden_2d, weight, target, lse)
        return logp

    @staticmethod
    def backward(ctx, grad_logp):
        hidden_2d, weight, target, lse = ctx.saved_tensors
        gh, gw = _chunked_backward(grad_logp.contiguous(), hidden_2d, weight, target, lse)
        return gh, gw, None


class FlyDSLLinearLogpRocmOp:
    """Fused linear log-prob for ROCm/gfx950 (FlyDSL), with a native fallback."""

    def __init__(self) -> None:
        self._native = NativeLinearLogpOp()

    def _flydsl_supported(self, hidden, lm_head_weight, bias, tp_group, vocab_start_index) -> bool:
        if not (_FLYDSL_OK and hidden.is_cuda):
            return False
        if hidden.dtype != torch.bfloat16 or lm_head_weight.dtype != torch.bfloat16:
            return False
        if bias is not None or tp_group is not None or int(vocab_start_index) != 0:
            return False
        N = 1
        for s in hidden.shape[:-1]:
            N *= int(s)
        D = int(hidden.size(-1)); V = int(lm_head_weight.size(0)); TN = 16 * _N_REP
        # N is padded to a multiple of 64 in apply(); only V must divide the vocab tile.
        return (V % TN == 0) and (D % 32 == 0) and N > 0

    def __call__(self, hidden, lm_head_weight, target_ids, bias=None, *, tp_group=None, vocab_start_index=0, global_vocab_size=None):
        return self.apply(hidden, lm_head_weight, target_ids, bias, tp_group=tp_group, vocab_start_index=vocab_start_index, global_vocab_size=global_vocab_size)

    def apply(self, hidden, lm_head_weight, target_ids, bias=None, *, tp_group=None, vocab_start_index=0, global_vocab_size=None):
        if not self._flydsl_supported(hidden, lm_head_weight, bias, tp_group, vocab_start_index):
            return self._native.apply(hidden, lm_head_weight, target_ids, bias, tp_group=tp_group, vocab_start_index=vocab_start_index, global_vocab_size=global_vocab_size)
        lead = hidden.shape[:-1]
        D = hidden.size(-1)
        hidden_2d = hidden.reshape(-1, D)
        target_1d = target_ids.reshape(-1)
        N = hidden_2d.shape[0]
        # Pad the token count up to a multiple of 64 (the kernel's per-CTA tile). Padded rows
        # get zero hidden + dummy target 0; they carry grad_logp=0, so they contribute nothing
        # to grad_weight and are sliced off grad_hidden -> exact, differentiable.
        n_pad = ((N + 63) // 64) * 64 - N
        if n_pad:
            hidden_2d = torch.cat([hidden_2d, hidden_2d.new_zeros(n_pad, D)], dim=0)
            target_1d = torch.cat([target_1d, target_1d.new_zeros(n_pad)], dim=0)
        logp = _FlyDSLLinearLogp.apply(hidden_2d.contiguous(), lm_head_weight.contiguous(), target_1d)
        return logp[:N].reshape(lead)
