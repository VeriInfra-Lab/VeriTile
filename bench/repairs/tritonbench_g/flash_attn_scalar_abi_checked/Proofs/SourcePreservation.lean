import bench.repairs.tritonbench_g.flash_attn_scalar_abi_checked.Kernels
import VeriTile.Triton.Float.ProfiledRewrite

/-! Historical proof fixture only. The authoritative original and candidate are
ordinary triton declarations in Kernels; neither depends on this file. This
retains the pre-cleanup program solely to check complete semantic preservation. -/
namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.SourcePreservation
open VeriTile Triton FP.Structural FP.ProfiledRewrite FlashAttn FlashAttn.ActualSource
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 3000000
private def historicalDeclaration (p : Parameters) : ComputeKernel :=
  let Q : RegionName := "Q"
  let K : RegionName := "K"
  let V : RegionName := "V"
  let L : RegionName := "L"
  let O : RegionName := "O"
  let IS_CAUSAL := p.IS_CAUSAL
  ComputeKernel.mk ["Q","K","V"] ["L","O"] (triton {
  tl.region Q = tl.float32, K = tl.float32, V = tl.float32, L = tl.float32, O = tl.float32
  sm_scale = $((0 : ℝ))
  start_m = tl.program_id(0)
  off_bs_head = tl.program_id(1)
  off_bs = off_bs_head // $((p.HEAD : Nat))
  off_head = off_bs_head % $((p.HEAD : Nat))
  q_base_offset = off_bs * $((p.stride_q_bs : Nat)) + off_head * $((p.stride_q_head : Nat))
  k_base_offset = off_bs * $((p.stride_k_bs : Nat)) + off_head * $((p.stride_k_head : Nat))
  v_base_offset = off_bs * $((p.stride_v_bs : Nat)) + off_head * $((p.stride_v_head : Nat))
  o_base_offset = off_bs * $((p.stride_o_bs : Nat)) + off_head * $((p.stride_o_head : Nat))
  Q_block_ptr = tl.make_block_ptr(base=Q + q_base_offset,
    shape=($((p.SEQLEN : Nat)), $((p.DIM : Nat))), strides=($((p.stride_q_seqlen : Nat)), $((p.stride_q_dim : Nat))),
    offsets=(start_m * $(128), 0), block_shape=(128, $((p.DIM : Nat))), order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + k_base_offset,
    shape=($((p.DIM : Nat)), $((p.SEQLEN : Nat))), strides=($((p.stride_k_dim : Nat)), $((p.stride_k_seqlen : Nat))),
    offsets=(0, 0), block_shape=($((p.DIM : Nat)), 64), order=(0, 1))
  V_block_ptr = tl.make_block_ptr(base=V + v_base_offset,
    shape=($((p.SEQLEN : Nat)), $((p.DIM : Nat))), strides=($((p.stride_v_seqlen : Nat)), $((p.stride_v_dim : Nat))),
    offsets=(0, 0), block_shape=(64, $((p.DIM : Nat))), order=(1, 0))
  off_m = start_m * $(128) + tl.arange(0, 128)
  off_n = tl.arange(0, 64)
  max = tl.zeros([128], dtype=tl.float32) - float("inf")
  denom = tl.zeros([128], dtype=tl.float32)
  out_buffer = tl.zeros([128, $((p.DIM : Nat))], dtype=tl.float32)
  qk_scale = sm_scale * 1.44269504
  q = tl.load(Q_block_ptr, boundary_check=([0] : List Nat), padding_option="zero")
  q = (q * qk_scale).to(tl.float16)
  lo = 0
  hi = (tl.minimum((start_m + $(1)) * $(128), $((p.SEQLEN : Nat))) if IS_CAUSAL else $((p.SEQLEN : Nat)))
  for start_n in range(lo, hi, 64) {
    k = tl.load(K_block_ptr, boundary_check=([1] : List Nat), padding_option="zero")
    v = tl.load(V_block_ptr, boundary_check=([0] : List Nat), padding_option="zero")
    qk = tl.zeros([128, 64], dtype=tl.float32)
    if IS_CAUSAL {
      qk = tl.where(off_m[:, None] >= (start_n + off_n[None, :]), qk, float("-inf"))
    }
    qk = tl.where(start_n + off_n[None, :] < $((p.SEQLEN : Nat)), qk, float("-inf"))
    qk += tl.dot(q.to(tl.float32), k)
    max_new = tl.maximum(max, tl.max(qk, 1))
    alpha = tl.math.exp2(max - max_new)
    nume = tl.math.exp2(qk - max_new[:, None])
    out_scale = denom * 0 + alpha
    out_buffer *= out_scale[:, None]
    out_buffer += tl.dot(nume.to(tl.float16).to(tl.float32), v)
    denom = denom * alpha + tl.sum(nume, 1)
    max = max_new
    K_block_ptr = tl.advance(K_block_ptr, (0, 64))
    V_block_ptr = tl.advance(V_block_ptr, (64, 0))
  }
  out_buffer = out_buffer / denom[:, None]
  l_ptr = L + off_bs_head * $((p.SEQLEN : Nat)) + off_m
  tl.store(l_ptr, max + tl.math.log2(denom), mask=off_m < $((p.SEQLEN : Nat)))
  O_block_ptr = tl.make_block_ptr(base=O + o_base_offset,
    shape=($((p.SEQLEN : Nat)), $((p.DIM : Nat))), strides=($((p.stride_o_seqlen : Nat)), $((p.stride_o_dim : Nat))),
    offsets=(start_m * $(128), 0), block_shape=(128, $((p.DIM : Nat))), order=(1, 0))
  tl.store(O_block_ptr, out_buffer.to(tl.float16).to(tl.float32), boundary_check=([0] : List Nat))
}).surfaceBody

private def historicalKernel (p : Parameters) : ComputeKernel :=
  ComputeKernel.mk ["Q","K","V"] ["L","O"] ((historicalDeclaration p).surfaceBody.drop 1)

private def before (p : Parameters) := (ActualSource.kernel p).surfaceBody.take 16
private def after (p : Parameters) := (ActualSource.kernel p).surfaceBody.drop 17
private def previousScale (p : Parameters) : ComputeStmt :=
  ((historicalKernel p).surfaceBody[16]?).getD (.effectMarker "tl.debug_barrier")
private def explicitScale (p : Parameters) : ComputeStmt :=
  ((ActualSource.kernel p).surfaceBody[16]?).getD (.effectMarker "tl.debug_barrier")

private theorem previous_split (p : Parameters) :
    (historicalKernel p).surfaceBody = before p ++ previousScale p :: after p := by rfl
private theorem current_split (p : Parameters) :
    (ActualSource.kernel p).surfaceBody = before p ++ explicitScale p :: after p := by rfl

/-- The only changed AST node makes the already selected default FP32 precision
explicit. This is exact in the existing program view for all primitive models. -/
private theorem scale_step {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) :
    step (engine M) (previousScale p) s = step (engine M) (explicitScale p) s := by
  simp only [previousScale,explicitScale,historicalKernel,historicalDeclaration,
    ActualSource.kernel,ComputeKernel.surfaceBody,List.getElem?_cons_zero,List.getElem?_cons_succ,
    List.drop,Option.getD,step,evalExpr,evalComputeOp,evalOp_unfold,engine,
    Algebra.withDefaultPrecision,resolvePrecision,numeric,ComputeDType.eraseDType]

/-- Complete pre-cleanup source execution equals the direct parameterized
quotation, including failure and every register/memory field. -/
theorem complete_program_preserved {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    FP.Structural.exec (engine M) (historicalKernel p) s =
      FP.Structural.exec (engine M) (ActualSource.kernel p) s := by
  change run (engine M) (historicalKernel p).surfaceBody s =
    run (engine M) (ActualSource.kernel p).surfaceBody s
  rw [previous_split,current_split,run_append,run_append]
  congr 1
  funext t
  simp only [run,scale_step]

#axiomsClean complete_program_preserved
end
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.SourcePreservation
