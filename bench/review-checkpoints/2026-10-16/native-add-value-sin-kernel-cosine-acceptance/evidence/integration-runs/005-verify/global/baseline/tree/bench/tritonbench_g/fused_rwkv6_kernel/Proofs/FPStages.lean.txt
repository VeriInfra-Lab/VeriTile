import bench.tritonbench_g.fused_rwkv6_kernel.FusedRwkv6Kernel
import bench.tritonbench_g.fused_recurrent_delta.Proofs.FPState
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FusedRwkv6Kernel.FPStages
open VeriTile.Triton FP.Structural
open FusedRecurrentDelta.FPStages (stateAddress keyAddress valueAddress outputAddress keyIndex valueIndex maskedInput stateActive stateScatter Post post_of_memory outputAddress_injective)
open FusedRecurrentHgrn.FPStages (scatter scatter_memory_congr)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 15000
set_option linter.unusedSimpArgs false

/-! fp32 stage contracts retaining the true exp callback, every source
cast, ordered scalar operations, masks and reductions. This is neither a
Real.exp interpretation nor an assertion of GPU exp/reduction conformance. -/
def stateStep
    (BHPrev k v w BHOut : RegionName)
    (t s_k_h s_v_h K V BK BV : Nat) : ComputeKernel :=
  ComputeKernel.mk [BHPrev, k, v, w] [BHOut] (triton {
  tl.region BHPrev = tl.float32, k = tl.float32, v = tl.float32, w = tl.float32, BHOut = tl.float32
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bv[:, None] & mask_bk[None, :]
  prev = tl.load(BHPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]),
    mask=mask_kv, other=0.0)
  b_k = tl.load(k + i_bh * $(s_k_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0).to(tl.float32)
  b_v = tl.load(v + i_bh * $(s_v_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0).to(tl.float32)
  b_w = tl.load(w + i_bh * $(s_k_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0).to(tl.float32)
  b_w = tl.exp(b_w)
  b_kv = b_k[None, :] * b_v[:, None]
  acc = prev * b_w[None, :] + b_kv
  tl.store(BHOut + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]),
    (acc).to(BHOut.dtype.element_ty), mask=mask_kv)
}).surfaceBody

def outputStep
    (BHPrev q k v u o : RegionName)
    (t s_k_h s_v_h B H T K V BK BV : Nat) (scale : ℝ) :
    ComputeKernel :=
  ComputeKernel.mk [BHPrev, q, k, v, u] [o] (triton {
  tl.region BHPrev = tl.float32, q = tl.float32, k = tl.float32, v = tl.float32, u = tl.float32, o = tl.float32
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  i_h = i_bh % $(H)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bv[:, None] & mask_bk[None, :]
  prev = tl.load(BHPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]),
    mask=mask_kv, other=0.0)
  b_k = tl.load(k + i_bh * $(s_k_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0).to(tl.float32)
  b_v = tl.load(v + i_bh * $(s_v_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0).to(tl.float32)
  b_q = tl.load(q + i_bh * $(s_k_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0).to(tl.float32) * $(scale)
  b_u = tl.load(u + i_h * $(K) + offs_k + i_k * $(BK), mask=mask_bk, other=0.0).to(tl.float32)
  b_kv = b_k[None, :] * b_v[:, None]
  b_o = (prev + b_kv * b_u[None, :]) * b_q[None, :]
  b_o = tl.sum(b_o, axis=1)
  tl.store(o + (i_bh + i_k * $(B) * $(H)) * $(s_v_h) + i_v * $(BV) + offs_v +
    $(t) * $(V), (b_o).to(o.dtype.element_ty), mask=mask_bv)
}).surfaceBody

def finalStep
    (BHPrev k v w BHOut : RegionName)
    (STORE_FINAL_STATE : Bool) (t s_k_h s_v_h K V BK BV : Nat) : ComputeKernel :=
  ComputeKernel.mk [BHPrev, k, v, w] [BHOut] (triton {
  tl.region BHPrev = tl.float32, k = tl.float32, v = tl.float32, w = tl.float32, BHOut = tl.float32
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bv[:, None] & mask_bk[None, :]
  prev = tl.load(BHPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]),
    mask=mask_kv, other=0.0)
  b_k = tl.load(k + i_bh * $(s_k_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0).to(tl.float32)
  b_v = tl.load(v + i_bh * $(s_v_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0).to(tl.float32)
  b_w = tl.load(w + i_bh * $(s_k_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0).to(tl.float32)
  b_w = tl.exp(b_w)
  b_kv = b_k[None, :] * b_v[:, None]
  acc = prev * b_w[None, :] + b_kv
  if STORE_FINAL_STATE {
  tl.store(BHOut + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]),
    (acc).to(BHOut.dtype.element_ty), mask=mask_kv)
  }
}).surfaceBody

def uAddress {α : Type} (s : State α) (H K BK : Nat) (i : Fin BK) : Nat :=
  (s.pids 2 % H) * K + i.val + s.pids 1 * BK

def keyValue {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (R : RegionName)
    (t stride K BK : Nat) (i : Fin BK) : α :=
  M.cast (some .fp32) .real .real (maskedInput M s R
    (keyAddress s t stride K BK (i,PUnit.unit)) (keyIndex s BK i < K))

def valueValue {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (R : RegionName)
    (t stride V BV : Nat) (i : Fin BV) : α :=
  M.cast (some .fp32) .real .real (maskedInput M s R
    (valueAddress s t stride V BV (i,PUnit.unit)) (valueIndex s BV i < V))

def previousValue {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (R : RegionName)
    (K V BK BV : Nat) (i : TileIndex [BV,BK]) : α :=
  M.fp32Load (maskedInput M s R (stateAddress s K V BK BV i)
    (valueIndex s BV i.1 < V ∧ keyIndex s BK i.2.1 < K))

def stateValues {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (BHPrev k v w : RegionName) (t s_k_h s_v_h K V BK BV : Nat) : TileIndex [BV,BK] → α := fun i =>
  let previous := previousValue M s BHPrev K V BK BV i
  let decay := M.unary (some .fp32) .exp (keyValue M s w t s_k_h K BK i.2.1)
  let kv := M.binary (some .fp32) .real .mul (keyValue M s k t s_k_h K BK i.2.1)
    (valueValue M s v t s_v_h V BV i.1)
  M.cast (some .fp32) .real .real (M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .mul previous decay) kv)

def outputValues {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (BHPrev q k v u : RegionName) (t s_k_h s_v_h H K V BK BV : Nat) (scale : ℝ) : TileIndex [BV] → α :=
  let products : TileIndex [BV,BK] → α := fun i =>
    let kv := M.binary (some .fp32) .real .mul (keyValue M s k t s_k_h K BK i.2.1)
      (valueValue M s v t s_v_h V BV i.1)
    let bias := M.cast (some .fp32) .real .real (maskedInput M s u (uAddress s H K BK i.2.1)
      (keyIndex s BK i.2.1 < K))
    let query := M.binary (some .fp32) .real .mul (keyValue M s q t s_k_h K BK i.2.1)
      (M.literal (some .fp32) .real scale)
    M.binary (some .fp32) .real .mul
      (M.binary (some .fp32) .real .add (previousValue M s BHPrev K V BK BV i)
        (M.binary (some .fp32) .real .mul kv bias)) query
  let sums := M.reduceSum (some .fp32) (shape := [BV,BK]) ⟨1, by simp⟩ Bool.false products
  fun i => M.cast (some .fp32) .real .real (sums i)

specification state_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (BHPrev k v w BHOut : RegionName) (t s_k_h s_v_h K V BK BV : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (stateStep BHPrev k v w BHOut t s_k_h s_v_h K V BK BV) s = some final ∧
      final.mem = (stateScatter s BHOut K V BK BV (stateValues M s BHPrev k v w t s_k_h s_v_h K V BK BV)).mem := by
  simp [stateStep, FP.Structural.exec, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, store, Option.bind, ComputeDType.eraseDType,
    numeric, bop, natLt, numericLt, integralMod, Region.cast, stateValues, stateScatter, stateActive,
    previousValue, keyValue, valueValue, maskedInput, stateAddress, keyAddress, valueAddress,
    keyIndex, valueIndex, and_comm, show (0.0 : ℝ) = 0 by norm_num]
  apply scatter_memory_congr
  rfl

theorem output_memory {α : Type} [Inhabited α] (M : Algebra α)
    (BHPrev q k v u o : RegionName) (t s_k_h s_v_h B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (outputStep BHPrev q k v u o t s_k_h s_v_h B H T K V BK BV scale) s = some final ∧
      final.mem = (scatter s o BV (outputAddress s t s_v_h B H V BV)
        (fun i => valueIndex s BV i.1 < V) (outputValues M s BHPrev q k v u t s_k_h s_v_h H K V BK BV scale)).mem := by
  simp [outputStep, FP.Structural.exec, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, store, Option.bind, ComputeDType.eraseDType,
    numeric, bop, natLt, numericLt, integralMod, Region.cast, outputValues, scatter,
    previousValue, keyValue, valueValue, maskedInput, stateAddress, keyAddress, valueAddress,
    uAddress, outputAddress, keyIndex, valueIndex, show (0.0 : ℝ) = 0 by norm_num]
  apply scatter_memory_congr
  rfl

specification output_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (BHPrev q k v u o : RegionName) (t s_k_h s_v_h B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (outputStep BHPrev q k v u o t s_k_h s_v_h B H T K V BK BV scale) s = some final ∧
      Post s final o BV (outputAddress s t s_v_h B H V BV)
        (fun i => valueIndex s BV i.1 < V) (outputValues M s BHPrev q k v u t s_k_h s_v_h H K V BK BV scale) := by
  obtain ⟨final, he, hm⟩ := output_memory M BHPrev q k v u o t s_k_h s_v_h B H T K V BK BV scale s
  exact ⟨final, he, post_of_memory s final o BV _ _ _ (outputAddress_injective s t s_v_h B H V BV) hm⟩

/-- One actual state update followed by the optional final-state store.
There is no premise equating a precomputed final buffer to an answer. -/
specification conditional_final_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (BHPrev k v w ht : RegionName) (STORE_FINAL_STATE : Bool)
    (t s_k_h s_v_h K V BK BV : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (finalStep BHPrev k v w ht STORE_FINAL_STATE t s_k_h s_v_h K V BK BV) s = some final ∧
      final.mem = (if STORE_FINAL_STATE then
        stateScatter s ht K V BK BV (stateValues M s BHPrev k v w t s_k_h s_v_h K V BK BV) else s).mem := by
  cases STORE_FINAL_STATE
  all_goals simp [finalStep, FP.Structural.exec, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, store, Option.bind, ComputeDType.eraseDType,
    numeric, bop, natLt, numericLt, integralMod, Region.cast, stateValues, stateScatter, stateActive,
    previousValue, keyValue, valueValue, maskedInput, stateAddress, keyAddress, valueAddress,
    keyIndex, valueIndex, and_comm, show (0.0 : ℝ) = 0 by norm_num]
  apply scatter_memory_congr
  rfl

#axiomsClean state_fp32_stage
#axiomsClean output_fp32_stage
#axiomsClean conditional_final_fp32_stage
end
end VeriTile.Bench.TritonBenchG.FusedRwkv6Kernel.FPStages
