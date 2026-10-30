import bench.tritonbench_g.fused_recurrent_delta.Proofs.FPStages
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentDelta.FPStages
open VeriTile.Triton FP.Structural
open FusedRecurrentHgrn.FPStages (scatter_memory_congr)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 15000
set_option linter.unusedSimpArgs false

/-- Selected state update before the enclosing recurrent loop. The beta flag
retains both original source branches; the residual is computed from the
pre-write v input, not a rounded intermediate store. -/
def stateStep (HPrev k v beta HOut : RegionName) (headwise : Bool)
    (t s_qk_h s_vo_h T K V BK BV : Nat) : ComputeKernel :=
  ComputeKernel.mk [HPrev,k,v,beta] [HOut] (triton {
  tl.region HPrev = tl.float32, k = tl.float32, v = tl.float32, beta = tl.float32, HOut = tl.float32
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bk[None, :] & mask_bv[:, None]
  prev = tl.load(HPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]),
    mask=mask_kv, other=0.0)
  b_k = tl.load(k + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0).to(tl.float32)
  b_v = tl.load(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0).to(tl.float32)
  _v_minus = tl.sum(prev * b_k[None, :], axis=1)
  b_v -= _v_minus
  if headwise {
    b_beta = tl.load(beta + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
      mask=mask_bv, other=0.0).to(tl.float32)
  } else {
    b_beta = tl.load(beta + i_bh * $(T) + $(t)).to(tl.float32)
  }
  b_v *= b_beta
  acc = prev + b_k[None, :] * b_v[:, None]
  tl.store(HOut + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]),
    (acc).to(HOut.dtype.element_ty), mask=mask_kv)
}).surfaceBody

abbrev stateActive {α : Type} (s : State α) (K V BK BV : Nat) (i : TileIndex [BV,BK]) : Prop :=
  keyIndex s BK i.2.1 < K ∧ valueIndex s BV i.1 < V

def stateValues {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (HPrev k v beta : RegionName) (headwise : Bool) (t s_qk_h s_vo_h T K V BK BV : Nat) :
    TileIndex [BV,BK] → α := fun i =>
  let old := M.fp32Load (maskedInput M s HPrev (stateAddress s K V BK BV i)
    (stateActive s K V BK BV i))
  let key := M.cast (some .fp32) .real .real
    (maskedInput M s k (keyAddress s t s_qk_h K BK i.2) (keyIndex s BK i.2.1 < K))
  let b := if headwise then M.cast (some .fp32) .real .real
    (maskedInput M s beta (valueAddress s t s_vo_h V BV (i.1,PUnit.unit)) (valueIndex s BV i.1 < V))
    else M.cast (some .fp32) .real .real ((s.mem beta (s.pids 2 * T + t)).read .real)
  let residual := residualValues M s HPrev k v t s_qk_h s_vo_h K V BK BV (i.1,PUnit.unit)
  M.cast (some .fp32) .real .real
    (M.binary (some .fp32) .real .add old
      (M.binary (some .fp32) .real .mul key (M.binary (some .fp32) .real .mul residual b)))

def stateScatter {α : Type} (s : State α) (HOut : RegionName) (K V BK BV : Nat)
    (values : TileIndex [BV,BK] → α) : State α :=
  (TileShape.allIndices [BV,BK]).foldl
    (fun u i => if stateActive s K V BK BV i then
      u.write HOut (stateAddress s K V BK BV i) (.mk .real (values i)) else u) s

specification state_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (HPrev k v beta HOut : RegionName) (headwise : Bool)
    (t s_qk_h s_vo_h T K V BK BV : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M
      (stateStep HPrev k v beta HOut headwise t s_qk_h s_vo_h T K V BK BV) s = some final ∧
      final.mem = (stateScatter s HOut K V BK BV
        (stateValues M s HPrev k v beta headwise t s_qk_h s_vo_h T K V BK BV)).mem := by
  cases headwise
  all_goals simp [stateStep, FP.Structural.exec, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, store, Option.bind, ComputeDType.eraseDType,
    numeric, bop, natLt, numericLt, Region.cast, stateValues, stateActive,
    residualValues, readoutValues, maskedInput, stateScatter,
    valueAddress, stateAddress, keyAddress, keyIndex, valueIndex,
    show (0.0 : ℝ) = 0 by norm_num]
  all_goals apply scatter_memory_congr
  all_goals rfl

#axiomsClean state_fp32_stage
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentDelta.FPStages
