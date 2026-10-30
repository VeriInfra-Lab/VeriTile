import bench.tritonbench_g.ksoftmax_triton.Proofs.Mathematical_ForwardTypedModel

/-! Regression draft for the unapplied shared provenance patch. Before the
patch the branched positive cases must fail; they use nonzero fp16 cells.
Conditions include runtime register booleans and the original Option String
mapping. No unbound arm is declared unreachable by the tests or expander.
-/
namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.PointerRegression
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 800000

def branched (M Y : RegionName) (mode : Option String) : ComputeKernel :=
  let MASK_QK := mode == some "qk"
  let MASK_BK := mode == some "bk"
  triton {
    tl.region M = tl.float16, Y = tl.float16
    k = tl.arange(0, 1)
    if MASK_QK {
      mask_ptrs = M + 13 + k
    } elif MASK_BK {
      mask_ptrs = M + 26 + k
    }
    value = tl.load(mask_ptrs)
    tl.store(Y + k, (value).to(tl.float16))
  }

def nested (M Y : RegionName) (mode : Option String) : ComputeKernel :=
  let MASK_QK := mode == some "qk"
  let MASK_BK := mode == some "bk"
  triton {
    tl.region M = tl.float16, Y = tl.float16
    k = tl.arange(0, 1)
    if MASK_QK {
      mask_ptrs = M + 13 + k
    } else {
      if MASK_BK {
        mask_ptrs = M + 26 + k
      }
    }
    value = tl.load(mask_ptrs)
    tl.store(Y + k, (value).to(tl.float16))
  }

theorem elif_is_nested (M Y : RegionName) (mode : Option String) :
    branched M Y mode = nested M Y mode := by rfl

def initial : State (WithBot ℝ) where
  mem := fun r o => Cell.mk .fp16 (some (if r = "M" then
    (if o = 13 then (5 : ℝ) else 11) else 7))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun dtype _ _ => defaultValue dtype

theorem qk_nonzero : ∃ t,
    run (engine TypedExecution.model) (branched "M" "Y" (some "qk")).surfaceBody initial = some t ∧
    t.mem "Y" 0 = (Cell.mk .fp16 (some (5 : ℝ)) : Cell (WithBot ℝ)) := by
  have hYM : ("Y" : String) ≠ "M" := by decide
  simp [branched, ComputeKernel.surfaceBody, run, step,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    engine, Algebra.withDefaultPrecision, resolvePrecision, TypedExecution.model,
    ComputeDType.eraseDType, numeric, bop, initial, Cell.read, ofFloat, toFloat,
    Region.cast, State.setReg, FP.Structural.store, State.write,
    TileShape.allIndices, List.finRange_succ, hYM]
  all_goals rfl

theorem bk_nonzero : ∃ t,
    run (engine TypedExecution.model) (branched "M" "Y" (some "bk")).surfaceBody initial = some t ∧
    t.mem "Y" 0 = (Cell.mk .fp16 (some (11 : ℝ)) : Cell (WithBot ℝ)) := by
  have hYM : ("Y" : String) ≠ "M" := by decide
  simp [branched, ComputeKernel.surfaceBody, run, step,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    engine, Algebra.withDefaultPrecision, resolvePrecision, TypedExecution.model,
    ComputeDType.eraseDType, numeric, bop, initial, Cell.read, ofFloat, toFloat,
    Region.cast, State.setReg, FP.Structural.store, State.write,
    TileShape.allIndices, List.finRange_succ, hYM]
  all_goals rfl

theorem unknown_mode_fails :
    run (engine TypedExecution.model) (branched "M" "Y" (some "unknown")).surfaceBody initial = none := by
  simp [branched, ComputeKernel.surfaceBody, run, step,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    engine, Algebra.withDefaultPrecision, resolvePrecision, initial, State.setReg]

/-- The condition is a register expression, so both arms remain runtime paths. -/
def runtimePartial (M Y : RegionName) (flag : Bool) : ComputeKernel := triton {
  tl.region M = tl.float16, Y = tl.float16
  k = tl.arange(0, 1)
  take_mask = 0 < $(if flag then (1 : Nat) else 0)
  if take_mask {
    mask_ptrs = M + 13 + k
  }
  value = tl.load(mask_ptrs)
  tl.store(Y + k, (value).to(tl.float16))
}

theorem runtime_missing_fails :
    run (engine TypedExecution.model) (runtimePartial "M" "Y" Bool.false).surfaceBody initial = none := by
  simp [runtimePartial, ComputeKernel.surfaceBody, run, step,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    engine, Algebra.withDefaultPrecision, resolvePrecision, initial, State.setReg,
    natLt, bop]

/-- Metadata retains both possible source buffers. It does not invent the
branch-local pointer name as an additional external region argument. -/
def twoSources (A B Y : RegionName) (first second : Bool) : ComputeKernel := triton {
  tl.region A = tl.float16, B = tl.float16, Y = tl.float16
  k = tl.arange(0, 1)
  if first { p = A + 13 + k } elif second { p = B + 26 + k }
  value = tl.load(p)
  tl.store(Y + k, (value).to(tl.float16))
}

theorem branch_region_inventory (A B Y : RegionName) (first second : Bool) :
    (twoSources A B Y first second).inputs = [A, B] ∧
    (twoSources A B Y first second).outputs = [Y] := by
  exact ⟨rfl, rfl⟩

def wideJoined (M Y : RegionName) (flag : Bool) : ComputeKernel := triton {
  tl.region M = tl.float64, Y = tl.float64
  if flag { p = M + 13 } else { p = M + 26 }
  value = tl.load(p)
  doubled = value + value
  tl.store(Y, (doubled).to(tl.float64))
}

def wideExplicit (M Y : RegionName) (flag : Bool) : ComputeKernel := triton {
  tl.region M = tl.float64, Y = tl.float64
  if flag { p = M + 13 } else { p = M + 26 }
  value = tl.load(p, dtype=tl.float64)
  doubled = value + value
  tl.store(Y, (doubled).to(tl.float64))
}

theorem wide_all_arms_precision (M Y : RegionName) (flag : Bool) :
    wideJoined M Y flag = wideExplicit M Y flag := by rfl

/-- error: pointer storage dtype conflict across branches: p -/
#guard_msgs in
def conflicting (A B Y : RegionName) (first second : Bool) : ComputeKernel := triton {
  tl.region A = tl.float16, B = tl.bfloat16, Y = tl.float16
  if first { p = A + 0 } elif second { p = B + 0 }
  value = tl.load(p)
  tl.store(Y, (value).to(tl.float16))
}

/-- error: fp64 pointer storage is not established on every branch: p -/
#guard_msgs in
def missingWide (M Y : RegionName) (flag : Bool) : ComputeKernel := triton {
  tl.region M = tl.float64, Y = tl.float64
  if flag { p = M + 0 }
  value = tl.load(p)
  tl.store(Y, (value).to(tl.float64))
}

#axiomsClean elif_is_nested
#axiomsClean qk_nonzero
#axiomsClean bk_nonzero
#axiomsClean unknown_mode_fails
#axiomsClean runtime_missing_fails
#axiomsClean branch_region_inventory
#axiomsClean wide_all_arms_precision
end
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.PointerRegression
