import bench.tritonbench_g.rms_norm_triton.Proofs.ActualRow

namespace VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.RmsNormTriton
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def Layout.source (c : Layout) (eps : ℝ) : ComputeKernel :=
  VeriTile.Bench.TritonBenchG.RmsNormTriton.kernel c.Y c.X c.W c.N 1 c.N 1 c.N eps c.B

/-- Actual load, zero fill and explicit input FP32 cast; no row result oracle. -/
def inputTile [Inhabited α] (mdl : Algebra α) (c : Layout)
    (xs : Nat → α) : Values α .real [c.B] := fun j =>
  mdl.cast (some .fp32) .real .real
    (mdl.fp32Load (if j.1.val < c.N then xs j.1.val else mdl.literal (some .fp32) .real 0.0))

def weightTile [Inhabited α] (mdl : Algebra α) (c : Layout)
    (ws : Nat → α) : Values α .real [c.B] := fun j =>
  mdl.fp32Load (if j.1.val < c.N then ws j.1.val else mdl.literal (some .fp32) .real 0.0)

/-- Independent ordered operation graph. The actual reduction callback is
retained; no reassociation, split FMA identity or exact IEEE/Real claim occurs. -/
def rowValue [Inhabited α] (mdl : Algebra α) (c : Layout) (eps : ℝ)
    (xs ws : Nat → α) (i : Nat) : α :=
  let loaded := inputTile mdl c xs
  let squared := fun j => mdl.binary (some .fp32) .real .mul (loaded j) (loaded j)
  let total := mdl.reduceSum (some .fp32) (shape := [c.B]) ⟨0, by simp⟩ Bool.false squared PUnit.unit
  let variance := mdl.binary (some .fp32) .real .div total (mdl.fromNat (some .fp32) c.N)
  let epsilon := mdl.cast (some .fp32) .real .real (mdl.literal (some .fp32) .real eps)
  let rrms := mdl.binary (some .fp32) .real .div (mdl.literal (some .fp32) .real 1)
    (mdl.unary (some .fp32) .sqrt (mdl.binary (some .fp32) .real .add variance epsilon))
  mdl.binary (some .fp32) .real .mul
    (mdl.cast (some .fp32) .real .real (mdl.binary (some .fp32) .real .mul
      (mdl.cast (some .fp32) .real .real (mdl.fp32Load (xs i))) rrms))
    (mdl.fp32Load (ws i))

theorem rowValue_congr [Inhabited α] (mdl : Algebra α) (c : Layout) (eps : ℝ)
    (xs ws xs' ws' : Nat → α) (i : Nat) (hi : i < c.N)
    (hx : ∀ j, j < c.N → xs j = xs' j) (hw : ∀ j, j < c.N → ws j = ws' j) :
    rowValue mdl c eps xs ws i = rowValue mdl c eps xs' ws' i := by
  have tile : inputTile mdl c xs = inputTile mdl c xs' := by
    funext j
    unfold inputTile
    by_cases h : j.1.val < c.N
    · rw [if_pos h,if_pos h,hx j.1.val h]
    · rw [if_neg h,if_neg h]
  simp only [rowValue,tile,hx i hi,hw i hi]

theorem literal_run [Inhabited α] (mdl : Algebra α) (c : Layout) (eps : ℝ)
    (pid : Nat) (s : State α) :
    run (engine mdl) (c.source eps).surfaceBody (rowState pid s) =
      run (engine mdl) (ActualRow.source c).surfaceBody
        (ActualRow.entry (ActualRow.argument mdl eps) pid s) := by
  change run (engine mdl)
    ([ScalarArgument.statement (epsilonArgument eps)] ++ (ActualRow.source c).surfaceBody)
    (rowState pid s) = _
  rw [run_append]
  simp [run,step,evalExpr,evalComputeOp,evalOp_unfold,ScalarArgument.statement,
    epsilonArgument,ActualRow.argument,ActualRow.entry,engine,Algebra.withDefaultPrecision,
    resolvePrecision,ofFloat,toFloat]

def sourceRun [Inhabited α] (mdl : Algebra α) (c : Layout) (eps : ℝ)
    (pid : Nat) (s : State α) :
    { t : State α // run (engine mdl) (c.source eps).surfaceBody (rowState pid s) = some t } :=
  ⟨(ActualRow.sourceRun mdl c (ActualRow.argument mdl eps) pid s).val,
    (literal_run mdl c eps pid s).trans (ActualRow.sourceRun mdl c (ActualRow.argument mdl eps) pid s).property⟩

theorem row_readback [Inhabited α] (mdl : Algebra α) (c : Layout) (eps : ℝ)
    (pid : Nat) (s : State α) (i : Fin c.B) (hi : i.val < c.N) :
    (sourceRun mdl c eps pid s).val.mem c.Y (pid*c.N+i.val) =
      Cell.mk .real (rowValue mdl c eps
        (fun j => (s.mem c.X (pid*c.N+j)).read .real)
        (fun j => (s.mem c.W j).read .real) i.val) := by
  simpa only [sourceRun,rowValue,ActualRow.rowValue,inputTile,ActualRow.inputTile,
    ActualRow.argument] using ActualRow.row_readback mdl c (ActualRow.argument mdl eps) pid s i hi

def Outside (c : Layout) (pid : Nat) (region : RegionName) (a : Nat) : Prop :=
  region ≠ c.Y ∨ ∀ i : Fin c.B, i.val < c.N → a ≠ pid*c.N+i.val

theorem row_frame [Inhabited α] (mdl : Algebra α) (c : Layout) (eps : ℝ)
    (pid : Nat) (s : State α) (region : RegionName) (a : Nat)
    (outside : Outside c pid region a) :
    (sourceRun mdl c eps pid s).val.mem region a = s.mem region a :=
  ActualRow.row_frame mdl c (ActualRow.argument mdl eps) pid s region a outside

#axiomsClean literal_run
#axiomsClean sourceRun
#axiomsClean row_readback
#axiomsClean row_frame
#specNonCircular rowValue avoiding [Layout.source,sourceRun]
end
end VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole
