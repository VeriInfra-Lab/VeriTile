import bench.tritonbench_g.rms_norm_triton.Kernels
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

structure Layout where
  X : RegionName
  W : RegionName
  Y : RegionName
  M : Nat
  N : Nat
  B : Nat
  complete : N ≤ B

def Layout.source (c : Layout) (eps : ℝ) : ComputeKernel :=
  kernel c.Y c.X c.W c.N 1 c.N 1 c.N eps c.B

def rowState (pid : Nat) (s : State α) : State α :=
  { s with pids := fun ax => if ax = 0 then pid else 0 }

/-- Actual load, zero fill and explicit input FP32 cast; no row result oracle. -/
def inputTile [Inhabited α] (mdl : Algebra α) (c : Layout)
    (xs : Nat → α) : Values α .real [c.B] := fun j =>
  mdl.cast (some .fp32) .real .real
    (if j.1.val < c.N then xs j.1.val else mdl.literal (some .fp32) .real 0.0)

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
  let variance := mdl.binary (some .fp32) .real .div total (mdl.literal (some .fp32) .real c.N)
  let epsilon := mdl.cast (some .fp32) .real .real (mdl.literal (some .fp32) .real eps)
  let rrms := mdl.binary (some .fp32) .real .div (mdl.literal (some .fp32) .real 1)
    (mdl.unary (some .fp32) .sqrt (mdl.binary (some .fp32) .real .add variance epsilon))
  mdl.binary (some .fp32) .real .mul
    (mdl.cast (some .fp32) .real .real (mdl.binary (some .fp32) .real .mul
      (mdl.cast (some .fp32) .real .real (xs i)) rrms))
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

def sourceRun [Inhabited α] (mdl : Algebra α) (c : Layout) (eps : ℝ)
    (pid : Nat) (s : State α) :
    { t : State α // run (engine mdl) (c.source eps).surfaceBody (rowState pid s) = some t } := by
  refine ⟨?_, ?_⟩
  swap
  simp (maxSteps := 500000) [Layout.source, kernel, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, store, Option.bind,
    engine, Algebra.withDefaultPrecision, resolvePrecision, ComputeDType.eraseDType,
    numeric, bop, natLt, rowState, ofFloat, toFloat,
    TileShape.axisDim, Region.cast, State.setReg]
  try rfl

theorem row_readback [Inhabited α] (mdl : Algebra α) (c : Layout) (eps : ℝ)
    (pid : Nat) (s : State α) (i : Fin c.B) (hi : i.val < c.N) :
    (sourceRun mdl c eps pid s).val.mem c.Y (pid*c.N+i.val) =
      Cell.mk .real (rowValue mdl c eps
        (fun j => (s.mem c.X (pid*c.N+j)).read .real)
        (fun j => (s.mem c.W j).read .real) i.val) := by
  simp only [sourceRun, store]
  have injective : Function.Injective (fun j : TileIndex [c.B] => pid*c.N+j.1.val) := by
    intro a b h
    dsimp only at h
    apply Prod.ext
    · apply Fin.ext; omega
    · exact Subsingleton.elim _ _
  erw [State.masked_scatter_readback (r := c.Y)
    (active := fun j : TileIndex [c.B] => j.1.val < c.N)
    (i := (i,PUnit.unit)) (hi := hi) (hinj := injective)]
  simp only [rowValue,inputTile,weightTile,hi,if_true,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,State.setReg,rowState,
    Nat.zero_add,Nat.mul_one]
  rfl

def Outside (c : Layout) (pid : Nat) (region : RegionName) (a : Nat) : Prop :=
  region ≠ c.Y ∨ ∀ i : Fin c.B, i.val < c.N → a ≠ pid*c.N+i.val

theorem row_frame [Inhabited α] (mdl : Algebra α) (c : Layout) (eps : ℝ)
    (pid : Nat) (s : State α) (region : RegionName) (a : Nat)
    (outside : Outside c pid region a) :
    (sourceRun mdl c eps pid s).val.mem region a = s.mem region a := by
  simp only [sourceRun,store]
  rw [State.masked_scatter_frame]
  rcases outside with hr | ho
  · exact Or.inl hr
  · exact Or.inr (fun i _ hi => ho i.1 hi)

end
#guard_msgs (drop info) in
#specNonCircular rowValue avoiding [Layout.source,sourceRun]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole
