import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Grid
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.KernelSafety
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 40000000
set_option maxRecDepth 64000
set_option linter.unusedSimpArgs false

private theorem fold_regs {α ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active] (r : RegionName)
    (off : ι → Nat) (v : ι → Cell α) (s : State α) :
    (xs.foldl (fun t i => if active i then t.write r (off i) (v i) else t) s).regs = s.regs := by
  induction xs generalizing s with
  | nil => rfl
  | cons i xs ih => simp only [List.foldl_cons, ih]; split <;> rfl

/-- Bounds are precisely the active coordinate and eleven-channel output
windows. Safety is proved for the actual complete source at any program ID. -/
theorem forward {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α)
    (fit : p.columnOffset + 11 ≤ p.outputWidth)
    (coords : ∀ r, r < p.rows → ∀ c : Fin 3, r*3+c.val < bounds coordinateRegion)
    (outputs : ∀ r, r < p.rows → ∀ c : Fin 11,
      r*p.outputWidth+p.columnOffset+c.val < bounds outputRegion) :
    TraceSafe M bounds (forwardKernel p) s := by
  simp (config := {maxSteps := 8000000}) [TraceSafe, forwardKernel, Original.kernel,
    VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP.kernel,
    VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP.source, forwardParameters,
    ComputeKernel.surfaceBody, Safety.traceSafeList_unfold, Safety.stepSafe_unfold,
    Safety.safeExpr_unfold, Safety.safeCompute_unfold, Safety.safeOp_unfold,
    Safety.safeAddress_unfold, Safety.safeMask_unfold, Safety.addressSafe_unfold,
    Safety.maskActive_unfold, step, evalExpr, evalComputeOp, evalOp_unfold,
    State.setReg, store, ofFloat, toFloat, Region.cast, ComputeDType.eraseDType,
    numeric, numericLt, natLt, bop, Option.bind, fold_regs]
  have hc (c : Fin 3) (a : Fin p.blockSize)
      (active : a.val*3+p.blockSize*3*s.pids 0+c.val < 3*p.rows) :
      a.val*3+p.blockSize*3*s.pids 0+c.val < bounds coordinateRegion := by
    have eq : a.val*3+p.blockSize*3*s.pids 0+c.val = (a.val+p.blockSize*s.pids 0)*3+c.val := by ring
    rw [eq] at active ⊢
    exact coords _ (by omega) c
  have ho (c : Fin 11) (a : Fin p.blockSize)
      (active : a.val*p.outputWidth+p.blockSize*p.outputWidth*s.pids 0+p.columnOffset+c.val < p.outputWidth*p.rows) :
      a.val*p.outputWidth+p.blockSize*p.outputWidth*s.pids 0+p.columnOffset+c.val < bounds outputRegion := by
    have eq : a.val*p.outputWidth+p.blockSize*p.outputWidth*s.pids 0+p.columnOffset+c.val =
        (a.val+p.blockSize*s.pids 0)*p.outputWidth+p.columnOffset+c.val := by ring
    rw [eq] at active ⊢
    exact outputs _ ((Grid.row_mask _ _ _ _ _ fit c.isLt).mp active) c
  exact ⟨hc 0, hc 1, hc 2, ho 0, ho 1, ho 2, ho 3, ho 4, ho 5, ho 6, ho 7, ho 8, ho 9, ho 10⟩

/-- Backward safety includes all eleven strided upstream channels and both
reads and writes of the in-place prepared gradient. -/
theorem backward {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α)
    (fit : p.columnOffset + 11 ≤ p.outputWidth)
    (coords : ∀ r, r < p.rows → ∀ c : Fin 3, r*3+c.val < bounds coordinateRegion)
    (upstream : ∀ r, r < p.rows → ∀ c : Fin 11,
      r*p.upstreamRowStride+(p.columnOffset+c.val)*p.upstreamColStride < bounds upstreamRegion)
    (gradient : ∀ r, r < p.rows → ∀ c : Fin 3, r*3+c.val < bounds gradientRegion) :
    TraceSafe M bounds (backwardKernel p) s := by
  simp (config := {maxSteps := 8000000}) [TraceSafe, backwardKernel, backwardSource,
    ComputeKernel.surfaceBody, Safety.traceSafeList_unfold, Safety.stepSafe_unfold,
    Safety.safeExpr_unfold, Safety.safeCompute_unfold, Safety.safeOp_unfold,
    Safety.safeAddress_unfold, Safety.safeMask_unfold, Safety.addressSafe_unfold,
    Safety.maskActive_unfold, step, evalExpr, evalComputeOp, evalOp_unfold,
    State.setReg, store, ofFloat, toFloat, Region.cast, ComputeDType.eraseDType,
    numeric, numericLt, natLt, bop, Option.bind, fold_regs]
  have bounded (region : RegionName)
      (h : ∀ r, r < p.rows → ∀ c : Fin 3, r*3+c.val < bounds region)
      (c : Fin 3) (a : Fin p.blockSize)
      (active : a.val*3+p.blockSize*3*s.pids 0+c.val < 3*p.rows) :
      a.val*3+p.blockSize*3*s.pids 0+c.val < bounds region := by
    have eq : a.val*3+p.blockSize*3*s.pids 0+c.val = (a.val+p.blockSize*s.pids 0)*3+c.val := by ring
    rw [eq] at active ⊢
    exact h _ (by omega) c
  have hu (c : Fin 11) (a : Fin p.blockSize)
      (active : a.val*p.outputWidth+p.blockSize*p.outputWidth*s.pids 0+p.columnOffset+c.val < p.outputWidth*p.rows) :
      (a.val+p.blockSize*s.pids 0)*p.upstreamRowStride+p.columnOffset*p.upstreamColStride+c.val*p.upstreamColStride < bounds upstreamRegion := by
    have eq : a.val*p.outputWidth+p.blockSize*p.outputWidth*s.pids 0+p.columnOffset+c.val =
        (a.val+p.blockSize*s.pids 0)*p.outputWidth+p.columnOffset+c.val := by ring
    rw [eq] at active
    have h := upstream _ ((Grid.row_mask _ _ _ _ _ fit c.isLt).mp active) c
    convert h using 1 <;> ring
  repeat' constructor
  all_goals first
    | exact bounded coordinateRegion coords 0
    | exact bounded coordinateRegion coords 1
    | exact bounded coordinateRegion coords 2
    | simpa only [Fin.val_zero, Nat.zero_mul, Nat.add_zero] using hu 0
    | simpa only [Fin.val_one, Nat.one_mul] using hu 1
    | exact hu 2
    | exact hu 3
    | exact hu 4
    | exact hu 5
    | exact hu 6
    | exact hu 7
    | exact hu 8
    | exact hu 9
    | exact hu 10
    | exact bounded gradientRegion gradient 0
    | exact bounded gradientRegion gradient 1
    | exact bounded gradientRegion gradient 2

#axiomsClean forward
#axiomsClean backward

end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.KernelSafety
