import bench.repairs.tritonbench_g.l2_norm_triton2.Correct
import bench.repairs.tritonbench_g.l2_norm_triton1.Misc.TensorBridge

/-! Optional wrapper-check correspondence and ordinary Tensor/view binding facts. -/
namespace VeriTile.Bench.Repairs.L2NormTriton2.TensorBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section

theorem python_checks_iff_preconditions (x dy : ArgumentMetadata) :
    checkInputs x dy = .ok () ↔ AdmittedArguments x dy := checkInputs_iff x dy

theorem forward_checks_iff_preconditions (x : ArgumentMetadata) :
    checkForward x = .ok () ↔ L2NormTriton1.AdmittedArguments x := L2NormTriton1.checkInputs_iff x

theorem contract_memory_meaning (x dy : TensorMetadata) (h : InputContract x dy)
    (input gradient : View) (bx : TensorBinding x input) (bd : TensorBinding dy gradient) :
    DirectView input ∧ DirectView gradient := by
  refine ⟨direct_view x input bx h.contiguous.1 h.noNeg.1 (fun hn => (h.alignment hn).1),?_⟩
  apply direct_view dy gradient bd h.contiguous.2 h.noNeg.2
  intro nonempty
  exact (h.alignment (by simpa only [h.sameCount] using nonempty)).2

/-- Identical forward source, with the same runtime epsilon and checked domain. -/
theorem forward_correctness {M β : Type} (ops : Ops M ℝ β)
    (metadata : TensorMetadata) (h : ForwardContract metadata) (input : View)
    (p : L2NormTriton1.Physical.Layout)
    (binding : L2NormTriton1.Kernel.Buffers metadata input p)
    (eps : ℝ) (before : M) (row : Nat) (hr : row < p.rows) :
    Spec.Real (∃ after,
      FP.Structural.exec L2NormTriton1.RealExecution.algebra (forward p.cols p.block)
        (L2NormTriton1.Tile.rowState row eps
          (L2NormTriton1.Source.state ops L2NormTriton1.Kernel.realAdapter p before)) = some after ∧
      (∀ col, col < p.cols → after.mem "Y" (row*p.cols+col) =
        Cell.mk .real (L2NormTriton1.Kernel.formula ops input p.cols eps before row col)) ∧
      (∀ (r : RegionName) (i : Nat), r ≠ "Y" ∨ (∀ col, col < p.cols → i ≠ row*p.cols+col) →
        after.mem r i =
          (L2NormTriton1.Source.state ops L2NormTriton1.Kernel.realAdapter p before).mem r i)) :=
  L2NormTriton1.TensorBridge.kernel_correctness ops metadata h input p binding eps before row hr

/-- The identical forward launch covers every row and every exterior byte. -/
theorem forward_grid_correctness {M β : Type} (ops : Ops M ℝ β)
    (metadata : TensorMetadata) (h : ForwardContract metadata) (input : View)
    (p : L2NormTriton1.Physical.Layout)
    (binding : L2NormTriton1.Kernel.Buffers metadata input p) (eps : ℝ) (before : M) :
    Spec.Real (L2NormTriton1.Source.Certificates ops L2NormTriton1.Kernel.realAdapter p eps
        (List.range p.rows) before ∧
      (∀ row col, row < p.rows → col < p.cols →
        ops.read (L2NormTriton1.Physical.run ops (L2NormTriton1.Kernel.realAdapter.compute p eps)
          p (List.range p.rows) before) (p.ykey row col) =
          L2NormTriton1.Kernel.formula ops input p.cols eps before row col) ∧
      (∀ owner address,
        (∀ row col, row < p.rows → col < p.cols → ¬(p.ykey row col).covers owner address) →
        ops.observe (L2NormTriton1.Physical.run ops (L2NormTriton1.Kernel.realAdapter.compute p eps)
          p (List.range p.rows) before) owner address = ops.observe before owner address)) :=
  L2NormTriton1.TensorBridge.grid_correctness ops metadata h input p binding eps before

/-- Full backward row, including both complete reductions and the independent
L2 gradient formula. No output or execution-success hypothesis is used. -/
theorem backward_correctness {M β : Type} (ops : Ops M ℝ β)
    (x dy : TensorMetadata) (_h : InputContract x dy) (input gradient : View)
    (p : Physical.Layout) (binding : Kernel.Buffers x dy input gradient p)
    (eps : ℝ) (before : M) (row : Nat) (hr : row < p.rows) :
    Spec.Real (∃ after,
      FP.Structural.exec RealExecution.algebra (backward p.cols p.block)
        (Tile.rowState row eps (Source.state ops Kernel.realAdapter p before)) = some after ∧
      (∀ col, col < p.cols → after.mem "DX" (row*p.cols+col) =
        Cell.mk .real (Kernel.formula ops input gradient p.cols eps before row col)) ∧
      (∀ (r : RegionName) (i : Nat), r ≠ "DX" ∨ (∀ col, col < p.cols → i ≠ row*p.cols+col) →
        after.mem r i = (Source.state ops Kernel.realAdapter p before).mem r i)) :=
  Kernel.real_row ops x dy input gradient p binding eps before row hr

/-- All backward programs have source certificates at their preceding physical
heap; every output element and every exterior byte are covered. -/
theorem backward_grid_correctness {M β : Type} (ops : Ops M ℝ β)
    (x dy : TensorMetadata) (_h : InputContract x dy) (input gradient : View)
    (p : Physical.Layout) (binding : Kernel.Buffers x dy input gradient p)
    (eps : ℝ) (before : M) :
    Spec.Real (Source.Certificates ops Kernel.realAdapter p eps (List.range p.rows) before ∧
      (∀ row col, row < p.rows → col < p.cols →
        ops.read (Physical.run ops (Kernel.realAdapter.compute p eps) p (List.range p.rows) before)
          (p.ykey row col) = Kernel.formula ops input gradient p.cols eps before row col) ∧
      (∀ owner address,
        (∀ row col, row < p.rows → col < p.cols → ¬(p.ykey row col).covers owner address) →
        ops.observe (Physical.run ops (Kernel.realAdapter.compute p eps) p (List.range p.rows) before)
          owner address = ops.observe before owner address)) :=
  Kernel.real_grid ops x dy input gradient p binding eps before

#axiomsClean forward_correctness
#axiomsClean forward_grid_correctness
#axiomsClean backward_correctness
#axiomsClean backward_grid_correctness
#axiomsClean python_checks_iff_preconditions
#axiomsClean forward_checks_iff_preconditions
#axiomsClean contract_memory_meaning
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton2.TensorBridge
