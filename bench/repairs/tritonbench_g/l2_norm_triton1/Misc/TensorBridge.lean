import bench.repairs.tritonbench_g.l2_norm_triton1.Correct

/-! Optional Python-check and physical input-binding correspondence. -/
namespace VeriTile.Bench.Repairs.L2NormTriton1.TensorBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section

/-- Full active row, independent x/sqrt(sum(x²)+eps), successful complete source
execution and the complete exterior logical-memory frame. -/
theorem kernel_correctness {M β : Type} (ops : Ops M ℝ β)
    (metadata : TensorMetadata) (_h : InputContract metadata) (input : View)
    (p : Physical.Layout) (binding : Kernel.Buffers metadata input p)
    (eps : ℝ) (before : M) (row : Nat) (hr : row < p.rows) :
    Spec.Real (∃ after,
      FP.Structural.exec RealExecution.algebra (source p.cols p.block)
        (Tile.rowState row eps (Source.state ops Kernel.realAdapter p before)) = some after ∧
      (∀ col, col < p.cols → after.mem "Y" (row*p.cols+col) =
        Cell.mk .real (Kernel.formula ops input p.cols eps before row col)) ∧
      (∀ (r : RegionName) (i : Nat), r ≠ "Y" ∨ (∀ col, col < p.cols → i ≠ row*p.cols+col) →
        after.mem r i = (Source.state ops Kernel.realAdapter p before).mem r i)) :=
  Kernel.real_row ops metadata input p binding eps before row hr

/-- Every row is certified at its actual preceding physical heap, and the whole
output and every byte outside it follow without any host-orchestration claim. -/
theorem grid_correctness {M β : Type} (ops : Ops M ℝ β)
    (metadata : TensorMetadata) (_h : InputContract metadata) (input : View)
    (p : Physical.Layout) (binding : Kernel.Buffers metadata input p)
    (eps : ℝ) (before : M) :
    Spec.Real (Source.Certificates ops Kernel.realAdapter p eps (List.range p.rows) before ∧
      (∀ row col, row < p.rows → col < p.cols →
        ops.read (Physical.run ops (Kernel.realAdapter.compute p eps) p (List.range p.rows) before)
          (p.ykey row col) = Kernel.formula ops input p.cols eps before row col) ∧
      (∀ owner address,
        (∀ row col, row < p.rows → col < p.cols → ¬(p.ykey row col).covers owner address) →
        ops.observe (Physical.run ops (Kernel.realAdapter.compute p eps) p (List.range p.rows) before)
          owner address = ops.observe before owner address)) :=
  Kernel.real_grid ops metadata input p binding eps before

theorem python_checks_iff_preconditions (argument : ArgumentMetadata) :
    checkInputs argument = .ok () ↔ AdmittedArguments argument := checkInputs_iff argument

theorem contract_memory_meaning (metadata : TensorMetadata) (h : InputContract metadata)
    (input : View) (binding : TensorBinding metadata input) : DirectInput input :=
  direct_input metadata input h binding

#axiomsClean kernel_correctness
#axiomsClean grid_correctness
#axiomsClean python_checks_iff_preconditions
#axiomsClean contract_memory_meaning
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton1.TensorBridge
