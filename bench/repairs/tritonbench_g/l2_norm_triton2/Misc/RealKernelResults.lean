import bench.repairs.tritonbench_g.l2_norm_triton2.Kernels
import bench.repairs.tritonbench_g.l2_norm_triton2.Proofs.Kernel
import bench.repairs.tritonbench_g.l2_norm_triton1.Misc.RealKernelResults

/-! Complete forward and backward kernels over raw states or physical allocations.
Tensor metadata and application results are ordinary theorems in TensorBridge. -/
namespace VeriTile.Bench.Repairs.L2NormTriton2.RealKernelResults
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section

/-- Complete row execution, independent L2 formula and exterior cell frame.
Only the raw source row, scalar epsilon and reduction-width coverage are needed. -/
theorem forward_correctness (N B row : Nat) (cover : N≤B) (eps : ℝ) (before : State ℝ) :
    Spec.Real (∃ after,
      FP.Structural.exec L2NormTriton1.RealExecution.algebra (forward N B) (L2NormTriton1.Tile.rowState row eps before)=some after ∧
      (∀ col,col<N → after.mem "Y" (row*N+col)=
        Cell.mk .real (L2NormTriton1.RealExecution.formula N eps (L2NormTriton1.Tile.inputs before N row) col)) ∧
      (∀ (r : RegionName) (i : Nat),r≠"Y" ∨ (∀ col,col<N → i≠row*N+col) →
        after.mem r i=before.mem r i)) :=
  L2NormTriton1.RealKernelResults.row_correct N B row cover eps before

/-- Every row executes at its preceding physical heap. Raw allocation capacity
and separation give complete output values and the exterior-byte frame. -/
theorem forward_grid_correctness {M β : Type} (ops : Ops M ℝ β)
    (p : L2NormTriton1.Physical.Layout) (valid : L2NormTriton1.Physical.Valid p) (eps : ℝ) (before : M) :
    Spec.Real (L2NormTriton1.Source.Certificates ops L2NormTriton1.Kernel.realAdapter p eps (List.range p.rows) before ∧
      (∀ row col,row<p.rows → col<p.cols →
        ops.read (L2NormTriton1.Physical.run ops (L2NormTriton1.Kernel.realAdapter.compute p eps) p (List.range p.rows) before)
          (p.ykey row col)=L2NormTriton1.RealExecution.formula p.cols eps (L2NormTriton1.Physical.rowInputs ops p before row) col) ∧
      (∀ owner address,
        (∀ row col,row<p.rows → col<p.cols → ¬(p.ykey row col).covers owner address) →
        ops.observe (L2NormTriton1.Physical.run ops (L2NormTriton1.Kernel.realAdapter.compute p eps) p (List.range p.rows) before)
          owner address=ops.observe before owner address)) :=
  L2NormTriton1.RealKernelResults.grid_correct ops p valid eps before

/-- Full backward row execution, independent gradient formula and full cell frame. -/
theorem backward_correctness (N B row : Nat) (cover : N≤B)
    (eps : ℝ) (before : State ℝ) :
    Spec.Real (∃ after,
      FP.Structural.exec RealExecution.algebra (backward N B) (Tile.rowState row eps before)=some after ∧
      (∀ col,col<N → after.mem "DX" (row*N+col)=
        Cell.mk .real (RealExecution.formula N eps
          (Tile.inputs before "X" N row) (Tile.inputs before "DY" N row) col)) ∧
      (∀ (r : RegionName) (i : Nat),r≠"DX" ∨ (∀ col,col<N → i≠row*N+col) →
        after.mem r i=before.mem r i)) := by
  refine ⟨(Tile.sourceRun RealExecution.algebra N B row eps before).val,
    (Tile.sourceRun RealExecution.algebra N B row eps before).property,?_,?_⟩
  · intro col hc
    rw [Tile.readback _ _ _ _ _ _ (⟨⟨col,lt_of_lt_of_le hc cover⟩,PUnit.unit⟩) hc,
      RealExecution.scalar_real N B eps _ _ col cover]
  · intro r i outside
    apply Tile.frame
    exact outside.imp id (fun h lane active => h lane.1.val active)

/-- The full backward grid executes from raw physical allocations, preserving
all exterior bytes and computing the independent gradient for every row. -/
theorem backward_grid_correctness {M β : Type} (ops : Ops M ℝ β)
    (p : Physical.Layout) (valid : Physical.Valid p) (eps : ℝ) (before : M) :
    Spec.Real (Source.Certificates ops Kernel.realAdapter p eps (List.range p.rows) before ∧
      (∀ row col,row<p.rows → col<p.cols →
        ops.read (Physical.run ops (Kernel.realAdapter.compute p eps) p (List.range p.rows) before)
          (p.ykey row col)=RealExecution.formula p.cols eps
            (Physical.rowInputs ops p before row Bool.false)
            (Physical.rowInputs ops p before row Bool.true) col) ∧
      (∀ owner address,
        (∀ row col,row<p.rows → col<p.cols → ¬(p.ykey row col).covers owner address) →
        ops.observe (Physical.run ops (Kernel.realAdapter.compute p eps) p (List.range p.rows) before)
          owner address=ops.observe before owner address)) := by
  refine ⟨Source.certificates ops Kernel.realAdapter p valid eps (List.range p.rows)
    (fun row member => List.mem_range.mp member) before,?_,?_⟩
  · intro row col hr hc
    rw [Physical.run_values ops (Kernel.realAdapter.compute p eps) p valid (List.range p.rows)
      (fun row member => List.mem_range.mp member) List.nodup_range row col
      (List.mem_range.mpr hr) hc before]
    dsimp only [Source.Adapter.compute,Kernel.realAdapter,Kernel.identityAdapter,id]
    exact RealExecution.scalar_real _ _ _ _ _ _ valid.main.blockCovers
  · intro owner address outside
    exact Physical.run_bytes ops (Kernel.realAdapter.compute p eps) p (List.range p.rows) before
      owner address (fun row member col hc => outside row col (List.mem_range.mp member) hc)

theorem kernel_address_bounds (p : Physical.Layout) (valid : Physical.Valid p)
    (row lane : Nat) (hr : row < p.rows) (hl : lane < p.block) :
    row*p.cols+lane < 2^63 := Physical.padded_index p valid row lane hr hl

#axiomsClean forward_correctness
#axiomsClean forward_grid_correctness
#axiomsClean backward_correctness
#axiomsClean backward_grid_correctness
#axiomsClean kernel_address_bounds
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton2.RealKernelResults
