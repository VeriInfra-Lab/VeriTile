import bench.tritonbench_g.l2_norm_triton1.Proofs.Kernel
import bench.tritonbench_g.l2_norm_triton2.Proofs.Contract
import bench.tritonbench_g.l2_norm_triton2.Proofs.Source
import bench.tritonbench_g.l2_norm_triton2.Proofs.Real

namespace VeriTile.Bench.Repairs.L2NormTriton2.Kernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

/-- Actual X/DY/DX storage bindings; X and DY may have different shapes but
the same flat element count. Row grouping uses the DY last dimension exactly
as in the source wrapper. No copy, result or execution premise occurs. -/
structure Buffers (x dy : TensorMetadata) (input gradient : View)
    (p : Physical.Layout) : Prop where
  inputTensor : TensorBinding x input
  gradientTensor : TensorBinding dy gradient
  valid : Physical.Valid p
  columns : p.cols = dy.shape.getLastD 0
  count : p.count = x.numel
  inputKeys : ∀ i, i < x.numel → p.input.key i = input.key i
  gradientKeys : ∀ i, i < x.numel → p.gradient.key i = gradient.key i

def identityAdapter {α : Type} (A : Algebra α) : Source.Adapter α α :=
  ⟨A,id,fun _ => id,fun _ _ => rfl⟩
def realAdapter : Source.Adapter ℝ ℝ := identityAdapter RealExecution.algebra

def formula {M β : Type} (ops : Ops M ℝ β) (input gradient : View) (N : Nat)
    (eps : ℝ) (before : M) (row col : Nat) : ℝ :=
  RealExecution.formula N eps
    (fun j => ops.read before (input.key (row*N+j)))
    (fun j => ops.read before (gradient.key (row*N+j))) col

theorem real_row {M β : Type} (ops : Ops M ℝ β)
    (x dy : TensorMetadata) (input gradient : View) (p : Physical.Layout)
    (binding : Buffers x dy input gradient p) (eps : ℝ) (before : M)
    (row : Nat) (hr : row < p.rows) :
    ∃ after, FP.Structural.exec RealExecution.algebra (backward p.cols p.block)
        (Tile.rowState row eps (Source.state ops realAdapter p before)) = some after ∧
      (∀ col, col < p.cols → after.mem "DX" (row*p.cols+col) =
        Cell.mk .real (formula ops input gradient p.cols eps before row col)) ∧
      (∀ (r : RegionName) (i : Nat), r ≠ "DX" ∨ (∀ col, col < p.cols → i ≠ row*p.cols+col) →
        after.mem r i = (Source.state ops realAdapter p before).mem r i) := by
  refine ⟨(Tile.sourceRun RealExecution.algebra p.cols p.block row eps
    (Source.state ops realAdapter p before)).val,
    (Tile.sourceRun RealExecution.algebra p.cols p.block row eps
      (Source.state ops realAdapter p before)).property,?_,?_⟩
  · intro col hc
    have cb := lt_of_lt_of_le hc binding.valid.main.blockCovers
    rw [Tile.readback _ _ _ _ _ _ (⟨⟨col,cb⟩,PUnit.unit⟩) hc]
    congr 1
    change Tile.scalar RealExecution.algebra p.cols p.block eps
      (Tile.inputs (Source.state ops realAdapter p before) "X" p.cols row)
      (Tile.inputs (Source.state ops realAdapter p before) "DY" p.cols row) col = _
    rw [Tile.scalar_congr RealExecution.algebra p.cols p.block eps _ _
      (fun j => ops.read before (input.key (row*p.cols+j)))
      (fun j => ops.read before (gradient.key (row*p.cols+j))) (by
        intro j hj
        have inside : row*p.cols+j < x.numel := binding.count ▸ Physical.index_bound p row j hr hj
        simp [Tile.inputs,Source.state,realAdapter,identityAdapter,Cell.read,binding.inputKeys _ inside]) (by
        intro j hj
        have inside : row*p.cols+j < x.numel := binding.count ▸ Physical.index_bound p row j hr hj
        simp [Tile.inputs,Source.state,realAdapter,identityAdapter,Cell.read,binding.gradientKeys _ inside])
      col hc,RealExecution.scalar_real _ _ _ _ _ _ binding.valid.main.blockCovers]
    rfl
  · intro r i outside
    apply Tile.frame
    rcases outside with different | outside
    · exact Or.inl different
    · exact Or.inr (fun lane active => outside lane.1.val active)

theorem real_grid {M β : Type} (ops : Ops M ℝ β)
    (x dy : TensorMetadata) (input gradient : View) (p : Physical.Layout)
    (binding : Buffers x dy input gradient p) (eps : ℝ) (before : M) :
    Source.Certificates ops realAdapter p eps (List.range p.rows) before ∧
      (∀ row col, row < p.rows → col < p.cols →
        ops.read (Physical.run ops (realAdapter.compute p eps) p (List.range p.rows) before)
          (p.ykey row col) = formula ops input gradient p.cols eps before row col) ∧
      (∀ owner address,
        (∀ row col, row < p.rows → col < p.cols → ¬(p.ykey row col).covers owner address) →
        ops.observe (Physical.run ops (realAdapter.compute p eps) p (List.range p.rows) before)
          owner address = ops.observe before owner address) := by
  refine ⟨Source.certificates ops realAdapter p binding.valid eps (List.range p.rows)
    (fun row member => List.mem_range.mp member) before,?_,?_⟩
  · intro row col hr hc
    rw [Physical.run_values ops (realAdapter.compute p eps) p binding.valid (List.range p.rows)
      (fun row member => List.mem_range.mp member) List.nodup_range row col
      (List.mem_range.mpr hr) hc before]
    dsimp only [Source.Adapter.compute,realAdapter,identityAdapter,id]
    rw [Tile.scalar_congr RealExecution.algebra p.cols p.block eps _ _
      (fun j => ops.read before (input.key (row*p.cols+j)))
      (fun j => ops.read before (gradient.key (row*p.cols+j))) (by
        intro j hj
        have inside : row*p.cols+j < x.numel := binding.count ▸ Physical.index_bound p row j hr hj
        simp [Physical.rowInputs,Physical.Layout.xkey,Physical.Layout.inputFor,hj,binding.inputKeys _ inside]) (by
        intro j hj
        have inside : row*p.cols+j < x.numel := binding.count ▸ Physical.index_bound p row j hr hj
        simp [Physical.rowInputs,Physical.Layout.xkey,Physical.Layout.inputFor,hj,binding.gradientKeys _ inside])
      col hc,RealExecution.scalar_real _ _ _ _ _ _ binding.valid.main.blockCovers]
    rfl
  · intro owner address outside
    exact Physical.run_bytes ops (realAdapter.compute p eps) p (List.range p.rows) before
      owner address (fun row member col hc => outside row col (List.mem_range.mp member) hc)

theorem commuted_execution {α : Type} [Inhabited α] (A : Algebra α)
    (N B row : Nat) (eps : α) (before : State α) :
    FP.Structural.exec A (backwardCommuted N B) (Tile.rowState row eps before) =
      FP.Structural.exec A (backward N B) (Tile.rowState row eps before) := by
  simp (maxSteps := 900000) [FP.Structural.exec,backwardCommuted,backward,backwardTemplate,
    commutedBackwardTemplate,ComputeKernel.surfaceBody,run,step,FP.Structural.evalExpr,
    evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,Option.bind,ComputeDType.eraseDType,
    numeric,numericLt,bop,natLt,Tile.rowState,ofFloat,toFloat,TileShape.axisDim,
    Region.cast,State.setReg,Int.mul_comm]

theorem executes {α : Type} [Inhabited α] (A : Algebra α)
    (N B row : Nat) (eps : α) (before : State α) :
    ∃ after, FP.Structural.exec A (backward N B) (Tile.rowState row eps before) = some after :=
  ⟨(Tile.sourceRun A N B row eps before).val,(Tile.sourceRun A N B row eps before).property⟩

#specNonCircular formula avoiding [backward,backwardCommuted,Tile.scalar,Physical.run]
#axiomsClean real_row
#axiomsClean real_grid
#axiomsClean commuted_execution
#axiomsClean executes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton2.Kernel
