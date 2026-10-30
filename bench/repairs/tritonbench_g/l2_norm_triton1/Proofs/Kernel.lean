import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.Contract
import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.Source
import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.Real

namespace VeriTile.Bench.Repairs.L2NormTriton1.Kernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

/-- Ordinary allocation/metadata binding for the actual kernel arguments.
`inputKeys` identifies the dense input descriptor with the original view,
including its storage offset. There is no copy, execution or result premise. -/
structure Buffers (metadata : TensorMetadata) (input : View) (p : Physical.Layout) : Prop where
  tensor : TensorBinding metadata input
  valid : Physical.Valid p
  columns : p.cols = metadata.shape.getLastD 0
  count : p.count = metadata.numel
  inputKeys : ∀ i, i < metadata.numel → p.input.key i = input.key i

def identityAdapter {α : Type} (M : Algebra α) : Source.Adapter α α :=
  ⟨M,id,fun _ => id,fun _ _ => rfl⟩

def realAdapter : Source.Adapter ℝ ℝ := identityAdapter RealExecution.algebra

/-- The independently stated formula at original Tensor coordinates. -/
def formula {M β : Type} (ops : Ops M ℝ β) (input : View) (N : Nat)
    (eps : ℝ) (before : M) (row col : Nat) : ℝ :=
  RealExecution.formula N eps (fun j => ops.read before (input.key (row*N+j))) col

theorem input_values {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (A : Algebra α) (metadata : TensorMetadata) (input : View) (p : Physical.Layout)
    (binding : Buffers metadata input p) (before : M) (row col : Nat)
    (hr : row < p.rows) (hc : col < p.cols) :
    Tile.inputs (Source.state ops (identityAdapter A) p before) p.cols row col =
      ops.read before (input.key (row*p.cols+col)) := by
  have inside : row*p.cols+col < metadata.numel :=
    binding.count ▸ Physical.index_bound p row col hr hc
  simp [Tile.inputs,Source.state,identityAdapter,Cell.read,binding.inputKeys _ inside]

/-- Successful complete row execution and the full logical-memory frame.
The existing source proof covers all operations; the independent formula is
connected only after the complete active row is established. -/
theorem real_row {M β : Type} (ops : Ops M ℝ β)
    (metadata : TensorMetadata) (input : View) (p : Physical.Layout)
    (binding : Buffers metadata input p) (eps : ℝ) (before : M)
    (row : Nat) (hr : row < p.rows) :
    ∃ after, FP.Structural.exec RealExecution.algebra (source p.cols p.block)
        (Tile.rowState row eps (Source.state ops realAdapter p before)) = some after ∧
      (∀ col, col < p.cols → after.mem "Y" (row*p.cols+col) =
        Cell.mk .real (formula ops input p.cols eps before row col)) ∧
      (∀ (r : RegionName) (i : Nat), r ≠ "Y" ∨ (∀ col, col < p.cols → i ≠ row*p.cols+col) →
        after.mem r i = (Source.state ops realAdapter p before).mem r i) := by
  refine ⟨(Tile.sourceRun RealExecution.algebra p.cols p.block row eps
    (Source.state ops realAdapter p before)).val,
    (Tile.sourceRun RealExecution.algebra p.cols p.block row eps
      (Source.state ops realAdapter p before)).property,?_,?_⟩
  · intro col hc
    have cb := lt_of_lt_of_le hc binding.valid.blockCovers
    rw [Tile.readback _ _ _ _ _ _ (⟨⟨col,cb⟩,PUnit.unit⟩) hc]
    congr 1
    change Tile.scalar RealExecution.algebra p.cols p.block eps
      (Tile.inputs (Source.state ops (identityAdapter RealExecution.algebra) p before) p.cols row) col = _
    rw [Tile.scalar_congr RealExecution.algebra p.cols p.block eps _
      (fun j => ops.read before (input.key (row*p.cols+j)))
      (fun j hj => input_values ops RealExecution.algebra metadata input p binding before row j hr hj)
      col hc,RealExecution.scalar_real _ _ _ _ _ binding.valid.blockCovers]
    rfl
  · intro r i outside
    apply Tile.frame
    rcases outside with different | outside
    · exact Or.inl different
    · exact Or.inr (fun lane active => outside lane.1.val active)

theorem real_compute (p : Physical.Layout) (valid : Physical.Valid p)
    (eps : ℝ) (xs : Nat → ℝ) :
    realAdapter.compute p eps xs = RealExecution.formula p.cols eps xs := by
  funext i
  exact RealExecution.scalar_real p.cols p.block eps xs i valid.blockCovers

/-- All rows have actual complete source-execution certificates at their
preceding physical heap. Every output element and every exterior byte follow
from the reused physical-grid proof. No host launch/allocator proof is added. -/
theorem real_grid {M β : Type} (ops : Ops M ℝ β)
    (metadata : TensorMetadata) (input : View) (p : Physical.Layout)
    (binding : Buffers metadata input p) (eps : ℝ) (before : M) :
    Source.Certificates ops realAdapter p eps (List.range p.rows) before ∧
      (∀ row col, row < p.rows → col < p.cols →
        ops.read (Physical.run ops (realAdapter.compute p eps) p (List.range p.rows) before)
          (p.ykey row col) = formula ops input p.cols eps before row col) ∧
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
    rw [Tile.scalar_congr RealExecution.algebra p.cols p.block eps _
      (fun j => ops.read before (input.key (row*p.cols+j))) (by
        intro j hj
        have inside : row*p.cols+j < metadata.numel :=
          binding.count ▸ Physical.index_bound p row j hr hj
        simp [Physical.rowInputs,Physical.Layout.xkey,hj,binding.inputKeys _ inside]) col hc,
      RealExecution.scalar_real _ _ _ _ _ binding.valid.blockCovers]
    rfl
  · intro owner address outside
    exact Physical.run_bytes ops (realAdapter.compute p eps) p (List.range p.rows) before
      owner address (fun row member col hc => outside row col (List.mem_range.mp member) hc)

/-- The actual candidate changes integer address products only. This equality
holds for every arithmetic backend and every raw scalar epsilon value. -/
theorem commuted_execution {α : Type} [Inhabited α] (A : Algebra α)
    (N B row : Nat) (eps : α) (before : State α) :
    FP.Structural.exec A (indexCommuted N B) (Tile.rowState row eps before) =
      FP.Structural.exec A (source N B) (Tile.rowState row eps before) := by
  simp (maxSteps := 900000) [FP.Structural.exec,indexCommuted,source,template,commutedTemplate,
    ComputeKernel.surfaceBody,run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    evalOffsets,evalDeltas,store,Option.bind,ComputeDType.eraseDType,
    numeric,numericLt,bop,natLt,Tile.rowState,ofFloat,toFloat,TileShape.axisDim,
    Region.cast,State.setReg,Int.mul_comm]

/-- Opaque success interface reuses the constructed full source execution,
keeping its generated final-state expression out of public proof elaboration. -/
theorem executes {α : Type} [Inhabited α] (A : Algebra α)
    (N B row : Nat) (eps : α) (before : State α) :
    ∃ after, FP.Structural.exec A (source N B) (Tile.rowState row eps before) = some after :=
  ⟨(Tile.sourceRun A N B row eps before).val,(Tile.sourceRun A N B row eps before).property⟩

#specNonCircular formula avoiding [source,indexCommuted,Tile.scalar,Physical.run]
#axiomsClean real_row
#axiomsClean real_grid
#axiomsClean commuted_execution
#axiomsClean executes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton1.Kernel
