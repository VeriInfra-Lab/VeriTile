import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.Stores
import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.Proofs.Reference
import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.QueryWhole

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

/-- Source-level output values are independent of the execution and of
initial output cells. DQ/DK are deliberately the value-shard scratch slices. -/
def sourceQuery (a : Scalars α) (shape : Shape) (scale : ℝ) (x : TensorInputs α)
    (tile : Tile) (r e : Nat) : α := queryLane (a.ops scale) sourcePlans shape x tile r e
def sourceKeyValue (a : Scalars α) (shape : Shape) (scale : ℝ) (x : TensorInputs α)
    (tile : Tile) (r e v : Nat) : α×α := keyValueLane (a.ops scale) sourcePlans shape x tile r e v
def sourceForward (a : Scalars α) (shape : Shape) (scale : ℝ) (x : TensorInputs α)
    (tile : Tile) (r v : Nat) : α×α := forwardLane (a.ops scale) sourcePlans shape x tile r v

theorem query_stores {α : Type} [Inhabited α] (a : Scalars α) (shape : Shape) (scale : ℝ)
    (tile : Tile) (s : State α) (values : TileIndex [64,shape.BK] → α) :
    queryOutputMemory (scalarAlgebra a) (dqConfig shape scale tile) s values =
      rectangleStore (queryRectangle shape tile) s (fun i => a.cast (values i)) := by
  simp only [queryOutputMemory,rectangleStore,queryRectangle,dqConfig,backwardConfig,forwardConfig,
    Rectangle.active,Rectangle.address,scalarAlgebra,Nat.mul_one]

theorem backward_stores {α : Type} [Inhabited α] (a : Scalars α) (shape : Shape) (scale : ℝ)
    (tile : Tile) (s : State α) (values : BackwardAccumulator α 64 shape.BK shape.BV) :
    backwardOutputMemory (scalarAlgebra a) (backwardConfig shape scale tile) s values =
      rectangleStore (valueRectangle shape tile)
        (rectangleStore (keyRectangle shape tile) s (fun i => a.cast (values.key i)))
        (fun i => a.cast (values.value i)) := by
  simp only [backwardOutputMemory,rectangleStore,keyRectangle,valueRectangle,queryRectangle,
    outputRectangle,backwardConfig,forwardConfig,Rectangle.active,Rectangle.address,scalarAlgebra,Nat.mul_one]

/-- The complete readable DQ quotation executes and writes its independent
source-order scalar target. Every cell outside its active rectangle is framed. -/
theorem query_source {α : Type} [Inhabited α] (a : Scalars α) (shape : Shape) (scale : ℝ)
    (tile : Tile) (s : State α) :
    ∃ final,FP.Structural.exec (withDot (scalarAlgebra a) (scalarDotPrimitive a))
        (queryKernel (dqConfig shape scale tile)) s=some final ∧
      (∀ (r : Fin 64) (e : Fin shape.BK),(queryRectangle shape tile).active ⟨r,e,()⟩ →
        final.mem "dq" ((queryRectangle shape tile).address ⟨r,e,()⟩)=
          .mk .real (sourceQuery a shape scale (inputReaders shape s) tile r.val e.val)) ∧
      (∀ R off,¬(queryRectangle shape tile).writes 64 shape.BK R off → final.mem R off=s.mem R off) := by
  obtain ⟨final,executed,memory⟩ := query_fp32 (scalarAlgebra a) (scalarDotPrimitive a) (dqConfig shape scale tile) s
  refine ⟨final,executed,?_,?_⟩
  · intro r e active
    erw [memory,query_stores,rectangleStore_read _ _ _ _ active,query_scalar_values]
    rfl
  · intro R off outside
    rw [memory,query_stores]
    exact rectangleStore_frame _ _ _ R off outside

/-- Both DK and DV are observed from the complete source execution, including
the future reverse range. The frame retains raw cell tags and arbitrary offsets. -/
theorem keyValue_source {α : Type} [Inhabited α] (a : Scalars α) (shape : Shape) (scale : ℝ)
    (tile : Tile) (s : State α) (keyWidth : 0<shape.BK) (valueWidth : 0<shape.BV) :
    ∃ final,FP.Structural.exec (withDot (scalarAlgebra a) (scalarDotPrimitive a))
        (backwardKernel (backwardConfig shape scale tile)) s=some final ∧
      (∀ (r : Fin 64) (e : Fin shape.BK),(keyRectangle shape tile).active ⟨r,e,()⟩ →
        final.mem "dk" ((keyRectangle shape tile).address ⟨r,e,()⟩)=
          .mk .real (sourceKeyValue a shape scale (inputReaders shape s) tile r.val e.val 0).1) ∧
      (∀ (r : Fin 64) (v : Fin shape.BV),(valueRectangle shape tile).active ⟨r,v,()⟩ →
        final.mem "dv" ((valueRectangle shape tile).address ⟨r,v,()⟩)=
          .mk .real (sourceKeyValue a shape scale (inputReaders shape s) tile r.val 0 v.val).2) ∧
      (∀ R off,¬((keyRectangle shape tile).writes 64 shape.BK R off ∨
        (valueRectangle shape tile).writes 64 shape.BV R off) → final.mem R off=s.mem R off) := by
  obtain ⟨final,executed,memory⟩ := backward_fp32 (scalarAlgebra a) (scalarDotPrimitive a) (backwardConfig shape scale tile) s
  refine ⟨final,executed,?_,?_,?_⟩
  · intro r e active
    rw [memory,backward_stores,rectangleStore_frame]
    · erw [rectangleStore_read _ _ _ _ active]
      exact congrArg (fun z : α×α => Cell.mk .real z.1)
        (keyValue_scalar_values a shape scale tile s r e ⟨0,valueWidth⟩)
    · intro h
      exact (by decide : ("dk" : RegionName)≠"dv") h.1
  · intro r v active
    erw [memory,backward_stores,rectangleStore_read _ _ _ _ active]
    exact congrArg (fun z : α×α => Cell.mk .real z.2)
      (keyValue_scalar_values a shape scale tile s r ⟨0,keyWidth⟩ v)
  · intro R off outside
    rw [memory,backward_stores,rectangleStore_frame,rectangleStore_frame]
    · exact fun h => outside (Or.inl h)
    · exact fun h => outside (Or.inr h)

#axiomsClean query_stores
#axiomsClean backward_stores
#axiomsClean query_source
#axiomsClean keyValue_source
#specNonCircular sourceQuery avoiding [FP.Structural.exec,FP.Structural.run,queryValues,queryOutputMemory]
#specNonCircular sourceKeyValue avoiding [FP.Structural.exec,FP.Structural.run,backwardValues,backwardOutputMemory]
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel
