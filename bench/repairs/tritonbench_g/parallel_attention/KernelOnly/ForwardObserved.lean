import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.Observed

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
set_option maxHeartbeats 4000000

def normalizerAddress (shape : Shape) (tile : Tile) (r : Nat) : Nat :=
  (tile.bh+shape.B*shape.H*tile.key)*shape.T+tile.chunk*128+r

def normalizerWrites (shape : Shape) (tile : Tile) (R : RegionName) (off : Nat) : Prop :=
  tile.value=0 ∧ R="z" ∧ ∃ r : Fin 128,tile.chunk*128+r.val<shape.T ∧ off=normalizerAddress shape tile r.val

def normalizerStore (shape : Shape) (tile : Tile) (s : State α) (values : Fin 128 → α) : State α :=
  if tile.value=0 then (TileShape.allIndices [128]).foldl (fun t i =>
    if tile.chunk*128+i.1.val<shape.T then t.write "z" (normalizerAddress shape tile i.1.val)
      (.mk .real (values i.1)) else t) s else s

theorem normalizerStore_frame (shape : Shape) (tile : Tile) (s : State α) (values : Fin 128 → α)
    (R : RegionName) (off : Nat) (outside : ¬normalizerWrites shape tile R off) :
    (normalizerStore shape tile s values).mem R off=s.mem R off := by
  unfold normalizerStore
  split
  · rename_i hz
    apply State.masked_scatter_frame
    by_cases hR : R="z"
    · right
      intro i _ hi heq
      exact outside ⟨hz,hR,i.1,hi,heq⟩
    · exact Or.inl hR
  · rfl

theorem normalizerStore_read (shape : Shape) (tile : Tile) (s : State α) (values : Fin 128 → α)
    (r : Fin 128) (writer : tile.value=0) (active : tile.chunk*128+r.val<shape.T) :
    (normalizerStore shape tile s values).mem "z" (normalizerAddress shape tile r.val)=.mk .real (values r) := by
  unfold normalizerStore
  rw [if_pos writer]
  apply State.masked_scatter_readback (shape := [128]) _ _ _ _ s ⟨r,()⟩ active
  intro i j same
  have h : i.1=j.1 := Fin.ext (by simp only [normalizerAddress] at same; omega)
  rcases i with ⟨i,⟨⟩⟩
  rcases j with ⟨j,⟨⟩⟩
  cases h; rfl

theorem forward_stores {α : Type} [Inhabited α] (a : Scalars α) (shape : Shape) (scale : ℝ)
    (s : State α) (values : ForwardAccumulator α 128 shape.BV) :
    outputMemory (scalarAlgebra a) (forwardConfig shape scale) s values =
      normalizerStore shape (launchTile shape s)
        (rectangleStore (outputRectangle shape (launchTile shape s) 128) s
          (fun i => a.cast (values.output i)))
        (fun r => a.cast (values.normalizer ⟨r,()⟩)) := by
  simp only [outputMemory,normalizerStore,rectangleStore,outputRectangle,launchTile,forwardConfig,
    queryChunk,valueChunk,Rectangle.active,Rectangle.address,normalizerAddress,scalarAlgebra,Nat.mul_one,Nat.add_assoc]

/-- Forward source execution, O/Z readback and its complete tagged frame. Z
has exactly the source's value-shard-zero write guard. -/
theorem forward_source {α : Type} [Inhabited α] (a : Scalars α) (shape : Shape) (scale : ℝ)
    (s : State α) (valueWidth : 0<shape.BV) :
    ∃ final,FP.Structural.exec (withDot (scalarAlgebra a) (scalarDotPrimitive a))
        (forwardKernel (forwardConfig shape scale)) s=some final ∧
      (∀ (r : Fin 128) (v : Fin shape.BV),(outputRectangle shape (launchTile shape s) 128).active ⟨r,v,()⟩ →
        final.mem "o" ((outputRectangle shape (launchTile shape s) 128).address ⟨r,v,()⟩)=
          .mk .real (sourceForward a shape scale (inputReaders shape s) (launchTile shape s) r.val v.val).1) ∧
      (∀ (r : Fin 128),(launchTile shape s).value=0 → (launchTile shape s).chunk*128+r.val<shape.T →
        final.mem "z" (normalizerAddress shape (launchTile shape s) r.val)=
          .mk .real (sourceForward a shape scale (inputReaders shape s) (launchTile shape s) r.val 0).2) ∧
      (∀ R off,¬((outputRectangle shape (launchTile shape s) 128).writes 128 shape.BV R off ∨
        normalizerWrites shape (launchTile shape s) R off) → final.mem R off=s.mem R off) := by
  obtain ⟨final,executed,memory⟩ := forward_fp32 (scalarAlgebra a) (scalarDotPrimitive a) (forwardConfig shape scale) s
  change final.mem=(outputMemory (scalarAlgebra a) (forwardConfig shape scale) s
    (forwardValues (scalarAlgebra a) (scalarDotPrimitive a) (forwardConfig shape scale) s)).mem at memory
  refine ⟨final,executed,?_,?_,?_⟩
  · intro r v active
    rw [memory,forward_stores,normalizerStore_frame]
    · erw [rectangleStore_read _ _ _ _ active]
      exact congrArg (fun z : α×α => Cell.mk .real z.1) (forward_scalar_values a shape scale s r v)
    · intro h
      exact (by decide : ("o" : RegionName)≠"z") h.2.1
  · intro r writer active
    erw [memory,forward_stores,normalizerStore_read _ _ _ _ r writer active]
    exact congrArg (fun z : α×α => Cell.mk .real z.2)
      (forward_scalar_values a shape scale s r ⟨0,valueWidth⟩)
  · intro R off outside
    rw [memory,forward_stores,normalizerStore_frame,rectangleStore_frame]
    · exact fun h => outside (Or.inl h)
    · exact fun h => outside (Or.inr h)

#axiomsClean normalizerStore_frame
#axiomsClean normalizerStore_read
#axiomsClean forward_stores
#axiomsClean forward_source
#specNonCircular sourceForward avoiding [FP.Structural.exec,FP.Structural.run,forwardValues,outputMemory]
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel
