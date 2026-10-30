import bench.tritonbench_g.parallel_attention.NativeReference

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.NativeStage
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
set_option maxHeartbeats 4000000

/-- Complete DQ quotation execution under the staged native interpretation,
all observable scratch words, and the complete raw tagged memory frame. -/
theorem query_native (actual reference : Scalars Word) (laws : PrimitiveLaws actual reference)
    (d : NativePlans.Descriptor) (σ : ℝ) (tile : Tile) (s : State Word) :
    ∃ graph final,
      FP.Structural.exec (withDot (algebra .query d) (dots .query d.shape))
        (queryKernel (dqConfig d.shape σ tile)) (entry s)=some graph ∧
      queryRun actual d σ tile s=some final ∧
      (∀ (r : Fin 64) (e : Fin d.shape.BK),(queryRectangle d.shape tile).active ⟨r,e,()⟩ →
        final.mem "dq" ((queryRectangle d.shape tile).address ⟨r,e,()⟩)=.mk .real (queryTarget reference d σ tile s r e)) ∧
      (∀ R off,¬(queryRectangle d.shape tile).writes 64 d.shape.BK R off → final.mem R off=s.mem R off) := by
  have same := laws.eq
  subst reference
  obtain ⟨graph,executed,memory⟩ := query_fp32 (algebra .query d) (dots .query d.shape) (dqConfig d.shape σ tile) (entry s)
  refine ⟨graph,finish actual .query d tile graph,executed,?_,?_,?_⟩
  · simp only [queryRun,executed,Option.map_some]
  · intro r e active
    change mapCell _ (graph.mem _ _) = _
    erw [memory,query_stores_algebra,rectangleStore_read _ _ _ _ active]
    rfl
  · intro R off outside
    apply finish_frame
    rw [memory,query_stores_algebra]
    exact rectangleStore_frame _ _ _ R off outside

theorem keyValue_native (actual reference : Scalars Word) (laws : PrimitiveLaws actual reference)
    (d : NativePlans.Descriptor) (σ : ℝ) (tile : Tile) (s : State Word) :
    ∃ graph final,
      FP.Structural.exec (withDot (algebra .keyValue d) (dots .keyValue d.shape))
        (backwardKernel (backwardConfig d.shape σ tile)) (entry s)=some graph ∧
      keyValueRun actual d σ tile s=some final ∧
      (∀ (r : Fin 64) (e : Fin d.shape.BK),(keyRectangle d.shape tile).active ⟨r,e,()⟩ →
        final.mem "dk" ((keyRectangle d.shape tile).address ⟨r,e,()⟩)=.mk .real (keyTarget reference d σ tile s r e)) ∧
      (∀ (r : Fin 64) (v : Fin d.shape.BV),(valueRectangle d.shape tile).active ⟨r,v,()⟩ →
        final.mem "dv" ((valueRectangle d.shape tile).address ⟨r,v,()⟩)=.mk .real (valueTarget reference d σ tile s r v)) ∧
      (∀ R off,¬((keyRectangle d.shape tile).writes 64 d.shape.BK R off ∨
        (valueRectangle d.shape tile).writes 64 d.shape.BV R off) → final.mem R off=s.mem R off) := by
  have same := laws.eq
  subst reference
  obtain ⟨graph,executed,memory⟩ := backward_fp32 (algebra .keyValue d) (dots .keyValue d.shape) (backwardConfig d.shape σ tile) (entry s)
  refine ⟨graph,finish actual .keyValue d tile graph,executed,?_,?_,?_,?_⟩
  · simp only [keyValueRun,executed,Option.map_some]
  · intro r e active
    change mapCell _ (graph.mem _ _) = _
    rw [memory,backward_stores_algebra,rectangleStore_frame]
    · erw [rectangleStore_read _ _ _ _ active]
      rfl
    · intro h; exact (by decide : ("dk" : RegionName)≠"dv") h.1
  · intro r v active
    change mapCell _ (graph.mem _ _) = _
    erw [memory,backward_stores_algebra,rectangleStore_read _ _ _ _ active]
    rfl
  · intro R off outside
    apply finish_frame
    rw [memory,backward_stores_algebra,rectangleStore_frame,rectangleStore_frame]
    · exact fun h => outside (Or.inl h)
    · exact fun h => outside (Or.inr h)

theorem forward_native (actual reference : Scalars Word) (laws : PrimitiveLaws actual reference)
    (d : NativePlans.Descriptor) (σ : ℝ) (s : State Word) :
    ∃ graph final,
      FP.Structural.exec (withDot (algebra .forward d) (dots .forward d.shape))
        (forwardKernel (forwardConfig d.shape σ)) (entry s)=some graph ∧
      forwardRun actual d σ s=some final ∧
      (∀ (r : Fin 128) (v : Fin d.shape.BV),(outputRectangle d.shape (launchTile d.shape s) 128).active ⟨r,v,()⟩ →
        final.mem "o" ((outputRectangle d.shape (launchTile d.shape s) 128).address ⟨r,v,()⟩)=.mk .real (outputTarget reference d σ s r v)) ∧
      (∀ (r : Fin 128),(launchTile d.shape s).value=0 → (launchTile d.shape s).chunk*128+r.val<d.shape.T →
        final.mem "z" (normalizerAddress d.shape (launchTile d.shape s) r.val)=.mk .real (normalizerTarget reference d σ s r)) ∧
      (∀ R off,¬((outputRectangle d.shape (launchTile d.shape s) 128).writes 128 d.shape.BV R off ∨
        normalizerWrites d.shape (launchTile d.shape s) R off) → final.mem R off=s.mem R off) := by
  have same := laws.eq
  subst reference
  obtain ⟨graph,executed,memory⟩ := forward_fp32 (algebra .forward d) (dots .forward d.shape) (forwardConfig d.shape σ) (entry s)
  change graph.mem=(outputMemory (algebra .forward d) (forwardConfig d.shape σ) (entry s)
    (forwardValues (algebra .forward d) (dots .forward d.shape) (forwardConfig d.shape σ) (entry s))).mem at memory
  refine ⟨graph,finish actual .forward d (launchTile d.shape s) graph,executed,?_,?_,?_,?_⟩
  · simp only [forwardRun,executed,Option.map_some]
  · intro r v active
    change mapCell _ (graph.mem _ _) = _
    rw [memory,forward_stores_algebra,normalizerStore_frame]
    · erw [rectangleStore_read _ _ _ _ active]
      rfl
    · intro h; exact (by decide : ("o" : RegionName)≠"z") h.2.1
  · intro r writer active
    change mapCell _ (graph.mem _ _) = _
    erw [memory,forward_stores_algebra,normalizerStore_read _ _ _ _ r writer active]
    rfl
  · intro R off outside
    apply finish_frame
    rw [memory,forward_stores_algebra,normalizerStore_frame,rectangleStore_frame]
    · exact fun h => outside (Or.inl h)
    · exact fun h => outside (Or.inr h)

#axiomsClean query_native
#axiomsClean keyValue_native
#axiomsClean forward_native
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.NativeStage
