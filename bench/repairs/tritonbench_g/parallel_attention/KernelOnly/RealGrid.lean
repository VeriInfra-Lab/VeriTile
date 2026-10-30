import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.GridTargets

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
set_option maxHeartbeats 300000

/-- Every DQ scratch slice across the complete source grid, with the actual
incoming scale kept separate from an ideal square-root expression. -/
theorem query_grid_real (shape : Shape) (accepted : ShapeContract shape) (σ : ℝ) (s : State ℝ) :
    ∃ final,runPrograms (queryLaunch realScalars shape σ) (programs shape 64) s=some final ∧ SameInputs s final ∧
      (∀ (p : Program shape 64) (r : Fin 64) (e : Fin shape.BK),
        (queryRectangle shape p.tile).active ⟨r,e,()⟩ →
        final.mem "dq" ((queryRectangle shape p.tile).address ⟨r,e,()⟩)=
          .mk .real (queryPartial shape σ (inputReaders shape s) p.head.val p.value.val (p.chunk.val*64+r.val) e.val)) ∧
      (∀ R off,(∀ p∈programs shape 64,¬(queryRectangle shape p.tile).writes 64 shape.BK R off) → final.mem R off=s.mem R off) := by
  obtain ⟨final,executed,inputs,reads,frame⟩ := query_grid realScalars shape σ s
  refine ⟨final,executed,inputs,?_,frame⟩
  intro p r e active
  rw [reads p (program_member shape 64 p) "dq" _ ⟨rfl,⟨r,e,()⟩,active,rfl⟩,queryGridTarget]
  rw [rectangleTarget_at _ _ _ _ active]
  apply congrArg (Cell.mk .real)
  have domain := program_domain shape accepted 64 (by decide) p
  have he : e.val<shape.K := by simpa [queryRectangle,Rectangle.active,Program.tile] using active.2
  exact Math.query_whole_real (σ := σ) sourcePlans shape (inputReaders shape s)
    p.tile r.val e.val rfl domain.keyFits r.isLt active.1 he

#axiomsClean query_grid_real

theorem keyValue_grid_real (shape : Shape) (accepted : ShapeContract shape) (σ : ℝ) (s : State ℝ) :
    ∃ final,runPrograms (keyValueLaunch realScalars shape σ) (programs shape 64) s=some final ∧ SameInputs s final ∧
      (∀ (p : Program shape 64) (r : Fin 64) (e : Fin shape.BK),
        (keyRectangle shape p.tile).active ⟨r,e,()⟩ →
        final.mem "dk" ((keyRectangle shape p.tile).address ⟨r,e,()⟩)=
          .mk .real (keyPartial shape σ (inputReaders shape s) p.head.val p.value.val (p.chunk.val*64+r.val) e.val)) ∧
      (∀ (p : Program shape 64) (r : Fin 64) (v : Fin shape.BV),
        (valueRectangle shape p.tile).active ⟨r,v,()⟩ →
        final.mem "dv" ((valueRectangle shape p.tile).address ⟨r,v,()⟩)=
          .mk .real (valueGradient shape σ (inputReaders shape s) p.head.val (p.chunk.val*64+r.val) (p.value.val*shape.BV+v.val))) ∧
      (∀ R off,(∀ p∈programs shape 64,¬keyValueWrites shape p R off) → final.mem R off=s.mem R off) := by
  obtain ⟨final,executed,inputs,reads,frame⟩ := keyValue_grid realScalars shape σ s
    (by have := accepted.key_positive; have := accepted.key_fits shape; omega) (accepted.value_tile_positive shape)
  refine ⟨final,executed,inputs,?_,?_,frame⟩
  · intro p r e active
    rw [reads p (program_member shape 64 p) "dk" _ (Or.inl ⟨rfl,⟨r,e,()⟩,active,rfl⟩)]
    simp only [keyValueGridTarget,ite_true]
    rw [rectangleTarget_at _ _ _ _ active]
    apply congrArg (Cell.mk .real)
    have domain := program_domain shape accepted 64 (by decide) p
    have he : e.val<shape.K := by simpa [keyRectangle,queryRectangle,Rectangle.active,Program.tile] using active.2
    exact congrArg Prod.fst (Math.keyValue_whole_real (σ := σ) sourcePlans shape (inputReaders shape s)
      p.tile r.val e.val 0 rfl domain.keyFits active.1 he (by simpa using domain.value) r.isLt)
  · intro p r v active
    rw [reads p (program_member shape 64 p) "dv" _ (Or.inr ⟨rfl,⟨r,v,()⟩,active,rfl⟩)]
    simp only [keyValueGridTarget,show ¬(("dv" : RegionName)="dk") by decide,ite_false]
    rw [rectangleTarget_at _ _ _ _ active]
    apply congrArg (Cell.mk .real)
    have domain := program_domain shape accepted 64 (by decide) p
    exact congrArg Prod.snd (Math.keyValue_whole_real (σ := σ) sourcePlans shape (inputReaders shape s)
      p.tile r.val 0 v.val rfl domain.keyFits active.1 domain.keyPositive active.2 r.isLt)

#axiomsClean keyValue_grid_real

theorem forward_grid_real (shape : Shape) (accepted : ShapeContract shape) (σ : ℝ) (s : State ℝ) :
    ∃ final,runPrograms (forwardLaunch realScalars shape σ) (programs shape 128) s=some final ∧ SameInputs s final ∧
      (∀ (p : Program shape 128) (r : Fin 128) (v : Fin shape.BV),
        (outputRectangle shape p.tile 128).active ⟨r,v,()⟩ →
        final.mem "o" ((outputRectangle shape p.tile 128).address ⟨r,v,()⟩)=
          .mk .real (outputSpec shape.T shape.K σ (headInputs (inputReaders shape s) p.head.val)
            (p.chunk.val*128+r.val) (p.value.val*shape.BV+v.val))) ∧
      (∀ (p : Program shape 128) (r : Fin 128),p.value.val=0 → p.chunk.val*128+r.val<shape.T →
        final.mem "z" (normalizerAddress shape p.tile r.val)=
          .mk .real (normalizerSpec shape.T shape.K σ (headInputs (inputReaders shape s) p.head.val) (p.chunk.val*128+r.val))) ∧
      (∀ R off,(∀ p∈programs shape 128,¬forwardWrites shape p R off) → final.mem R off=s.mem R off) := by
  obtain ⟨final,executed,inputs,reads,frame⟩ := forward_grid realScalars shape σ s (accepted.value_tile_positive shape)
  refine ⟨final,executed,inputs,?_,?_,frame⟩
  · intro p r v active
    rw [reads p (program_member shape 128 p) "o" _ (Or.inl ⟨rfl,⟨r,v,()⟩,active,rfl⟩)]
    simp only [forwardGridTarget,ite_true]
    rw [rectangleTarget_at _ _ _ _ active]
    apply congrArg (Cell.mk .real)
    have domain := program_domain shape accepted 128 (by decide) p
    exact congrArg Prod.fst (Math.forward_whole_real (σ := σ) sourcePlans shape (inputReaders shape s)
      p.tile r.val v.val rfl domain.keyFits r.isLt active.1 active.2)
  · intro p r writer active
    rw [reads p (program_member shape 128 p) "z" _ (Or.inr ⟨writer,rfl,r,active,rfl⟩)]
    simp only [forwardGridTarget,show ¬(("z" : RegionName)="o") by decide,ite_false]
    refine (normalizerTarget_at shape p.tile _ _ r active).trans ?_
    apply congrArg (Cell.mk .real)
    have domain := program_domain shape accepted 128 (by decide) p
    exact congrArg Prod.snd (Math.forward_whole_real (σ := σ) sourcePlans shape (inputReaders shape s)
      p.tile r.val 0 rfl domain.keyFits r.isLt active (by simpa using domain.value))

#axiomsClean query_grid_real
#axiomsClean keyValue_grid_real
#axiomsClean forward_grid_real
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel
