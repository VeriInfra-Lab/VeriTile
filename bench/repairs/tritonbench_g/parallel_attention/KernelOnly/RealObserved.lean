import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.ForwardObserved

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
set_option maxHeartbeats 3000000

/-- Selected program coordinates. The checked kernel has exactly one key
shard; value shards remain observable DQ/DK scratch dimensions. -/
structure TileDomain (shape : Shape) (L : Nat) (tile : Tile) : Prop where
  head : tile.bh<shape.B*shape.H
  row : tile.chunk*L<shape.T
  key : tile.key=0
  value : tile.value*shape.BV<shape.V
  keyFits : shape.K≤ shape.BK
  keyPositive : 0<shape.K
  valueWidth : 0<shape.BV

/-- Successful complete source execution and all observable DQ scratch
values at the actual supplied Real scale, together with the full cell frame. -/
theorem query_real (shape : Shape) (σ : ℝ) (tile : Tile) (s : State ℝ)
    (domain : TileDomain shape 64 tile) :
    ∃ final,FP.Structural.exec (withDot (scalarAlgebra realScalars) (scalarDotPrimitive realScalars))
        (queryKernel (dqConfig shape σ tile)) s=some final ∧
      (∀ (r : Fin 64) (e : Fin shape.BK),(queryRectangle shape tile).active ⟨r,e,()⟩ →
        final.mem "dq" ((queryRectangle shape tile).address ⟨r,e,()⟩)=
          .mk .real (queryPartial shape σ (inputReaders shape s) tile.bh tile.value (tile.chunk*64+r.val) e.val)) ∧
      (∀ R off,¬(queryRectangle shape tile).writes 64 shape.BK R off → final.mem R off=s.mem R off) := by
  obtain ⟨final,executed,reads,frame⟩ := query_source realScalars shape σ tile s
  refine ⟨final,executed,?_,frame⟩
  intro r e active
  rw [reads r e active]
  apply congrArg (Cell.mk .real)
  have ht := active.1
  have he : e.val<shape.K := by simpa [queryRectangle,Rectangle.active,domain.key] using active.2
  exact Math.query_whole_real (σ := σ) sourcePlans shape (inputReaders shape s)
    tile r.val e.val domain.key domain.keyFits r.isLt ht he

/-- DK is a partial value-shard gradient and DV is the complete value
coordinate gradient. No host sum or normalization appears in this endpoint. -/
theorem keyValue_real (shape : Shape) (σ : ℝ) (tile : Tile) (s : State ℝ)
    (domain : TileDomain shape 64 tile) :
    ∃ final,FP.Structural.exec (withDot (scalarAlgebra realScalars) (scalarDotPrimitive realScalars))
        (backwardKernel (backwardConfig shape σ tile)) s=some final ∧
      (∀ (r : Fin 64) (e : Fin shape.BK),(keyRectangle shape tile).active ⟨r,e,()⟩ →
        final.mem "dk" ((keyRectangle shape tile).address ⟨r,e,()⟩)=
          .mk .real (keyPartial shape σ (inputReaders shape s) tile.bh tile.value (tile.chunk*64+r.val) e.val)) ∧
      (∀ (r : Fin 64) (v : Fin shape.BV),(valueRectangle shape tile).active ⟨r,v,()⟩ →
        final.mem "dv" ((valueRectangle shape tile).address ⟨r,v,()⟩)=
          .mk .real (valueGradient shape σ (inputReaders shape s) tile.bh (tile.chunk*64+r.val) (tile.value*shape.BV+v.val))) ∧
      (∀ R off,¬((keyRectangle shape tile).writes 64 shape.BK R off ∨
        (valueRectangle shape tile).writes 64 shape.BV R off) → final.mem R off=s.mem R off) := by
  obtain ⟨final,executed,keyReads,valueReads,frame⟩ := keyValue_source realScalars shape σ tile s
    (by have := domain.keyFits; have := domain.keyPositive; omega) domain.valueWidth
  refine ⟨final,executed,?_,?_,frame⟩
  · intro r e active
    rw [keyReads r e active]
    apply congrArg (Cell.mk .real)
    have he : e.val<shape.K := by simpa [keyRectangle,queryRectangle,Rectangle.active,domain.key] using active.2
    exact congrArg Prod.fst (Math.keyValue_whole_real (σ := σ) sourcePlans shape (inputReaders shape s)
      tile r.val e.val 0 domain.key domain.keyFits active.1 he (by simpa using domain.value) r.isLt)
  · intro r v active
    rw [valueReads r v active]
    apply congrArg (Cell.mk .real)
    exact congrArg Prod.snd (Math.keyValue_whole_real (σ := σ) sourcePlans shape (inputReaders shape s)
      tile r.val 0 v.val domain.key domain.keyFits active.1 domain.keyPositive active.2 r.isLt)

theorem forward_real (shape : Shape) (σ : ℝ) (s : State ℝ)
    (domain : TileDomain shape 128 (launchTile shape s)) :
    ∃ final,FP.Structural.exec (withDot (scalarAlgebra realScalars) (scalarDotPrimitive realScalars))
        (forwardKernel (forwardConfig shape σ)) s=some final ∧
      (∀ (r : Fin 128) (v : Fin shape.BV),(outputRectangle shape (launchTile shape s) 128).active ⟨r,v,()⟩ →
        final.mem "o" ((outputRectangle shape (launchTile shape s) 128).address ⟨r,v,()⟩)=
          .mk .real (outputSpec shape.T shape.K σ (headInputs (inputReaders shape s) (launchTile shape s).bh)
            ((launchTile shape s).chunk*128+r.val) ((launchTile shape s).value*shape.BV+v.val))) ∧
      (∀ (r : Fin 128),(launchTile shape s).value=0 → (launchTile shape s).chunk*128+r.val<shape.T →
        final.mem "z" (normalizerAddress shape (launchTile shape s) r.val)=
          .mk .real (normalizerSpec shape.T shape.K σ (headInputs (inputReaders shape s) (launchTile shape s).bh)
            ((launchTile shape s).chunk*128+r.val))) ∧
      (∀ R off,¬((outputRectangle shape (launchTile shape s) 128).writes 128 shape.BV R off ∨
        normalizerWrites shape (launchTile shape s) R off) → final.mem R off=s.mem R off) := by
  obtain ⟨final,executed,outputReads,normReads,frame⟩ := forward_source realScalars shape σ s domain.valueWidth
  refine ⟨final,executed,?_,?_,frame⟩
  · intro r v active
    rw [outputReads r v active]
    apply congrArg (Cell.mk .real)
    exact congrArg Prod.fst (Math.forward_whole_real (σ := σ) sourcePlans shape (inputReaders shape s)
      (launchTile shape s) r.val v.val domain.key domain.keyFits r.isLt active.1 active.2)
  · intro r writer active
    rw [normReads r writer active]
    apply congrArg (Cell.mk .real)
    exact congrArg Prod.snd (Math.forward_whole_real (σ := σ) sourcePlans shape (inputReaders shape s)
      (launchTile shape s) r.val 0 domain.key domain.keyFits r.isLt active (by simpa using domain.value))

#axiomsClean query_real
#axiomsClean keyValue_real
#axiomsClean forward_real
#specNonCircular queryPartial avoiding [FP.Structural.exec,FP.Structural.run,queryValues,queryLane]
#specNonCircular keyPartial avoiding [FP.Structural.exec,FP.Structural.run,backwardValues,keyValueLane]
#specNonCircular valueGradient avoiding [FP.Structural.exec,FP.Structural.run,backwardValues,keyValueLane]
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel
