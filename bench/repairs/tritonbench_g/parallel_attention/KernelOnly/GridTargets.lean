import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.GridComposition

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
set_option maxHeartbeats 3000000

/-- The unique active lane is chosen solely from integer addresses. The
reference value is independent of store order and output memory contents. -/
def rectangleTarget (p : Rectangle) (values : TileIndex [m,n] → α) (zero : α) (off : Nat) : Cell α := by
  classical
  exact if h : ∃ i : TileIndex [m,n],p.active i ∧ off=p.address i then
    .mk .real (values h.choose) else .mk .real zero

theorem rectangleTarget_at (p : Rectangle) (values : TileIndex [m,n] → α) (zero : α)
    (i : TileIndex [m,n]) (active : p.active i) :
    rectangleTarget p values zero (p.address i)=.mk .real (values i) := by
  classical
  unfold rectangleTarget
  have h : ∃ j : TileIndex [m,n],p.active j ∧ p.address i=p.address j := ⟨i,active,rfl⟩
  rw [dif_pos h]
  rw [p.unique h.choose i h.choose_spec.1 active h.choose_spec.2.symm]

def normalizerTarget (shape : Shape) (tile : Tile) (values : Fin 128 → α) (zero : α) (off : Nat) : Cell α := by
  classical
  exact if h : ∃ r : Fin 128,tile.chunk*128+r.val<shape.T ∧ off=normalizerAddress shape tile r.val then
    .mk .real (values h.choose) else .mk .real zero

theorem normalizerTarget_at (shape : Shape) (tile : Tile) (values : Fin 128 → α) (zero : α)
    (r : Fin 128) (active : tile.chunk*128+r.val<shape.T) :
    normalizerTarget shape tile values zero (normalizerAddress shape tile r.val)=.mk .real (values r) := by
  classical
  unfold normalizerTarget
  have h : ∃ j : Fin 128,tile.chunk*128+j.val<shape.T ∧ normalizerAddress shape tile r.val=normalizerAddress shape tile j.val := ⟨r,active,rfl⟩
  rw [dif_pos h]
  have injective : Function.Injective (fun j : Fin 128 => normalizerAddress shape tile j.val) := by
    intro x y same
    apply Fin.ext
    simp only [normalizerAddress] at same
    omega
  have same : h.choose=r := injective h.choose_spec.2.symm
  rw [same]

def queryGridTarget {α : Type} [Inhabited α] (a : Scalars α) (shape : Shape) (σ : ℝ)
    (p : Program shape 64) (s : State α) (_R : RegionName) (off : Nat) : Cell α :=
  rectangleTarget (queryRectangle shape p.tile)
    (fun i : TileIndex [64,shape.BK] => sourceQuery a shape σ (inputReaders shape s) p.tile i.1.val i.2.1.val) (a.literal 0) off

def keyValueGridTarget {α : Type} [Inhabited α] (a : Scalars α) (shape : Shape) (σ : ℝ)
    (p : Program shape 64) (s : State α) (R : RegionName) (off : Nat) : Cell α :=
  if R="dk" then rectangleTarget (keyRectangle shape p.tile)
    (fun i : TileIndex [64,shape.BK] => (sourceKeyValue a shape σ (inputReaders shape s) p.tile i.1.val i.2.1.val 0).1) (a.literal 0) off
  else rectangleTarget (valueRectangle shape p.tile)
    (fun i : TileIndex [64,shape.BV] => (sourceKeyValue a shape σ (inputReaders shape s) p.tile i.1.val 0 i.2.1.val).2) (a.literal 0) off

def forwardGridTarget {α : Type} [Inhabited α] (a : Scalars α) (shape : Shape) (σ : ℝ)
    (p : Program shape 128) (s : State α) (R : RegionName) (off : Nat) : Cell α :=
  if R="o" then rectangleTarget (outputRectangle shape p.tile 128)
    (fun i : TileIndex [128,shape.BV] => (sourceForward a shape σ (inputReaders shape s) p.tile i.1.val i.2.1.val).1) (a.literal 0) off
  else normalizerTarget shape p.tile
    (fun r => (sourceForward a shape σ (inputReaders shape s) p.tile r.val 0).2) (a.literal 0) off

theorem query_grid {α : Type} [Inhabited α] (a : Scalars α) (shape : Shape) (σ : ℝ) (s : State α) :
    ∃ final,runPrograms (queryLaunch a shape σ) (programs shape 64) s=some final ∧ SameInputs s final ∧
      (∀ p∈programs shape 64,∀ R off,(queryRectangle shape p.tile).writes 64 shape.BK R off →
        final.mem R off=queryGridTarget a shape σ p s R off) ∧
      (∀ R off,(∀ p∈programs shape 64,¬(queryRectangle shape p.tile).writes 64 shape.BK R off) → final.mem R off=s.mem R off) := by
  refine runPrograms_correct _ _ _ ?_ (query_write_unique shape) ?_ ?_ _ _
  · intro p R off member writes
    have eq := writes.1
    simp only [queryRectangle] at eq
    subst R
    simp at member
  · intro p before after same R off
    simp only [queryGridTarget,sameReaders shape before after same]
  · intro p before
    obtain ⟨final,executed,reads,frame⟩ := query_source a shape σ p.tile (withProgram p before)
    refine ⟨final,executed,?_,frame⟩
    intro R off write
    rcases write with ⟨rfl,⟨⟨r,e,⟨⟩⟩,active,rfl⟩⟩
    erw [queryGridTarget,rectangleTarget_at _ _ _ _ active]
    exact reads r e active

theorem keyValue_grid {α : Type} [Inhabited α] (a : Scalars α) (shape : Shape) (σ : ℝ) (s : State α)
    (keyWidth : 0<shape.BK) (valueWidth : 0<shape.BV) :
    ∃ final,runPrograms (keyValueLaunch a shape σ) (programs shape 64) s=some final ∧ SameInputs s final ∧
      (∀ p∈programs shape 64,∀ R off,keyValueWrites shape p R off → final.mem R off=keyValueGridTarget a shape σ p s R off) ∧
      (∀ R off,(∀ p∈programs shape 64,¬keyValueWrites shape p R off) → final.mem R off=s.mem R off) := by
  refine runPrograms_correct _ _ _ ?_ (keyValue_write_unique shape) ?_ ?_ _ _
  · intro p R off member writes
    rcases writes with ⟨eq,_⟩|⟨eq,_⟩ <;> subst R <;> simp [keyRectangle,queryRectangle,valueRectangle,outputRectangle] at member
  · intro p before after same R off
    simp only [keyValueGridTarget,sameReaders shape before after same]
  · intro p before
    obtain ⟨final,executed,keyReads,valueReads,frame⟩ := keyValue_source a shape σ p.tile (withProgram p before) keyWidth valueWidth
    refine ⟨final,executed,?_,frame⟩
    intro R off write
    rcases write with ⟨rfl,⟨⟨r,e,⟨⟩⟩,active,rfl⟩⟩|⟨rfl,⟨⟨r,v,⟨⟩⟩,active,rfl⟩⟩
    · simp only [keyValueGridTarget,keyRectangle,queryRectangle,ite_true]
      erw [rectangleTarget_at _ _ _ _ active]
      exact keyReads r e active
    · simp only [keyValueGridTarget,valueRectangle,outputRectangle,show ¬(("dv" : RegionName)="dk") by decide,ite_false]
      erw [rectangleTarget_at _ _ _ _ active]
      exact valueReads r v active

theorem forward_grid {α : Type} [Inhabited α] (a : Scalars α) (shape : Shape) (σ : ℝ) (s : State α)
    (valueWidth : 0<shape.BV) :
    ∃ final,runPrograms (forwardLaunch a shape σ) (programs shape 128) s=some final ∧ SameInputs s final ∧
      (∀ p∈programs shape 128,∀ R off,forwardWrites shape p R off → final.mem R off=forwardGridTarget a shape σ p s R off) ∧
      (∀ R off,(∀ p∈programs shape 128,¬forwardWrites shape p R off) → final.mem R off=s.mem R off) := by
  refine runPrograms_correct _ _ _ ?_ (forward_write_unique shape) ?_ ?_ _ _
  · intro p R off member writes
    rcases writes with ⟨eq,_⟩|⟨_,eq,_⟩ <;> subst R <;> simp [outputRectangle] at member
  · intro p before after same R off
    simp only [forwardGridTarget,sameReaders shape before after same]
  · intro p before
    obtain ⟨final,executed,outputReads,normReads,frame⟩ := forward_source a shape σ (withProgram p before) valueWidth
    rw [program_tile] at outputReads normReads frame
    refine ⟨final,executed,?_,frame⟩
    intro R off write
    rcases write with ⟨rfl,⟨⟨r,v,⟨⟩⟩,active,rfl⟩⟩|⟨writer,rfl,⟨r,active,rfl⟩⟩
    · simp only [forwardGridTarget,outputRectangle,ite_true]
      erw [rectangleTarget_at _ _ _ _ active]
      exact outputReads r v active
    · simp only [forwardGridTarget,show ¬(("z" : RegionName)="o") by decide,ite_false]
      erw [normalizerTarget_at _ _ _ _ r active]
      exact normReads r writer active

#axiomsClean rectangleTarget_at
#axiomsClean normalizerTarget_at
#axiomsClean query_grid
#axiomsClean keyValue_grid
#axiomsClean forward_grid
#specNonCircular queryGridTarget avoiding [runPrograms,queryLaunch,FP.Structural.exec,FP.Structural.run]
#specNonCircular keyValueGridTarget avoiding [runPrograms,keyValueLaunch,FP.Structural.exec,FP.Structural.run]
#specNonCircular forwardGridTarget avoiding [runPrograms,forwardLaunch,FP.Structural.exec,FP.Structural.run]
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel
