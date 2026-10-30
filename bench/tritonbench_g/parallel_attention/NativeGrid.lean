import bench.tritonbench_g.parallel_attention.GridTargets
import bench.tritonbench_g.parallel_attention.NativeObserved

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.NativeStage
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
set_option maxHeartbeats 3000000

def queryGridTarget (a : Scalars Word) (d : NativePlans.Descriptor) (σ : ℝ)
    (p : Program d.shape 64) (s : State Word) (_R : RegionName) (off : Nat) : Cell Word :=
  rectangleTarget (queryRectangle d.shape p.tile)
    (fun i : TileIndex [64,d.shape.BK] => queryTarget a d σ p.tile s i.1 i.2.1) (a.literal 0) off

def keyValueGridTarget (a : Scalars Word) (d : NativePlans.Descriptor) (σ : ℝ)
    (p : Program d.shape 64) (s : State Word) (R : RegionName) (off : Nat) : Cell Word :=
  if R="dk" then rectangleTarget (keyRectangle d.shape p.tile)
    (fun i : TileIndex [64,d.shape.BK] => keyTarget a d σ p.tile s i.1 i.2.1) (a.literal 0) off
  else rectangleTarget (valueRectangle d.shape p.tile)
    (fun i : TileIndex [64,d.shape.BV] => valueTarget a d σ p.tile s i.1 i.2.1) (a.literal 0) off

def forwardGridTarget (a : Scalars Word) (d : NativePlans.Descriptor) (σ : ℝ)
    (p : Program d.shape 128) (s : State Word) (R : RegionName) (off : Nat) : Cell Word :=
  if R="o" then rectangleTarget (outputRectangle d.shape p.tile 128)
    (fun i : TileIndex [128,d.shape.BV] => outputTarget a d σ (withProgram p s) i.1 i.2.1) (a.literal 0) off
  else ParallelAttentionKernel.normalizerTarget d.shape p.tile
    (fun r => normalizerTarget a d σ (withProgram p s) r) (a.literal 0) off

theorem queryTarget_inputs (a : Scalars Word) (d : NativePlans.Descriptor) (σ : ℝ) (tile : Tile)
    (s t : State Word) (same : SameInputs s t) (r : Fin 64) (e : Fin d.shape.BK) :
    queryTarget a d σ tile s r e=queryTarget a d σ tile t r e := by
  unfold queryTarget
  rw [query_values_inputs _ _ _ _ _ _ _ (entry_sameInputs s t same)]

theorem keyTarget_inputs (a : Scalars Word) (d : NativePlans.Descriptor) (σ : ℝ) (tile : Tile)
    (s t : State Word) (same : SameInputs s t) (r : Fin 64) (e : Fin d.shape.BK) :
    keyTarget a d σ tile s r e=keyTarget a d σ tile t r e := by
  unfold keyTarget
  rw [keyValue_values_inputs _ _ _ _ _ _ _ (entry_sameInputs s t same)]

theorem valueTarget_inputs (a : Scalars Word) (d : NativePlans.Descriptor) (σ : ℝ) (tile : Tile)
    (s t : State Word) (same : SameInputs s t) (r : Fin 64) (v : Fin d.shape.BV) :
    valueTarget a d σ tile s r v=valueTarget a d σ tile t r v := by
  unfold valueTarget
  rw [keyValue_values_inputs _ _ _ _ _ _ _ (entry_sameInputs s t same)]

theorem outputTarget_inputs (a : Scalars Word) (d : NativePlans.Descriptor) (σ : ℝ)
    (p : Program d.shape 128) (s t : State Word) (same : SameInputs s t) (r : Fin 128) (v : Fin d.shape.BV) :
    outputTarget a d σ (withProgram p s) r v=outputTarget a d σ (withProgram p t) r v := by
  unfold outputTarget
  rw [program_tile,program_tile]
  rw [forward_values_inputs _ _ _ _ _ _ (entry_sameInputs (withProgram p s) (withProgram p t) same) (by rfl)]

theorem normalizerTarget_inputs (a : Scalars Word) (d : NativePlans.Descriptor) (σ : ℝ)
    (p : Program d.shape 128) (s t : State Word) (same : SameInputs s t) (r : Fin 128) :
    normalizerTarget a d σ (withProgram p s) r=normalizerTarget a d σ (withProgram p t) r := by
  unfold normalizerTarget
  rw [program_tile,program_tile]
  rw [forward_values_inputs _ _ _ _ _ _ (entry_sameInputs (withProgram p s) (withProgram p t) same) (by rfl)]

/-- Every submitted DQ program executes its entire shared quotation, and
all value-shard scratch cells read the independent fixed native graph. -/
theorem query_grid (actual reference : Scalars Word) (laws : PrimitiveLaws actual reference)
    (d : NativePlans.Descriptor) (σ : ℝ) (s : State Word) :
    ∃ final,runPrograms (fun p => queryRun actual d σ p.tile) (programs d.shape 64) s=some final ∧ SameInputs s final ∧
      (∀ p∈programs d.shape 64,∀ R off,(queryRectangle d.shape p.tile).writes 64 d.shape.BK R off →
        final.mem R off=queryGridTarget reference d σ p s R off) ∧
      (∀ R off,(∀ p∈programs d.shape 64,¬(queryRectangle d.shape p.tile).writes 64 d.shape.BK R off) → final.mem R off=s.mem R off) := by
  refine runPrograms_correct _ _ _ ?_ (query_write_unique d.shape) ?_ ?_ _ _
  · intro p R off member writes
    have eq := writes.1
    simp only [queryRectangle] at eq
    subst R
    simp at member
  · intro p before after same R off
    unfold queryGridTarget
    congr 1
    funext i
    exact queryTarget_inputs reference d σ p.tile before after same i.1 i.2.1
  · intro p before
    obtain ⟨graph,final,_,executed,reads,frame⟩ := query_native actual reference laws d σ p.tile before
    refine ⟨final,executed,?_,frame⟩
    intro R off write
    rcases write with ⟨rfl,⟨⟨r,e,⟨⟩⟩,active,rfl⟩⟩
    erw [queryGridTarget,rectangleTarget_at _ _ _ _ active]
    exact reads r e active

theorem keyValue_grid (actual reference : Scalars Word) (laws : PrimitiveLaws actual reference)
    (d : NativePlans.Descriptor) (σ : ℝ) (s : State Word) :
    ∃ final,runPrograms (fun p => keyValueRun actual d σ p.tile) (programs d.shape 64) s=some final ∧ SameInputs s final ∧
      (∀ p∈programs d.shape 64,∀ R off,keyValueWrites d.shape p R off → final.mem R off=keyValueGridTarget reference d σ p s R off) ∧
      (∀ R off,(∀ p∈programs d.shape 64,¬keyValueWrites d.shape p R off) → final.mem R off=s.mem R off) := by
  refine runPrograms_correct _ _ _ ?_ (keyValue_write_unique d.shape) ?_ ?_ _ _
  · intro p R off member writes
    rcases writes with ⟨eq,_⟩|⟨eq,_⟩ <;> subst R <;> simp [keyRectangle,queryRectangle,valueRectangle,outputRectangle] at member
  · intro p before after same R off
    unfold keyValueGridTarget
    split
    · congr 1; funext i; exact keyTarget_inputs reference d σ p.tile before after same i.1 i.2.1
    · congr 1; funext i; exact valueTarget_inputs reference d σ p.tile before after same i.1 i.2.1
  · intro p before
    obtain ⟨graph,final,_,executed,keyReads,valueReads,frame⟩ := keyValue_native actual reference laws d σ p.tile before
    refine ⟨final,executed,?_,frame⟩
    intro R off write
    rcases write with ⟨rfl,⟨⟨r,e,⟨⟩⟩,active,rfl⟩⟩|⟨rfl,⟨⟨r,v,⟨⟩⟩,active,rfl⟩⟩
    · simp only [keyValueGridTarget,keyRectangle,queryRectangle,ite_true]
      erw [rectangleTarget_at _ _ _ _ active]
      exact keyReads r e active
    · simp only [keyValueGridTarget,valueRectangle,outputRectangle,show ¬(("dv" : RegionName)="dk") by decide,ite_false]
      erw [rectangleTarget_at _ _ _ _ active]
      exact valueReads r v active

theorem forward_grid (actual reference : Scalars Word) (laws : PrimitiveLaws actual reference)
    (d : NativePlans.Descriptor) (σ : ℝ) (s : State Word) :
    ∃ final,runPrograms (fun p before => forwardRun actual d σ (withProgram p before)) (programs d.shape 128) s=some final ∧ SameInputs s final ∧
      (∀ p∈programs d.shape 128,∀ R off,forwardWrites d.shape p R off → final.mem R off=forwardGridTarget reference d σ p s R off) ∧
      (∀ R off,(∀ p∈programs d.shape 128,¬forwardWrites d.shape p R off) → final.mem R off=s.mem R off) := by
  refine runPrograms_correct _ _ _ ?_ (forward_write_unique d.shape) ?_ ?_ _ _
  · intro p R off member writes
    rcases writes with ⟨eq,_⟩|⟨_,eq,_⟩ <;> subst R <;> simp [outputRectangle] at member
  · intro p before after same R off
    unfold forwardGridTarget
    split
    · congr 1; funext i; exact outputTarget_inputs reference d σ p before after same i.1 i.2.1
    · congr 1; funext r; exact normalizerTarget_inputs reference d σ p before after same r
  · intro p before
    obtain ⟨graph,final,_,executed,outputReads,normReads,frame⟩ := forward_native actual reference laws d σ (withProgram p before)
    rw [program_tile] at outputReads normReads frame
    refine ⟨final,executed,?_,frame⟩
    intro R off write
    rcases write with ⟨rfl,⟨⟨r,v,⟨⟩⟩,active,rfl⟩⟩|⟨writer,rfl,⟨r,active,rfl⟩⟩
    · simp only [forwardGridTarget,outputRectangle,ite_true]
      erw [rectangleTarget_at _ _ _ _ active]
      exact outputReads r v active
    · simp only [forwardGridTarget,show ¬(("z" : RegionName)="o") by decide,ite_false]
      erw [ParallelAttentionKernel.normalizerTarget_at _ _ _ _ r active]
      exact normReads r writer active

#axiomsClean query_grid
#axiomsClean keyValue_grid
#axiomsClean forward_grid
#specNonCircular queryGridTarget avoiding [runPrograms,queryRun,FP.Structural.exec,FP.Structural.run]
#specNonCircular keyValueGridTarget avoiding [runPrograms,keyValueRun,FP.Structural.exec,FP.Structural.run]
#specNonCircular forwardGridTarget avoiding [runPrograms,forwardRun,FP.Structural.exec,FP.Structural.run]
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.NativeStage
