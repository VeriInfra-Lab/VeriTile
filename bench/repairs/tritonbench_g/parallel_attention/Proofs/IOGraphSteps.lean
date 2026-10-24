import bench.repairs.tritonbench_g.parallel_attention.Proofs.IOSchedule

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.IOGraph
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
open IOData
noncomputable section
variable {α : Type} [Inhabited α]

def queryTarget (M : Algebra α) (D : DotPrimitive α) (shape : Shape) (scale : ℝ)
    (p : Program shape 64) (s : State α) (_R : RegionName) (off : Nat) : Cell α :=
  rectangleTarget (queryRectangle shape p.tile)
    (fun i=>M.cast (some .fp32) .real .real (queryValues M D (dqConfig shape scale p.tile) s i)) default off

def keyValueTarget (M : Algebra α) (D : DotPrimitive α) (shape : Shape) (scale : ℝ)
    (p : Program shape 64) (s : State α) (R : RegionName) (off : Nat) : Cell α :=
  if R="dk" then rectangleTarget (keyRectangle shape p.tile)
    (fun i=>M.cast (some .fp32) .real .real ((backwardValues M D (backwardConfig shape scale p.tile) s).key i)) default off
  else rectangleTarget (valueRectangle shape p.tile)
    (fun i=>M.cast (some .fp32) .real .real ((backwardValues M D (backwardConfig shape scale p.tile) s).value i)) default off

def forwardTarget (M : Algebra α) (D : DotPrimitive α) (shape : Shape) (scale : ℝ)
    (p : Program shape 128) (s : State α) (R : RegionName) (off : Nat) : Cell α :=
  if R="o" then rectangleTarget (outputRectangle shape p.tile 128)
    (fun i=>M.cast (some .fp32) .real .real
      ((forwardValues M D (forwardConfig shape scale) (withProgram p s)).output i)) default off
  else normalizerTarget shape p.tile
    (fun r=>M.cast (some .fp32) .real .real
      ((forwardValues M D (forwardConfig shape scale) (withProgram p s)).normalizer ⟨r,()⟩)) default off

theorem forward_normalizer_at (M : Algebra α) (D : DotPrimitive α) (shape : Shape) (scale : ℝ)
    (p : Program shape 128) (s : State α) (r : Fin 128) (active : p.chunk.val*128+r.val<shape.T) :
    forwardTarget M D shape scale p s "z" (normalizerAddress shape p.tile r.val)=
      Cell.mk .real (M.cast (some .fp32) .real .real
        ((forwardValues M D (forwardConfig shape scale) (withProgram p s)).normalizer ⟨r,()⟩)) := by
  simp only [forwardTarget,show ¬(("z" : RegionName)="o") by decide,ite_false]
  exact normalizerTarget_at shape p.tile _ default r active

theorem query_step (M : Algebra α) (D : DotPrimitive α) (shape : Shape) (scale : ℝ)
    (p : Program shape 64) (s : State α) :
    ∃ final,(queryJob M D shape scale p).run s=some final ∧
      (∀ R off,(queryRectangle shape p.tile).writes 64 shape.BK R off →
        final.mem R off=queryTarget M D shape scale p s R off) ∧
      (∀ R off,¬(queryRectangle shape p.tile).writes 64 shape.BK R off → final.mem R off=s.mem R off) := by
  obtain ⟨final,executed,memory⟩:=query_fp32 M D (dqConfig shape scale p.tile) s
  refine ⟨final,executed,?_,?_⟩
  · intro R off write
    rcases write with ⟨rfl,⟨i,active,rfl⟩⟩
    erw [memory,NativeStage.query_stores_algebra,rectangleStore_read _ _ _ _ active,
      queryTarget,rectangleTarget_at _ _ _ _ active]
  · intro R off outside
    rw [memory,NativeStage.query_stores_algebra]
    exact rectangleStore_frame _ _ _ R off outside

theorem keyValue_step (M : Algebra α) (D : DotPrimitive α) (shape : Shape) (scale : ℝ)
    (p : Program shape 64) (s : State α) :
    ∃ final,(keyValueJob M D shape scale p).run s=some final ∧
      (∀ R off,keyValueWrites shape p R off → final.mem R off=keyValueTarget M D shape scale p s R off) ∧
      (∀ R off,¬keyValueWrites shape p R off → final.mem R off=s.mem R off) := by
  obtain ⟨final,executed,memory⟩:=backward_fp32 M D (backwardConfig shape scale p.tile) s
  refine ⟨final,executed,?_,?_⟩
  · intro R off write
    rcases write with ⟨rfl,⟨i,active,rfl⟩⟩|⟨rfl,⟨i,active,rfl⟩⟩
    · rw [memory,NativeStage.backward_stores_algebra,rectangleStore_frame]
      · erw [rectangleStore_read _ _ _ _ active]
        simp only [keyValueTarget,keyRectangle,queryRectangle,ite_true]
        erw [rectangleTarget_at _ _ _ _ active]
      · intro h
        exact (by decide : ("dk" : RegionName)≠"dv") h.1
    · erw [memory,NativeStage.backward_stores_algebra,rectangleStore_read _ _ _ _ active]
      simp only [keyValueTarget,valueRectangle,outputRectangle,
        show ¬(("dv" : RegionName)="dk") by decide,ite_false]
      erw [rectangleTarget_at _ _ _ _ active]
  · intro R off outside
    rw [memory,NativeStage.backward_stores_algebra,rectangleStore_frame,rectangleStore_frame]
    · exact fun h=>outside (Or.inl h)
    · exact fun h=>outside (Or.inr h)

theorem forward_step (M : Algebra α) (D : DotPrimitive α) (shape : Shape) (scale : ℝ)
    (p : Program shape 128) (s : State α) :
    ∃ final,(forwardJob M D shape scale p).run s=some final ∧
      (∀ R off,forwardWrites shape p R off → final.mem R off=forwardTarget M D shape scale p s R off) ∧
      (∀ R off,¬forwardWrites shape p R off → final.mem R off=s.mem R off) := by
  obtain ⟨final,executed,memory⟩:=forward_fp32 M D (forwardConfig shape scale) (withProgram p s)
  change final.mem=(outputMemory M (forwardConfig shape scale) (withProgram p s)
    (forwardValues M D (forwardConfig shape scale) (withProgram p s))).mem at memory
  rw [NativeStage.forward_stores_algebra,program_tile] at memory
  refine ⟨final,executed,?_,?_⟩
  · intro R off write
    rcases write with ⟨rfl,⟨i,active,rfl⟩⟩|⟨writer,rfl,⟨r,active,rfl⟩⟩
    · rw [memory,normalizerStore_frame]
      · erw [rectangleStore_read _ _ _ _ active]
        simp only [forwardTarget,outputRectangle,ite_true]
        erw [rectangleTarget_at _ _ _ _ active]
      · intro h
        exact (by decide : ("o" : RegionName)≠"z") h.2.1
    · erw [memory,normalizerStore_read _ _ _ _ r writer active]
      simp only [forwardTarget,show ¬(("z" : RegionName)="o") by decide,ite_false]
      erw [normalizerTarget_at _ _ _ _ r active]
      rfl
  · intro R off outside
    rw [memory,normalizerStore_frame,rectangleStore_frame]
    · rfl
    · exact fun h=>outside (Or.inl h)
    · exact fun h=>outside (Or.inr h)

#axiomsClean query_step
#axiomsClean keyValue_step
#axiomsClean forward_step
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.IOGraph
