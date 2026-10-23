import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.Combined
import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.RealGrid

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
set_option maxHeartbeats 3000000

/-- Independent mathematical targets for the parent backward kernel's
three tagged output regions. DQ/DK retain the value-shard coordinate. -/
def backwardMathTarget (shape : Shape) (σ : ℝ) (x : TensorInputs ℝ)
    (p : Program shape 64) (R : RegionName) (off : Nat) : Cell ℝ :=
  if R="dq" then rectangleTarget (queryRectangle shape p.tile)
    (fun i : TileIndex [64,shape.BK] => queryPartial shape σ x p.head.val p.value.val (p.chunk.val*64+i.1.val) i.2.1.val) 0 off
  else if R="dk" then rectangleTarget (keyRectangle shape p.tile)
    (fun i : TileIndex [64,shape.BK] => keyPartial shape σ x p.head.val p.value.val (p.chunk.val*64+i.1.val) i.2.1.val) 0 off
  else rectangleTarget (valueRectangle shape p.tile)
    (fun i : TileIndex [64,shape.BV] => valueGradient shape σ x p.head.val (p.chunk.val*64+i.1.val) (p.value.val*shape.BV+i.2.1.val)) 0 off

/-- Successful complete combined backward source grid at the actual Real
ABI scale, all three independent mathematical outputs and full tagged frame. -/
theorem backward_grid_real (shape : Shape) (accepted : ShapeContract shape) (σ : ℝ) (s : State ℝ) :
    ∃ final,runPrograms (combinedLaunch realScalars shape σ) (programs shape 64) s=some final ∧ SameInputs s final ∧
      (∀ p∈programs shape 64,∀ R off,backwardWrites shape p R off →
        final.mem R off=backwardMathTarget shape σ (inputReaders shape s) p R off) ∧
      (∀ R off,(∀ p∈programs shape 64,¬backwardWrites shape p R off) → final.mem R off=s.mem R off) := by
  obtain ⟨final,executed,inputs,reads,frame⟩ := backward_grid realScalars shape σ s
    (by have := accepted.key_positive; have := accepted.key_fits shape; omega) (accepted.value_tile_positive shape)
  refine ⟨final,executed,inputs,?_,frame⟩
  intro p member R off writes
  rw [reads p member R off writes]
  have domain := program_domain shape accepted 64 (by decide) p
  rcases writes with ⟨rfl,⟨⟨r,e,⟨⟩⟩,active,rfl⟩⟩|
    ⟨rfl,⟨⟨r,e,⟨⟩⟩,active,rfl⟩⟩|⟨rfl,⟨⟨r,v,⟨⟩⟩,active,rfl⟩⟩
  · simp only [backwardGridTarget,backwardMathTarget,queryGridTarget,queryRectangle,ite_true]
    erw [rectangleTarget_at _ _ _ _ active,rectangleTarget_at _ _ _ _ active]
    apply congrArg (Cell.mk .real)
    have he : e.val<shape.K := by simpa [queryRectangle,Rectangle.active,Program.tile] using active.2
    exact Math.query_whole_real (σ := σ) sourcePlans shape (inputReaders shape s)
      p.tile r.val e.val rfl domain.keyFits r.isLt active.1 he
  · simp only [backwardGridTarget,backwardMathTarget,keyValueGridTarget,keyRectangle,queryRectangle,
      show ¬(("dk" : RegionName)="dq") by decide,ite_false,ite_true]
    erw [rectangleTarget_at _ _ _ _ active,rectangleTarget_at _ _ _ _ active]
    apply congrArg (Cell.mk .real)
    have he : e.val<shape.K := by simpa [keyRectangle,queryRectangle,Rectangle.active,Program.tile] using active.2
    exact congrArg Prod.fst (Math.keyValue_whole_real (σ := σ) sourcePlans shape (inputReaders shape s)
      p.tile r.val e.val 0 rfl domain.keyFits active.1 he (by simpa using domain.value) r.isLt)
  · simp only [backwardGridTarget,backwardMathTarget,keyValueGridTarget,valueRectangle,outputRectangle,
      show ¬(("dv" : RegionName)="dq") by decide,show ¬(("dv" : RegionName)="dk") by decide,ite_false]
    erw [rectangleTarget_at _ _ _ _ active,rectangleTarget_at _ _ _ _ active]
    apply congrArg (Cell.mk .real)
    exact congrArg Prod.snd (Math.keyValue_whole_real (σ := σ) sourcePlans shape (inputReaders shape s)
      p.tile r.val 0 v.val rfl domain.keyFits active.1 domain.keyPositive active.2 r.isLt)

#axiomsClean backward_grid_real
#specNonCircular backwardMathTarget avoiding [combinedLaunch,combinedKernel,FP.Structural.exec,runPrograms,queryValues,backwardValues]
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel
