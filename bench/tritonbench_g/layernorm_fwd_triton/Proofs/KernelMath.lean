import bench.tritonbench_g.layernorm_fwd_triton.Proofs.KernelGraph
import bench.tritonbench_g.layernorm_fwd_triton.Proofs.Real
import bench.tritonbench_g.layernorm_fwd_triton.Contract

namespace VeriTile.Bench.Repairs.LayernormFwdTriton.Mathematics
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section

/-- Ideal arithmetic interpretation, independent of all native words. Only
source-reached operations are used; no FP-to-Real equality is asserted. -/
def ideal : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op a b => match op with
    | .add => a+b | .sub => a-b | .mul => a*b | .div => a/b | _ => 0
  unary := fun _ op a => match op with | .sqrt => Real.sqrt a | _ => 0
  cast := fun _ _ _ a => a
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ _ _ _ _ => 0
  reduceSum := fun _ {shape} ax keep xs out => match keep with
    | Bool.false => ∑ k, xs (TileShape.insertAxisIndex shape ax out k)
    | Bool.true => ∑ k, xs (TileShape.replaceAxisIndex shape ax out k)

theorem mean_real (N : Nat) (x : Nat → ℝ) :
    Graph.mean ideal N x=Contract.Math.mean N x := by
  unfold Graph.mean Graph.divide Graph.sum
  simp only [ideal,Graph.add,Graph.loadX,Graph.zero,TileShape.insertAxisIndex]
  change (∑ i : Fin 128, foldRange (RealProof.addStep x N) 0 N 128 (fun _ => 0) i)/(N:ℝ)=_
  rw [RealProof.accumulation_total]
  rfl

theorem variance_real (N : Nat) (x : Nat → ℝ) :
    Graph.variance ideal N x (Graph.mean ideal N x)=Contract.Math.variance N x := by
  rw [mean_real]
  have step : (fun off a (lane : Fin 128) =>
      let z := Graph.centered ideal N x (Contract.Math.mean N x) (off+lane.val)
      Graph.add ideal (a lane) (Graph.mul ideal z z))=
      RealProof.addStep (fun j => (x j-Contract.Math.mean N x)^2) N := by
    funext off a lane
    by_cases h : off+lane.val<N
    · simp [Graph.centered,Graph.loadX,Graph.sub,Graph.mul,Graph.add,Graph.zero,ideal,
        RealProof.addStep,h,pow_two]
    · simp [Graph.centered,Graph.mul,Graph.add,Graph.zero,ideal,RealProof.addStep,h]
  unfold Graph.variance Graph.divide Graph.sum
  rw [step]
  simp only [ideal,Graph.zero,TileShape.insertAxisIndex]
  change (∑ i : Fin 128, foldRange (RealProof.addStep (fun j => (x j-Contract.Math.mean N x)^2) N)
    0 N 128 (fun _ => 0) i)/(N:ℝ)=_
  rw [RealProof.accumulation_total]
  rfl

theorem output_real (N : Nat) (x w : Nat → ℝ) (eps : ℝ) (j : Nat) (hj : j<N) :
    Graph.output ideal N x w eps j=Contract.Math.output N x w eps j := by
  unfold Graph.output
  dsimp only
  rw [variance_real,mean_real]
  simp only [Graph.inverse,Graph.loadX,if_pos hj,
    Graph.add,Graph.sub,Graph.mul,ideal,Contract.Math.output,Contract.Math.inverse,one_div]

theorem variance_nonnegative (N : Nat) (x : Nat → ℝ) : 0≤Contract.Math.variance N x := by
  apply div_nonneg
  · exact Finset.sum_nonneg (fun _ _ => sq_nonneg _)
  · exact Nat.cast_nonneg N

theorem inverse_positive (N : Nat) (x : Nat → ℝ) (eps : ℝ) (heps : 0<eps) :
    0<Contract.Math.inverse N x eps := by
  apply inv_pos.mpr
  exact Real.sqrt_pos.2 (add_pos_of_nonneg_of_pos (variance_nonnegative N x) heps)

/-- Original source omitted the inner stride: these two different logical
coordinates write the same element. The repaired coordinates are distinct. -/
theorem original_stride_collision :
    (0*4+0*1+1:Nat)=(0*4+1*1+0:Nat) ∧
    (0*4+0*1+1*2:Nat)≠(0*4+1*1+0*2:Nat) := by decide

#axiomsClean mean_real
#axiomsClean variance_real
#axiomsClean output_real
#axiomsClean original_stride_collision
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.Mathematics
