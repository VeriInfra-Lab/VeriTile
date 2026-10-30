import bench.repairs.tritonbench_g.rope_transform.Proofs.KernelMath

namespace VeriTile.Bench.Repairs.RopeTransform.KernelContract

theorem forward_not_backward : realPair 1 2 3 4=(-5,10) ∧
    VeriTile.Bench.Repairs.RopeBackwardTransform.Spec.pair 1 2 3 4=(11,2) := by
  norm_num [realPair,VeriTile.Bench.Repairs.RopeBackwardTransform.Spec.pair]

theorem fixed_coefficient_adjoint (a b u v c s : ℝ) :
    (realPair a b c s).1*u+(realPair a b c s).2*v=
      a*(u*c+v*s)+b*(v*c-u*s) := by simp only [realPair]; ring

theorem nonunit_not_inverse :
    let f:=realPair 1 2 2 0
    VeriTile.Bench.Repairs.RopeBackwardTransform.Spec.pair f.1 f.2 2 0≠(1,2) := by
  norm_num [realPair,VeriTile.Bench.Repairs.RopeBackwardTransform.Spec.pair,Prod.mk.injEq]

theorem formed_index_counterexample :
    let r : Request:=⟨1,2^30+1,1,1,1,by decide,by decide,by decide⟩
    r.rows<2^31 ∧ r.stride Bool.false<2^31 ∧
      outputLast r Bool.false=2^31+1 ∧ ¬outputLast r Bool.false<2^31 := by
  have one : (1 : Nat).nextPowerOfTwo=1 := by
    have low:=nextPower_covers 1
    have high:=nextPower_lt 1 (by decide)
    omega
  have two : (2 : Nat).nextPowerOfTwo/2=1 := by
    have low:=nextPower_covers 2
    have high:=nextPower_lt 2 (by decide)
    omega
  norm_num [outputLast,Request.rows,Request.stride,Request.heads,Request.dimension,
    Request.paddedHeads,Request.width,Request.paddedDimension,one,two]

#axiomsClean forward_not_backward
#axiomsClean fixed_coefficient_adjoint
#axiomsClean nonunit_not_inverse
#axiomsClean formed_index_counterexample
end VeriTile.Bench.Repairs.RopeTransform.KernelContract
