import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.NativeRawLoads
import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.Normalize
import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.NativeReference

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 10000000
set_option linter.unusedSimpArgs false

theorem nativeValuesAt_moment {α : Type} [Inhabited α] (a : Arithmetic α) (z : ZeroLaws a)
    (fp8 : Bool) (p : Arguments) (s : State α) (epsilon theta : α)
    (t : Nat) (row : Fin 16) (k : Fin 64) :
    (nativeValuesAt a fp8 p s t).moments (row,k,PUnit.unit)=
      moment a.toScalar (parameters p epsilon theta)
        (rawRow a fp8 p s (s.pids 0) ((programRow p s*16+row.val)%p.M)) k t := by
  induction t with
  | zero => exact z.literal
  | succ t ih =>
    change a.add ((nativeValuesAt a fp8 p s t).moments (row,k,PUnit.unit))
      (a.powi (xTile a p s (nativeValuesAt a fp8 p s t) t (row,k,PUnit.unit)) 2)=_
    rw [ih,native_xTile_reference a z fp8 p s epsilon theta]
    rfl

theorem nativeValuesAt_accumulation {α : Type} [Inhabited α] (a : Arithmetic α) (z : ZeroLaws a)
    (fp8 : Bool) (p : Arguments) (s : State α) (epsilon theta : α)
    (t : Nat) (row : Fin 16) (n : Fin 64) :
    (nativeValuesAt a fp8 p s t).accumulator (row,n,PUnit.unit)=
      nativeAccumulation a.toScalar (parameters p epsilon theta)
        (rawRow a fp8 p s (s.pids 0) ((programRow p s*16+row.val)%p.M))
        (fun _ => a.plan) ((programCol p s*64+n.val)%p.N) t := by
  induction t with
  | zero => exact z.literal
  | succ t ih =>
    change (tensor a (batch:=[]) (scaledTile a fp8 p s (nativeValuesAt a fp8 p s t) t)
        (weightTile a fp8 p s (nativeValuesAt a fp8 p s t) t)
        (nativeValuesAt a fp8 p s t).accumulator
        (row,n,PUnit.unit))=_
    simp only [tensor]
    rw [ih]
    simp only [native_scaledTile_reference a z fp8 p s epsilon theta,
      native_weightTile_reference a z fp8 p s epsilon theta t (s.pids 0)
        ((programRow p s*16+row.val)%p.M),chain_plan,nativeAccumulation,
      RmsMatmulRbe.dotBlock,RmsMatmulRbe.instruction]

theorem native_normalized_reference {α : Type} [Inhabited α] (a : Arithmetic α) (z : ZeroLaws a)
    (fp8 : Bool) (p : Arguments) (s : State α) (epsilon theta : α)
    (row : Fin 16) (n : Fin 64) :
    normalized a p epsilon (nativeValuesAt a fp8 p s ((p.K+63)/64)).moments
      (nativeValuesAt a fp8 p s ((p.K+63)/64)).accumulator (row,n,PUnit.unit)=
      nativeReferenceProjection a.toScalar (parameters p epsilon theta)
        (rawRow a fp8 p s (s.pids 0) ((programRow p s*16+row.val)%p.M))
        (fun _ _ => a.plan) a.reduction ((programCol p s*64+n.val)%p.N) := by
  have hm : (fun k : Fin 64 => moment a.toScalar (parameters p epsilon theta)
      (rawRow a fp8 p s (s.pids 0) ((programRow p s*16+row.val)%p.M)) k ((p.K+63)/64))=
      referenceMoment a.toScalar (parameters p epsilon theta)
        (rawRow a fp8 p s (s.pids 0) ((programRow p s*16+row.val)%p.M)) := by
    funext k
    exact moment_reference a.toScalar (parameters p epsilon theta) _ k
  simp only [normalized,rowNorm,nativeReferenceProjection,referenceNorm,
    nativeValuesAt_accumulation a z fp8 p s epsilon theta,native_accumulation_reference,
    reduction_plan,←moment_reference,parameters,RmsMatmulRbe.steps,
    nativeValuesAt_moment a z fp8 p s epsilon theta]
  simp only [parameters] at hm
  rw [hm]

#axiomsClean nativeValuesAt_moment
#axiomsClean nativeValuesAt_accumulation
#axiomsClean native_normalized_reference
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
