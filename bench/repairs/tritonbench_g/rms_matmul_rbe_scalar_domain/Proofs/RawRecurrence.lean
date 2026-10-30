import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.RawLoads
import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.Normalize

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 10000000
set_option linter.unusedSimpArgs false

theorem valuesAt_moment {α : Type} [Inhabited α] (a : Arithmetic α) (z : ZeroLaws a)
    (fp8 : Bool) (p : Arguments) (s : State α) (epsilon theta : α)
    (t : Nat) (row : Fin 16) (k : Fin 64) :
    (valuesAt a fp8 p s t).moments (row,k,PUnit.unit)=
      moment a.toScalar (parameters p epsilon theta)
        (rawRow a fp8 p s (s.pids 0) ((programRow p s*16+row.val)%p.M)) k t := by
  induction t with
  | zero => exact z.literal
  | succ t ih =>
    change a.add ((valuesAt a fp8 p s t).moments (row,k,PUnit.unit))
      (a.powi (xTile a p s (valuesAt a fp8 p s t) t (row,k,PUnit.unit)) 2)=_
    rw [ih,xTile_reference a z fp8 p s epsilon theta]
    rfl

theorem valuesAt_accumulation {α : Type} [Inhabited α] (a : Arithmetic α) (z : ZeroLaws a)
    (fp8 : Bool) (p : Arguments) (s : State α) (epsilon theta : α)
    (t : Nat) (row : Fin 16) (n : Fin 64) :
    (valuesAt a fp8 p s t).accumulator (row,n,PUnit.unit)=
      accumulation a.toScalar (parameters p epsilon theta)
        (rawRow a fp8 p s (s.pids 0) ((programRow p s*16+row.val)%p.M))
        (fun _ => a.plan) ((programCol p s*64+n.val)%p.N) t := by
  induction t with
  | zero => exact z.literal
  | succ t ih =>
    change a.add ((valuesAt a fp8 p s t).accumulator (row,n,PUnit.unit))
      (tensor a (batch:=[]) (scaledTile a fp8 p s (valuesAt a fp8 p s t) t)
        (weightTile a fp8 p s (valuesAt a fp8 p s t) t) (fun _ => a.zero)
        (row,n,PUnit.unit))=_
    simp only [tensor]
    rw [ih]
    simp only [scaledTile_reference a z fp8 p s epsilon theta,
      weightTile_reference a z fp8 p s epsilon theta t (s.pids 0)
        ((programRow p s*16+row.val)%p.M),chain_plan,accumulation,
      RmsMatmulRbe.dotBlock,RmsMatmulRbe.instruction]

theorem normalized_reference {α : Type} [Inhabited α] (a : Arithmetic α) (z : ZeroLaws a)
    (fp8 : Bool) (p : Arguments) (s : State α) (epsilon theta : α)
    (row : Fin 16) (n : Fin 64) :
    normalized a p epsilon (valuesAt a fp8 p s ((p.K+63)/64)).moments
      (valuesAt a fp8 p s ((p.K+63)/64)).accumulator (row,n,PUnit.unit)=
      referenceProjection a.toScalar (parameters p epsilon theta)
        (rawRow a fp8 p s (s.pids 0) ((programRow p s*16+row.val)%p.M))
        (fun _ _ => a.plan) a.reduction ((programCol p s*64+n.val)%p.N) := by
  rw [←projection_reference]
  simp only [normalized,rowNorm,projection,norm,parameters,RmsMatmulRbe.steps,
    valuesAt_accumulation a z fp8 p s epsilon theta,
    valuesAt_moment a z fp8 p s epsilon theta,reduction_plan]

#axiomsClean valuesAt_moment
#axiomsClean valuesAt_accumulation
#axiomsClean normalized_reference
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
