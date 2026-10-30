import bench.repairs.tritonbench_g.triton_matmul.KernelOnly.Raw
import bench.repairs.tritonbench_g.triton_matmul.Proofs.RawAccess
import bench.repairs.tritonbench_g.triton_matmul.Proofs.RawSafetyLoop

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly.SourceSafety
open VeriTile.Triton FP.Structural FPValue
noncomputable section
variable {α : Type} [Inhabited α]

/-- The coordinate facts are derived from the original launch geometry. Every
load/store obligation is then discharged on the literal signed source trace. -/
theorem source_safe (M : Algebra α) (D : DotAcc α) (p : Parameters) (st : Storage)
    (bounds : RegionName → Nat) (initial : State α)
    (hpm : (RawSource.control p initial).pidM = (pidM p initial : Int))
    (hpn : (RawSource.control p initial).pidN = (pidN p initial : Int))
    (ha : ∀ q (i : AIndex p), i.2.1.val+q*BK p<p.K → aOffset p initial q i<bounds p.A)
    (hb : ∀ q (i : BIndex p), i.1.val+q*BK p<p.K → bOffset p initial q i<bounds p.B)
    (hc : ∀ i : CIndex p, active p initial i → cOffset p initial i<bounds p.C) :
    TraceSafe (backend M D) bounds (Source.typed p st) initial := by
  apply RawSafety.source_safe
  · intro q i hi
    rw [RawSource.addressA_model p initial hpm q i hi]
    have hk : i.2.1.val < p.K-q*BK p := by
      simpa only [RawSource.innerMask_model,decide_eq_true_eq] using hi
    exact ha q i (by omega)
  · intro q i hi
    rw [RawSource.addressB_model p initial hpn q i hi]
    have hk : i.1.val < p.K-q*BK p := by
      simpa only [RawSource.innerMask_model,decide_eq_true_eq] using hi
    exact hb q i (by omega)
  · intro i hi
    have hactive : active p initial i := by
      simpa only [RawSource.active_model p initial hpm hpn,decide_eq_true_eq] using hi
    rw [RawSource.outputOffset_model p initial hpm hpn i hactive]
    exact hc i hactive

#axiomsClean source_safe
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly.SourceSafety
