import bench.tritonbench_g.fp4_to_bf16.IO
import bench.tritonbench_g.fp4_to_bf16.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.Fp4ToBf16.ConversionIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16 FPValue Host RealSpec
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Both directions preserve the full domain, finite observation, safety and frame. -/
theorem logical_real_iff (rest : Algebra Word) (h : Input) (jobs : List Nat)
    (f : Bits.Nibble → ℝ) :
    (logicalRealIO rest h jobs).ImplementsOn (LogicalByteDomain h)
      (fun xs j => some (f (logicalNibble h xs j))) ↔
    (realIO rest h jobs).ImplementsOn (ByteDomain h)
      (fun xs => Fin.cases (fun j => some (f (nibble xs j.1))) (fun i => Fin.elim0 i)) :=
  Iff.rfl

theorem logical_real_implements (rest : Algebra Word) (h : Input)
    (storage : Finite.Storage h.shape.count) (jobs : List Nat)
    (fresh : h.input ≠ h.output) (gridSafe : grid h ≤ 2147483647)
    (order : jobs.Perm (List.range (grid h))) :
    (logicalRealIO rest h jobs).ImplementsOn (LogicalByteDomain h)
      (fun xs j => some (e2m1 (logicalNibble h xs j))) :=
  (logical_real_iff rest h jobs e2m1).mpr
    (real_implements rest h storage jobs fresh gridSafe order)
end
end VeriTile.Bench.Repairs.Fp4ToBf16.ConversionIO
