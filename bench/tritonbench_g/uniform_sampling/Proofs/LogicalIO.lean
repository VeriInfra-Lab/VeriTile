import bench.tritonbench_g.uniform_sampling.IO
import bench.tritonbench_g.uniform_sampling.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.UniformSampling.KernelPublic
open VeriTile.Triton FP.Structural
noncomputable section

theorem logical_iff {α : Type} [Inhabited α] (M : Algebra α) (y : RegionName) (N : Nat)
    (seed offset : Int) (lower upper : Float32Bits) (B : Nat)
    (hB : Random.ArangeSizeDomain B) (pid : Nat) (f : Fin 4 → Fin B → α) :
    (logicalIO M y N seed offset lower upper B hB pid).Implements (fun _ => f) ↔
      (io M y N seed offset lower upper B hB pid).Implements
        (fun _ => Fin.cases (fun j => f j.1 j.2.1) (fun z => Fin.elim0 z)) := by rfl
end
end VeriTile.Bench.Repairs.UniformSampling.KernelPublic
