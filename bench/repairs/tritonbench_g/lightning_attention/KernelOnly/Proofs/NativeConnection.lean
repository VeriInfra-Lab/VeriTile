import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Proofs.NativeResults
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Proofs.NativeSafety

namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.NativeConnection
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.LightningAttention.Native (Instructions Plan LocalConformance instructions_eq)
noncomputable section

theorem forward (d : Host.Dimensions) (actual selected : Instructions) (plan : Plan)
    (numeric : LocalConformance actual selected) :
    (NativeIO.forward actual plan d).ImplementsCells (NativeReference.forward selected plan d) := by
  rw [instructions_eq actual selected numeric]
  exact MaterializedProgramIO.ImplementsCells.intro
    (fun vals bounds backed initial _ => NativeSafety.forward selected plan d vals bounds backed initial)
    (NativeResults.forward selected plan d)

theorem backward (d : Host.Dimensions) (actual selected : Instructions) (plan : Plan)
    (numeric : LocalConformance actual selected) :
    (NativeIO.backward actual plan d).ImplementsCells (NativeReference.backward selected plan d) := by
  rw [instructions_eq actual selected numeric]
  exact MaterializedProgramIO.ImplementsCells.intro
    (fun vals bounds backed initial _ => NativeSafety.backward selected plan d vals bounds backed initial)
    (NativeResults.backward selected plan d)

#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.NativeConnection
