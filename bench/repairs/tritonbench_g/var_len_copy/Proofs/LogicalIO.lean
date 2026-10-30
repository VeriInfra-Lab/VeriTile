import bench.repairs.tritonbench_g.var_len_copy.IO
import bench.repairs.tritonbench_g.var_len_copy.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.VarLenCopy.PublicIO
open VeriTile.Triton
noncomputable section

theorem logical_iff (a : Segment) :
    (logicalIO a).ImplementsOn (logicalDomain a) (fun xs _ _ _ => xs.source) ↔
      (io a).RegionalImplementsOn3D (metadataDomain a) (formula a) := by rfl
end
end VeriTile.Bench.Repairs.VarLenCopy.PublicIO
