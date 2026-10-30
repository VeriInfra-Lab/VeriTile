import bench.repairs.tritonbench_g.var_len_copy.Proofs.IOExecution
import VeriTile.Triton.Memory.RegionalKernelIOOn

namespace VeriTile.Bench.Repairs.VarLenCopy.PublicIO
open VeriTile.Triton
noncomputable section

theorem implements (a : Segment) :
    (io a).RegionalImplementsOn3D (metadataDomain a) (formula a) := by
  apply UKernelIO.RegionalImplementsOn3D.intro
  · intro bounds s xs domain binding hb
    exact source_safe a s xs domain binding bounds hb
  · exact source_result a

#axiomsClean implements
end
end VeriTile.Bench.Repairs.VarLenCopy.PublicIO
