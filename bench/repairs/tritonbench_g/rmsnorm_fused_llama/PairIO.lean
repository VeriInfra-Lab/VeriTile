import bench.repairs.tritonbench_g.rmsnorm_fused_llama.NativeIO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.PairIO
open VeriTile Triton FP.Structural
noncomputable section

/-- Both complete row grids keep the runtime FP32 epsilon register, arbitrary
original symbolic inputs, source reductions and the FP16 output round trip. -/
def original (R : Native.Reductions) (X Y W : RegionName) (rows columns : Nat)
    (epsilon : Native.Word) :=
  ProgramIO.Implementation.original (NativeIO.original R X Y W rows columns epsilon) (fun _ => True)

def indexCommuted (R : Native.Reductions) (X Y W : RegionName) (rows columns : Nat)
    (epsilon : Native.Word) :=
  ProgramIO.Implementation.ofVariant
    (NativeIO.commuted (Native.algebra R) X Y W rows columns 16384 (.leaf "epsilon" 0 epsilon))
    (fun _ => True)

end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.PairIO
