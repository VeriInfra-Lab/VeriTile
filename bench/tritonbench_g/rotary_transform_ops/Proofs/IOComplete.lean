import bench.tritonbench_g.rotary_transform_ops.Proofs.IOSafety
import bench.tritonbench_g.rotary_transform_ops.Proofs.IOResult
import bench.tritonbench_g.rotary_transform_ops.Proofs.KernelCompatibility

namespace VeriTile.Bench.Repairs.RotaryTransformOps.IOComplete
open VeriTile.Triton FP.Structural Interface
noncomputable section

theorem real_implements (c : Config) (valid : c.Valid) (cu64 off64 : Bool) :
    (io realOps c cu64 off64 id).ImplementsOn (Ready c) (realReference c) := by
  apply ReadProgramIO.ImplementsOn.intro
  · exact IOSafety.whole realOps c valid cu64 off64 id
  · intro vals ready _ _ u hin
    exact IOResult.real_result c valid cu64 off64 vals ready u hin

theorem native_implements (c : Config) (valid : c.Valid) (p : Native.Instructions)
    (cu64 off64 : Bool) :
    (io (Native.ops p) c cu64 off64 observeNative).ImplementsOn (Ready c) (nativeReference c p) := by
  apply ReadProgramIO.ImplementsOn.intro
  · exact IOSafety.whole (Native.ops p) c valid cu64 off64 observeNative
  · intro vals ready _ _ u hin
    exact IOResult.native_result c valid p cu64 off64 vals ready u hin

end
end VeriTile.Bench.Repairs.RotaryTransformOps.IOComplete
