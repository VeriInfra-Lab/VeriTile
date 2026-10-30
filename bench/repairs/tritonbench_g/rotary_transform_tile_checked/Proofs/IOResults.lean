import bench.repairs.tritonbench_g.rotary_transform_tile_checked.Proofs.ScheduleSafety
import bench.repairs.tritonbench_g.rotary_transform_tile_checked.Proofs.KernelValues

namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked.IOResults
open VeriTile.Bench.Repairs.RotaryTransform
open VeriTile.Triton FP.Structural PublicIO
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

theorem real (c : Config) (valid : c.Valid) (cu64 off64 : Bool) :
    (io realOps id c cu64 off64).ImplementsOn (domain c)
      (expected c (realFormula c)) := by
  apply ReadProgramIO.ImplementsOn.intro
  · exact schedule_safe realOps id c valid cu64 off64
  · intro vals legal bounds bound initial hin
    exact KernelValues.real_result c valid cu64 off64 vals legal initial hin

theorem native (c : Config) (valid : c.Valid) (p : Native.Instructions) (cu64 off64 : Bool) :
    (io (Native.ops p) (fun node => Native.atom node.value) c cu64 off64).ImplementsOn (domain c)
      (expected c (fun values b r h d => Native.atom (nativeFormula c p (Native.project values) b r h d))) := by
  apply ReadProgramIO.ImplementsOn.intro
  · exact schedule_safe (Native.ops p) (fun node => Native.atom node.value) c valid cu64 off64
  · intro vals legal bounds bound initial hin
    exact KernelValues.native_result c valid p cu64 off64 vals legal initial hin

#axiomsClean real
#axiomsClean native
end
end VeriTile.Bench.Repairs.RotaryTransformTileChecked.IOResults
