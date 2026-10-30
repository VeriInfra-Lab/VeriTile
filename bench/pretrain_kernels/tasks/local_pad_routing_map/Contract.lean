import bench.pretrain_kernels.tasks.local_pad_routing_map.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_pad_routing_map
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Contract
open scoped BigOperators Classical

def Allowed (_xs : IO.Inputs) : Prop :=
  True

def output : IO.Inputs → IO.OutputIndex → ℝ :=
  fun xs (i0,i1) =>
    let input0 := xs.input0
    let vto_copy : Fin 16 → Fin 128 → Int := fun i0 i1 => (if input0 i1 i0 then 1 else 0)
    let vsum_1 : Fin 16 → Int := fun i0 => (∑ lane : Fin 128, vto_copy i0 lane)
    let vneg : Fin 16 → Int := fun i0 => (- (vsum_1 i0))
    let vremainder : Fin 16 → Int := fun i0 => (vneg i0 % 16)
    let veq : Fin 16 → Fin 128 → Bool := fun i0 i1 => decide (vto_copy i0 i1 = 0)
    let vto_copy_1 : Fin 16 → Fin 128 → Int := fun i0 i1 => (if veq i0 i1 then 1 else 0)
    let vcumsum : Fin 16 → Fin 128 → Int := fun i0 i1 => (∑ lane : Fin 128, if lane.val ≤ i1.val then vto_copy_1 i0 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩ else 0)
    let vle : Fin 16 → Fin 128 → Bool := fun i0 i1 => decide (vcumsum i0 i1 ≤ vremainder i0)
    let vbitwise_and : Fin 16 → Fin 128 → Bool := fun i0 i1 => (veq i0 i1 && vle i0 i1)
    let vwhere : Fin 16 → Fin 128 → Int := fun i0 i1 => (if vbitwise_and i0 i1 then 1 else vto_copy i0 i1)
    let vto_copy_2 : Fin 128 → Fin 16 → ℝ := fun i0 i1 => (vwhere i1 i0 : ℝ)
    vto_copy_2 i0 i1

end Contract
end
end VeriTile.Bench.PretrainKernels.local_pad_routing_map
