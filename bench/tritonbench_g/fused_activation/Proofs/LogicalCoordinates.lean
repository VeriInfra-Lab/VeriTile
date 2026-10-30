import bench.tritonbench_g.fused_activation.IO
import bench.tritonbench_g.fused_activation.Proofs.RealInterface

namespace VeriTile.Bench.Repairs.FusedActivation.RealInterface
open VeriTile.Triton RealIO
noncomputable section

theorem logical_iff (x b input : RegionName) (W N B : Nat) (a : Bool) :
    (logicalActivationIO x b input W N B a).Implements
      (fun xs j => formula a (xs.x j) (xs.bias j) (xs.input j)) ↔
    (activationIO x b input W N B a).RegionalImplements
      (fun _ _ xs _ j => formula a (xs (0 : Fin 3) j)
        (xs (1 : Fin 3) j) (xs (2 : Fin 3) j)) := Iff.rfl

theorem logical_correct (x b input : RegionName) (W N B : Nat) (a : Bool) :
    (logicalActivationIO x b input W N B a).Implements
      (fun xs j => formula a (xs.x j) (xs.bias j) (xs.input j)) :=
  (logical_iff x b input W N B a).mpr (regional_correct x b input W N B a)
end
end VeriTile.Bench.Repairs.FusedActivation.RealInterface
