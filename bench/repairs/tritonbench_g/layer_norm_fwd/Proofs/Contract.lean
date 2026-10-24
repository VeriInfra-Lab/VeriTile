import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.Admission
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperStorageTypes
namespace VeriTile.Bench.Repairs.LayerNormFwd.Admission
open Wrapper WrapperNative
 theorem defaultInputs (c : Call) (h : InputDomain c) : DefaultInputs (request c) := by
  obtain ⟨hx,hw,hb,hw1,hb1,hr,hx1,hrs,_⟩ := h
  intro role ha value hv
  cases role with
  | x =>
    have he : c.x.value=value := Option.some.inj hv
    exact he ▸ hx.2.2.2.1
  | weight =>
    have he : c.weight.value=value := Option.some.inj hv
    exact he ▸ hw.2.2.2.1
  | residual => exact optional_dtype _ _ _ _ hr value hv
  | x1 => exact optional_dtype _ _ _ _ hx1 value hv
  | bias => exact optional_dtype _ _ _ _ hb value hv
  | weight1 => exact optional_dtype _ _ _ _ hw1 value hv
  | bias1 => exact optional_dtype _ _ _ _ hb1 value hv
  | rowscale | rowscale1 => exact optional_dtype _ _ _ _ hrs value hv
  | y | y1 | residualOut | rstd | mean | seeds | seeds1 | mask | mask1 =>
    simp [sourceInputRole] at ha

#axiomsClean defaultInputs
end VeriTile.Bench.Repairs.LayerNormFwd.Admission
