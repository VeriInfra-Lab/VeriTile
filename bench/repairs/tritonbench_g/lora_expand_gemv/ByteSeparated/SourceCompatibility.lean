import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Kernels
import bench.repairs.tritonbench_g.lora_expand_gemv.Kernels

/-! Preserve the previous generic compatibility interface. Its selected FP32 /
FP16 / FP32 case uses the complete raw source; historical other profiles keep
their previous definitions. Raw Kernels does not import the proof layer. -/
namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated
open VeriTile.Triton FP.Structural
noncomputable section

/-- Compatibility for the existing source interface. The selected profile is
explicit above; the other historical profiles retain their existing source. -/
def source (profile : Branches.InputProfile) (out : FPValue.OutputFormat)
    (p : Branches.Parameters) : ComputeKernel :=
  match profile,out with
  | .singleHalf,.single => fullSource p
  | _,_ => Branches.source profile out p

end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated
