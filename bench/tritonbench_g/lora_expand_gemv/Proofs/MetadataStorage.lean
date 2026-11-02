import bench.tritonbench_g.lora_expand_gemv.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.LoraExpandGemv.Pasted
open VeriTile.Bench.TritonBenchG.LoraExpandGemv.FPValue (OutputFormat)
noncomputable section
set_option maxRecDepth 40000

/-- Both supported signed metadata storage widths select the same integer
channel in the model. The checked caller separately bounds formed offsets. -/
theorem original_metadata_widths (input weight output : OutputFormat) :
    originalBody input weight output .i32 = originalBody input weight output .i64 := by rfl

theorem candidate_metadata_widths (input weight output : OutputFormat) :
    candidateBody input weight output .i32 = candidateBody input weight output .i64 := by rfl

#axiomsClean original_metadata_widths
#axiomsClean candidate_metadata_widths
end
end VeriTile.Bench.Repairs.LoraExpandGemv.Pasted
