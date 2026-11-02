import bench.tritonbench_g.sgmv_expand_slice.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.Pasted
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 40000

/-- All three native metadata channels are explicitly widened by the source.
The eight storage combinations therefore produce the same complete kernel,
for every remaining runtime argument and constexpr choice. -/
theorem original_metadata_widths (input_dtype : FloatDType)
    (weight_format : LowFormat) (output_format : ScalarFormat)
    (starts lengths adapters : MetadataWidth) :
    originalBody input_dtype weight_format output_format starts lengths adapters =
      originalBody input_dtype weight_format output_format .w32 .w32 .w32 := by
  cases starts <;> cases lengths <;> cases adapters <;> rfl

theorem candidate_metadata_widths (input_dtype : FloatDType)
    (weight_format : LowFormat) (output_format : ScalarFormat)
    (starts lengths adapters : MetadataWidth) :
    candidateBody input_dtype weight_format output_format starts lengths adapters =
      candidateBody input_dtype weight_format output_format .w32 .w32 .w32 := by
  cases starts <;> cases lengths <;> cases adapters <;> rfl

#axiomsClean original_metadata_widths
#axiomsClean candidate_metadata_widths
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.Pasted
