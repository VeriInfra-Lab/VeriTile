import bench.repairs.tritonbench_g.flash_attn.Contract
import bench.repairs.tritonbench_g.flash_attn.Proofs.CheckedFPEquiv
import bench.repairs.tritonbench_g.flash_attn.Proofs.SourceFPEquiv

/-! Public actual-byte endpoint: `Checked.whole_native` binds actual Tensor
pointers and raw scalar ABI to the selected source graph, returns the actual
allocated O, and proves O/L word readback and full physical byte frames.
The native numerical relation is separate. Historical relative-byte component
APIs remain available through `Proofs/SourceFPEquiv`. -/
