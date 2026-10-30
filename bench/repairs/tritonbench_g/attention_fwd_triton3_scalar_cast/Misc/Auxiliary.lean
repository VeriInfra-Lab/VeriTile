import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.RealTransport
import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.RawFPEquiv

/-! Preserved full operational endpoints: canonical flattened ordinary-grid
Real execution, all padded outputs and frame; raw-byte native execution,
primitive-law observation, reachability/provenance and exact publication frame.
These use the same established execution foundation as the public IO judgments.
-/
namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Auxiliary
export RawKernel (whole_real correct fp_equiv)
export RealInterface (canonical_flat_io)
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Auxiliary
