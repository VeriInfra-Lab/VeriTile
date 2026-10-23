import bench.repairs.tritonbench_g.matmul_dequant_int4.Proofs.Binding

namespace VeriTile.Bench.Repairs.MatmulDequantInt4.Direct
/-- Whole source API over arbitrary Real payloads, independently of all byte
codecs. Checked shapes/layouts, exact Tensor identities, complete independent
mathematical outputs, evolving source trials, and caller frames are included. -/
theorem matmul_dequant_int4_real : BoundReal := bound_real

#axiomsClean matmul_dequant_int4_real
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.MatmulDequantInt4.Direct
