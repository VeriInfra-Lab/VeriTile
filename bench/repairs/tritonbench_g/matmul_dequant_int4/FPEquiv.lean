import bench.repairs.tritonbench_g.matmul_dequant_int4.Proofs.Binding

namespace VeriTile.Bench.Repairs.MatmulDequantInt4.Direct
/-- Checked native API begins at original absolute-address byte memory and
returns exact Tensor allocations, every output byte, and caller/input frames. -/
theorem matmul_dequant_int4_native (I : Native.Instructions) : BoundNative I I := bound_native I

/-- The only numerical assumptions are the 26 scalar family containers and one
fixed K16 ISA component family in Native.Correspondence. The explicit installed
library/compiler selections are consumed by the modeled calls. -/
theorem matmul_dequant_int4_if_local (actual reference : Native.Instructions)
    (numeric : Native.Correspondence actual reference) : BoundNative actual reference :=
  bound_if_local actual reference numeric

#axiomsClean matmul_dequant_int4_native
#axiomsClean matmul_dequant_int4_if_local
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.MatmulDequantInt4.Direct
