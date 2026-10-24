import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Tensor
import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.InputSafety

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
#axiomsClean complete
#axiomsClean output_view
#axiomsClean output_real
#axiomsClean PhysicalKernel.grid_success
#axiomsClean PhysicalKernel.whole_native
#axiomsClean PhysicalKernel.whole_real
#axiomsClean PhysicalKernel.whole_tensor
#axiomsClean PhysicalKernel.whole_bytes
#axiomsClean PhysicalKernel.whole_real_tensor
#axiomsClean PhysicalKernel.whole_input_frame
#axiomsClean PhysicalKernel.output_word
#axiomsClean PhysicalKernel.native_scalar_argument
#specNonCircular PhysicalKernel.ABI avoiding
  [execution,output,accumulation,PhysicalKernel.grid,rowValue,Spec.output,Host.Allocation,Host.invoke]
#specNonCircular PhysicalKernel.nativeTarget avoiding [execution,output,accumulation,PhysicalKernel.grid]
#specNonCircular PhysicalKernel.realTarget avoiding [execution,output,accumulation,PhysicalKernel.grid,rowValue]

open Lean Elab Command in
run_cmd do
  let names := #[`PhysicalKernel.whole_native,`PhysicalKernel.whole_real,
    `PhysicalKernel.whole_tensor,`PhysicalKernel.whole_bytes,`PhysicalKernel.whole_real_tensor,
    `PhysicalKernel.whole_input_frame,`PhysicalKernel.ABI,`PhysicalKernel.NativePrimitives,
    `PhysicalKernel.FiniteInputs,`PhysicalKernel.Access,`Physical.WordCodec,
    `Spec.output,`rowValue,`PhysicalKernel.query_load,`PhysicalKernel.key_value_load,
    `PhysicalKernel.integer_load,`PhysicalKernel.csr_access,`PhysicalKernel.data_i32,
    `PhysicalKernel.query_i32,`PhysicalKernel.output_i32]
  for short in names do
    let name ← resolveGlobalConstNoOverload (mkIdent short)
    let info ← liftCoreM <| getConstInfo name
    let type ← liftTermElabM <| Lean.Meta.ppExpr info.type
    logInfo m!"BLOCK_SPARSE_KERNEL_STATEMENT_JSON {(Json.mkObj [("name",Json.str name.toString),("type",Json.str type.pretty)]).compress}"
  let env ← getEnv
  logInfo m!"BLOCK_SPARSE_KERNEL_MODULES_JSON {(Json.arr (env.header.modules.map fun m => Json.str m.module.toString)).compress}"
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
