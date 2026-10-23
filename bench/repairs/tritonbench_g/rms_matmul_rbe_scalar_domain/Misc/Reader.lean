import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Correct
import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Misc.NativeCorrect
import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Misc.TensorBridge

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
#axiomsClean Correct.kernel_real
#axiomsClean FPEquiv.kernel_native
#axiomsClean TensorBridge.whole_tensor
#axiomsClean TensorBridge.native_tensor
#axiomsClean TensorBridge.result_storage
#axiomsClean TensorBridge.result_count
#axiomsClean TensorBridge.head_offset
#axiomsClean TensorBridge.output_pointer_valid
#axiomsClean TensorBridge.x_load_safe
#axiomsClean TensorBridge.weight_load_safe
#axiomsClean TensorBridge.rms_load_safe
#axiomsClean source_reference
#axiomsClean native_reference
#axiomsClean qkv_complete
#axiomsClean PhysicalKernel.programTrace_complete
#axiomsClean PhysicalKernel.grid_complete
#axiomsClean PhysicalKernel.native_tensor
#axiomsClean PhysicalKernel.Ideal.whole_tensor
#specNonCircular Native.Primitives avoiding [execution,PhysicalKernel.grid,PhysicalKernel.nativeTarget]
#specNonCircular PhysicalKernel.nativeTarget avoiding [execution,PhysicalKernel.grid,nativeAccumulation]
#specNonCircular PhysicalKernel.Ideal.target avoiding [execution,PhysicalKernel.grid,accumulation]
#specNonCircular ScalarObservations avoiding [execution,PhysicalKernel.grid,PhysicalKernel.nativeTarget]

open Lean Elab Command in
run_cmd do
  let names := #[`PhysicalKernel.native_tensor,`PhysicalKernel.Ideal.whole_tensor,
    `PhysicalKernel.grid_complete,`PhysicalKernel.grid_results,`PhysicalKernel.programTrace_complete,
    `Native.Primitives,`Native.Primitives.mk,`Native.ZeroRealization,`Native.ZeroRealization.mk,
    `Native.output_storable,`PhysicalKernel.Access,`PhysicalKernel.Access.mk,
    `PhysicalKernel.ExistingQ,`PhysicalKernel.OrdinaryInputs,`PhysicalKernel.arguments,
    `ScalarObservations,`ScalarMaterialization,`ScalarMaterialization.mk,`PhysicalKernel.ExistingQ.mk,
    `PhysicalKernel.nativeTarget,`PhysicalKernel.Ideal.target,
    `PhysicalKernel.WrittenByte,`PhysicalKernel.x_load_safe,`PhysicalKernel.weight_load_safe,
    `PhysicalKernel.rms_load_safe,`PhysicalKernel.coverage,`PhysicalKernel.coordinates_unique,
    `PhysicalKernel.result_storage,`PhysicalKernel.result_count,`PhysicalKernel.head_offset]
  for short in names do
    let name ← resolveGlobalConstNoOverload (mkIdent short)
    let info ← liftCoreM <| getConstInfo name
    let type ← liftTermElabM <| Lean.Meta.ppExpr info.type
    logInfo m!"RMS_SCALAR_STATEMENT_JSON {(Json.mkObj [("name",Json.str name.toString),("type",Json.str type.pretty)]).compress}"
  let env ← getEnv
  logInfo m!"RMS_SCALAR_MODULES_JSON {(Json.arr (env.header.modules.map fun m => Json.str m.module.toString)).compress}"
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
