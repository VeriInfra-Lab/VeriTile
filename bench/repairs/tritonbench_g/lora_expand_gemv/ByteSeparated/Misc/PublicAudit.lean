import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.TensorBridge
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Correct
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.NativeCorrect
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.Guard
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.Access

namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated
open VeriTile.Triton
#axiomsClean Contract.python_checks_iff
#axiomsClean Contract.python_separate_iff
#axiomsClean valid
#axiomsClean FP32Binding.logical_read
#axiomsClean TensorBridge.checked_real
#axiomsClean TensorBridge.checked_fp
#axiomsClean TensorBridge.checked_caller_bytes
#axiomsClean caller_frame
#axiomsClean sentinel_frame
#axiomsClean output_padding_frame
#axiomsClean failure_frame
#axiomsClean empty
#axiomsClean signed_formed
#axiomsClean addresses
#axiomsClean metadata_span
#specNonCircular Real.target avoiding [execute,Branches.source,FP.Structural.exec,value]
#specNonCircular TensorBridge.nativeReference avoiding [execution,finalMemory,sourceStores,execute,Branches.source,FP.Structural.exec]
#specNonCircular TensorBridge.CompilerBinding avoiding [execution,finalMemory,sourceStores,execute,FP.Structural.exec]
#auditModuleAxioms
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated
