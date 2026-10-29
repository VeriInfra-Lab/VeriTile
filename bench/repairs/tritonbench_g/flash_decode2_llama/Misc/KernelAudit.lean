import bench.repairs.tritonbench_g.flash_decode2_llama.Proofs.KernelEndpoints
import bench.repairs.tritonbench_g.flash_decode2_llama.Proofs.KernelSafety
import bench.repairs.tritonbench_g.flash_decode2_llama.Proofs.KernelOrder

namespace VeriTile.Bench.Repairs.FlashDecode2Llama
open VeriTile.Triton
#axiomsClean KernelModel.typed
#axiomsClean KernelModel.count_bound
#axiomsClean KernelTensor.direct_logical
#axiomsClean KernelTensor.mid_read
#axiomsClean KernelTensor.log_read
#axiomsClean KernelTensor.output_index
#axiomsClean KernelNative.recurrence
#axiomsClean KernelReal.value
#axiomsClean KernelEndpoints.real
#axiomsClean KernelEndpoints.native
#axiomsClean KernelSafety.pointers
#axiomsClean KernelSafety.offsets
#axiomsClean KernelSafety.length_add
#axiomsClean KernelSafety.maximal_q_once
#axiomsClean KernelModel.order
#specNonCircular KernelTensor.Arguments avoiding
  [KernelModel.run,FlashDecode2.Grid.source,FP.Structural.exec,KernelNative.target,KernelReal.target]
#specNonCircular KernelNative.Ops avoiding
  [KernelModel.run,FlashDecode2.Grid.source,FP.Structural.exec,KernelNative.target,KernelReal.target]
#specNonCircular KernelNative.schedule avoiding
  [KernelTensor.word,KernelTensor.lengths,KernelNative.entry,KernelNative.target]
end VeriTile.Bench.Repairs.FlashDecode2Llama
