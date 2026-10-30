import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.RawCorrect
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.RawNative
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Misc.NativeIOResults
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.Checked
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.Representation

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast

#axiomsClean RawCorrect.kernel_correct
#axiomsClean Interface.Result.real
#axiomsClean RawNative.kernel_equiv
#axiomsClean Interface.Result.native
#axiomsClean Execution.sourceComplete
#axiomsClean Execution.nativeComplete
#axiomsClean Grid.whole
#axiomsClean ByteGrid.complete
#axiomsClean Real.source_formula
#axiomsClean NativeReference.logical_eq
#axiomsClean Words.signed_byte_exact
#axiomsClean Checked.indices
#specNonCircular Real.target avoiding [Execution.source,RealGrid.run,Grid.grid,Execution.rowAcc,Execution.rowMoment]
#specNonCircular ByteGrid.target avoiding [Execution.source,ByteGrid.run,Grid.grid,Execution.nativeExecution,
  Execution.logicalNative,Execution.rowAcc,Execution.rowMoment]

end VeriTile.Bench.Repairs.LlamaFFScalarCast
