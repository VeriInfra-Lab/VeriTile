import bench.repairs.tritonbench_g.streamk_matmul.Whole001.FPEquiv
import bench.repairs.tritonbench_g.streamk_matmul.Whole001.Correct
namespace VeriTile.Bench.Repairs.StreamK.Whole001
#print whole_returned_real
#print whole_returned_native
#axiomsClean whole_returned_real
#axiomsClean whole_returned_native
#axiomsClean Source.reducer_load_value
#axiomsClean Protocol.iteration_has_unique_chunk
#axiomsClean Schedule.reducer_ctas_disjoint
#specNonCircular Reference.matrixProduct avoiding [Source.call,Schedule.call,Memory.run,Source.chunkValue,Returned.value]
#specNonCircular Returned.value avoiding [Source.call,Schedule.call,Memory.run,Memory.step]
#specNonCircular Native.Numerical avoiding [Source.call,Schedule.call,Memory.run,Returned.value,Source.chunkValue]
#specNonCircular Schedule.CTAInterleaving avoiding [Source.call,Schedule.call,Memory.run,Memory.step,Returned.value]
#specNonCircular ABI.Valid avoiding [Source.call,Schedule.call,Memory.run,Returned.value,Native.Numerical]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.StreamK.Whole001
