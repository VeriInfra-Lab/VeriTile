import bench.repairs.tritonbench_g.attention_fwd_triton2.Whole001.Correct
import bench.repairs.tritonbench_g.attention_fwd_triton2.Whole001.FPEquiv
namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.Whole001
open VeriTile Triton
#axiomsClean whole_returned_real
#axiomsClean whole_returned_native
#axiomsClean whole_local_numerical_transport
#specNonCircular RealReturned.target avoiding [FP.Structural.exec, Whole.invoke, Physical.invoke, Whole.answer]
#specNonCircular NativeReturned.InputWords avoiding [FP.Structural.exec, Whole.invoke, Physical.invoke, Whole.answer]
#specNonCircular Native.LocalTransport avoiding [FP.Structural.exec, Whole.invoke, Physical.invoke, Native.reference]
end VeriTile.Bench.Repairs.AttentionFwdTriton2.Whole001
