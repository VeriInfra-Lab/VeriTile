import bench.repairs.tritonbench_g.lightning_attention.Proofs.Complete
import bench.repairs.tritonbench_g.lightning_attention.Proofs.NativeEndpoint
import bench.repairs.tritonbench_g.lightning_attention.Proofs.NativeMask
import bench.repairs.tritonbench_g.lightning_attention.Proofs.SourceNormalization
namespace VeriTile.Bench.Repairs.LightningAttention
#axiomsClean Host.forward_complete
#axiomsClean Host.backward_complete
#axiomsClean NativeHost.forward_complete
#axiomsClean NativeHost.backward_complete
#axiomsClean Host.returned_loss_derivatives
#axiomsClean Real.whole_correct
#axiomsClean Real.forward_correct
#axiomsClean Real.backward_correct
#axiomsClean Native.forward_execution
#axiomsClean Native.backward_execution
#axiomsClean Native.whole_execution
#axiomsClean NativeMask.lane_exact
#axiomsClean NativeMask.difference_exact
#axiomsClean NativeMask.lower_mask
#axiomsClean NativeMask.upper_mask
#axiomsClean SourceNormalization.reverse_fold
#axiomsClean SourceNormalization.reverse_skip
#axiomsClean SourceNormalization.reverse_row
#axiomsClean Host.launch_axes
#axiomsClean Host.forward_pointer_safe
#axiomsClean Host.backward_pointer_safe
#specNonCircular Target.loss avoiding [Target.dq,Target.dk,Target.dv]
#specNonCircular Host.ReturnedVJP avoiding [Host.publicBackward,modelBackward,Real.backwardMemory,Target.dq,Target.dk,Target.dv]
#specNonCircular Native.LocalConformance avoiding [Native.execute,Native.reference,Native.dotGraph]
#specNonCircular Native.NegationConformance avoiding [Native.execute,Native.reference,Native.dotGraph]
end VeriTile.Bench.Repairs.LightningAttention
