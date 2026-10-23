import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeLaunch

/-! Finite native input channels and the already established independent
instruction reference. Raw FP32 words are unrestricted; the ISA, runtime
scale word and reduction layout remain explicit numerical model parameters. -/
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativePorts
open VeriTile Triton FP.Structural NativeInstructions
noncomputable section
set_option maxRecDepth 30000

def attentionWord (c : Host.Dimensions) (I : ISA Word) (w : KernelProfile.Profile) (scale : Word)
    (values : ChannelValues Word (producerInputs c)) (off : Nat) : Word :=
  (NativeStaging.lower (Host.producerReference (KernelProfile.algebra w scale) NativeStaging.dot c
    NativeRuntime.scaleTag (producerMemory c values) "attention" off)).eval I

def outputWord (c : Host.Dimensions) (I : ISA Word) (w : KernelProfile.Profile) (scale : Word)
    (values : ChannelValues Word (consumerInputs c)) (i : Fin (Host.outputSize c)) : Word :=
  (NativeStaging.lower (FPReturned.value (KernelProfile.algebra w scale) NativeStaging.dot c.consumer
    NativeRuntime.scaleTag (consumerMemory c values) i.divNat.divNat i.divNat.modNat i.modNat)).eval I

#specNonCircular attentionWord avoiding [FPStages.execute,FPGrid.runJobs,FPGrid.invoke,FP.Structural.exec,
  NativeLaunch.producer,runFillSchedule]
#specNonCircular outputWord avoiding [FPStages.execute,FPGrid.runJobs,FPGrid.invoke,FP.Structural.exec,
  NativeLaunch.consumer,runFillSchedule]
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativePorts
