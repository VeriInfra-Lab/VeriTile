import bench.repairs.tritonbench_g.chunk_gla_fwd.Misc.Compatibility
import bench.repairs.tritonbench_g.chunk_gla_fwd.Correct
import bench.repairs.tritonbench_g.chunk_gla_fwd.FPEquiv
import bench.repairs.tritonbench_g.chunk_gla_fwd.Misc.TensorBridge
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativePhysical

/-! Reader-level checks for complete source IO, actual trace safety, finite
mathematical/native targets, physical observations and retained Tensor scope.
Proof helpers do not import these reader entries. -/
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.InterfaceChecks
open VeriTile Triton FP.Structural NativeInstructions
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
open NativePorts (Word Expr)
noncomputable section
set_option maxRecDepth 30000

example (c : Host.Dimensions) (scale : ℝ) :
    (RealIO.interSource c scale).kernel=
      (SourceBridge.interNew (ProducerGeometry.layout c.producer) ProducerGeometry.regions scale).eraseDType := rfl
example (c : Host.Dimensions) (scale : ℝ) :
    (RealIO.intraSource c scale).kernel=
      (SourceBridge.intraNew (IntraGrid.layout c.producer c.intraBK) IntraGrid.regions scale).eraseDType := rfl
example (c : Host.Dimensions) (scale : ℝ) :
    (RealIO.splitSource c scale).kernel=
      (SourceBridge.splitNew (SplitGrid.layout c.split) SplitGrid.regions scale).eraseDType := rfl
example (c : Host.Dimensions) :
    (RealIO.mergeSource c).kernel=
      (SourceBridge.mergeNew (MergeGrid.layout c.split) "scratch" "attention").eraseDType := rfl
example (c : Host.Dimensions) (scale : ℝ) :
    (RealIO.outputSource c scale).kernel=
      (SourceBridge.outputNew (OutputGrid.layout c.consumer) OutputGrid.regions scale).eraseDType := rfl

/-- Native stages use the same five quotations before Real dtype erasure. -/
example (c : Host.Dimensions) (scale : ℝ) :
    (FPStages.Stage.inter c.producer scale).kernel=
      SourceBridge.interNew (ProducerGeometry.layout c.producer) ProducerGeometry.regions scale := rfl
example (c : Host.Dimensions) (scale : ℝ) :
    (FPStages.Stage.intra c.producer c.intraBK scale).kernel=
      SourceBridge.intraNew (IntraGrid.layout c.producer c.intraBK) IntraGrid.regions scale := rfl
example (c : Host.Dimensions) (scale : ℝ) :
    (FPStages.Stage.split c.split scale).kernel=
      SourceBridge.splitNew (SplitGrid.layout c.split) SplitGrid.regions scale := rfl
example (c : Host.Dimensions) :
    (FPStages.Stage.merge c.split).kernel=
      SourceBridge.mergeNew (MergeGrid.layout c.split) "scratch" "attention" := rfl
example (c : Host.Dimensions) (scale : ℝ) :
    (FPStages.Stage.output c.consumer scale).kernel=
      SourceBridge.outputNew (OutputGrid.layout c.consumer) OutputGrid.regions scale := rfl

/-- Every native invocation selects the original stage kernel, and its
complete list of program IDs is the established ordinary grid enumeration. -/
example {α : Type} [Inhabited α] (M : Algebra α) (D : FPOutput.DotPrimitive α)
    (stage : FPStages.Stage) (idx : GridIndex stage.grid) :
    (NativeLaunch.job M D stage idx).kernel=stage.kernel := rfl
example {α : Type} [Inhabited α] (M : Algebra α) (D : FPOutput.DotPrimitive α)
    (stage : FPStages.Stage) :
    NativeLaunch.jobs M D stage=
      (FPGrid.jobs stage.grid).map (fun idx=>FillStage.source (NativeLaunch.job M D stage idx)) := rfl

example (c : Host.Dimensions) (I : ISA Word) (w : KernelProfile.Profile) (scale : Word)
    (values : (NativeIO.producer c I w scale).Inputs) :
    (NativeIO.producer c I w scale).stages values=
      NativeLaunch.producer (KernelProfile.algebra w scale) NativeStaging.dot c NativeRuntime.scaleTag := rfl
example (c : Host.Dimensions) (I : ISA Word) (w : KernelProfile.Profile) (scale : Word)
    (values : (NativeIO.consumer c I w scale).Inputs) :
    (NativeIO.consumer c I w scale).stages values=
      NativeLaunch.consumer (KernelProfile.algebra w scale) NativeStaging.dot c NativeRuntime.scaleTag := rfl

/-- Both output tags describe one complete attention tensor: no output cell
is omitted or claimed by both channels. The partition contains no values. -/
theorem attention_partition (c : Host.Dimensions) (I : ISA Word) (w : KernelProfile.Profile) (scale : Word)
    (values : (NativeIO.producer c I w scale).Inputs) (i : TileIndex [Host.attentionSize c]) :
    ((NativeIO.producer c I w scale).outputMask (0 : Fin 2) values i ∨
      (NativeIO.producer c I w scale).outputMask (1 : Fin 2) values i) ∧
    ¬((NativeIO.producer c I w scale).outputMask (0 : Fin 2) values i ∧
      (NativeIO.producer c I w scale).outputMask (1 : Fin 2) values i) := by
  change (NativeTags.producerExplicit c i.1.val=Bool.false ∨ NativeTags.producerExplicit c i.1.val=Bool.true) ∧
    ¬(NativeTags.producerExplicit c i.1.val=Bool.false ∧ NativeTags.producerExplicit c i.1.val=Bool.true)
  cases NativeTags.producerExplicit c i.1.val <;> simp

#axiomsClean Correct.producer_kernel_correctness
#axiomsClean Correct.consumer_kernel_correctness
#axiomsClean FPEquiv.producer_native_observation
#axiomsClean FPEquiv.consumer_native_observation
#axiomsClean RealStages.producer_trace
#axiomsClean RealStages.consumer_trace
#axiomsClean NativeSafety.source_safe
#axiomsClean NativeAccess.producer
#axiomsClean NativeAccess.consumer
#axiomsClean NativeWhole.producer
#axiomsClean NativeWhole.consumer
#axiomsClean NativeTags.producer_dtype
#axiomsClean NativeTags.consumer_dtype
#axiomsClean attention_partition
#axiomsClean NativePhysical.producer_binding
#axiomsClean NativePhysical.consumer_binding
#axiomsClean NativePhysical.producer_target
#axiomsClean NativePhysical.consumer_target

-- Exact historical raw and physical/Tensor declarations remain available.
#axiomsClean Correct.producer_correct
#axiomsClean Correct.consumer_correct
#axiomsClean Correct.kernel_raw_correct
#axiomsClean FPEquiv.producer_returned_fp32
#axiomsClean FPEquiv.consumer_returned_fp32
#axiomsClean FPEquiv.local_numerical_transport
#axiomsClean FPEquiv.kernel_raw_equiv
#axiomsClean TensorBridge.kernel_correct
#axiomsClean TensorBridge.kernel_equiv
#axiomsClean TensorBridge.wrapper_correct
#axiomsClean TensorBridge.wrapper_fp32
#axiomsClean TensorBridge.successful_values
#axiomsClean TensorBridge.rejected

#specNonCircular RealIO.attention avoiding [VeriTile.Triton.exec,Producer.direct,Producer.large,
  RealIO.producerStages,RealIO.interSource,RealIO.intraSource,RealIO.splitSource,RealIO.mergeSource]
#specNonCircular RealIO.output avoiding [VeriTile.Triton.exec,OutputGrid.executes,FP.Structural.exec,RealIO.outputSource]
#specNonCircular NativeIO.attention avoiding [FPStages.execute,FPGrid.runJobs,FPGrid.invoke,
  FP.Structural.exec,NativeLaunch.producer,NativeLaunch.job,FPStages.Stage.kernel,runFillSchedule]
#specNonCircular NativeIO.output avoiding [FPStages.execute,FPGrid.runJobs,FPGrid.invoke,
  FP.Structural.exec,NativeLaunch.consumer,NativeLaunch.job,FPStages.Stage.kernel,runFillSchedule]
#specNonCircular NativeTags.producerExplicit avoiding [State,FPStages.program,FPStages.reference,
  FP.Structural.run,FP.Structural.exec,NativeLaunch.producer]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.InterfaceChecks
