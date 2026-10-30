import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.RawCorrect
import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.RawNative
import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.PublicProducer

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Public.TensorBridge
open VeriTile.Triton FP.Structural Direct KernelExecution
noncomputable section
local instance : Inhabited Word32 := ⟨FP.Value.zero .fp32⟩

/-- Public standalone ideal theorem includes physical Tensor admission,
actual prepared-backing source execution, independent dense target/frame. -/
theorem correct (tree : KernelScalarProfile.Tree) (rest : Algebra ℝ) (base : Nat → Nat)
    (request : Tensor.Request) (callerDevice : Nat) (caller : Memory ℝ ℝ) (live : Nat → Prop)
    (alloc : GemmAllocation request.original caller live) (admit : Admission.Gemm base request caller live alloc)
    (template : State ℝ) (trials : List (Host.Invocation request.original.shape)) (chosen : Host.Invocation request.original.shape) :
    Binding id base request caller live alloc template ∧
    Simulation id request caller live alloc template (KernelScalarProfile.mathematical tree rest) mathematicalIntegers
      KernelIntegerIdeal.operations.toOps (·+·) (·+·)
      (Grid.zeroCell (request.original.buffers alloc.output.info.storage).outPrecision 0) chosen ∧
    Post id request caller live alloc callerDevice
      (fun cell => Grid.zeroCell (request.original.buffers alloc.output.info.storage).outPrecision
        (target request.original.shape (Gemm.realInput request.original.shape (Tensor.originalInput id request.original caller)) cell.1 cell.2))
      (Tensor.realRun tree rest request callerDevice caller live alloc template trials chosen) :=
  ⟨binding id base request caller live alloc admit template,
   simulation id base request caller live alloc admit template _ _ _ _ _ rfl rfl _ chosen,
   Tensor.real_complete tree rest request callerDevice caller live alloc template trials chosen⟩

/-- Exact native equivalence to the source-ordered word graph, with original
Tensor liveness/layout, output separation and actual prepared source backing.
This theorem does not identify native arithmetic with the Real theorem. -/
theorem fpEquiv (f : KernelNativeModel.FloatingInstructions) (tree : KernelScalarProfile.Tree)
    (rest : Algebra Word32) (base : Nat → Nat) (request : Tensor.Request) (callerDevice : Nat)
    (caller : Memory Word32 Word16) (live : Nat → Prop) (alloc : GemmAllocation request.original caller live)
    (admit : Admission.Gemm base request caller live alloc) (template : State Word32)
    (trials : List (Host.Invocation request.original.shape)) (chosen : Host.Invocation request.original.shape) :
    Binding KernelScalarProfile.pack16 base request caller live alloc template ∧
    Simulation KernelScalarProfile.pack16 request caller live alloc template (KernelScalarProfile.native f tree rest)
      signed32 (KernelNativeModel.operations f).toOps
      (fun x y => KernelScalarProfile.pack16 (f.atomic16 (KernelScalarProfile.unpack16 x) (KernelScalarProfile.unpack16 y)))
      f.atomic32 (Grid.zeroCell (request.original.buffers alloc.output.info.storage).outPrecision (FP.Value.zero .fp32)) chosen ∧
    Post KernelScalarProfile.pack16 request caller live alloc callerDevice
      (fun cell => Grid.nativeWord f tree rest chosen.plans (request.original.buffers alloc.output.info.storage).aScalePrecision
        (request.original.buffers alloc.output.info.storage).outPrecision request.original.shape chosen.config
        (request.original.buffers alloc.output.info.storage).bColumnMajor (Tensor.originalInput KernelScalarProfile.pack16 request.original caller)
        chosen.schedule cell)
      (Tensor.nativeRun f tree rest request callerDevice caller live alloc template trials chosen) :=
  ⟨binding KernelScalarProfile.pack16 base request caller live alloc admit template,
   simulation KernelScalarProfile.pack16 base request caller live alloc admit template _ _ _ _ _ rfl rfl _ chosen,
   Tensor.native_complete f tree rest request callerDevice caller live alloc template trials chosen⟩

theorem producerCorrect (base : Nat → Nat) (r : RowRequest) (laws : TensorLaws base r.input)
    (caller : Memory ℝ ℝ) (live : Nat → Prop) (inputLive : live r.input.info.storage) (alloc : RowAllocation r caller live)
    (dataLaws : TensorLaws base alloc.data) (scaleLaws : TensorLaws base alloc.scales)
    (template : State ℝ) (width : Nat) (positive : 0<width) (covers : r.K≤width)
    (selection : Producer.Selection (ProducerTensor.geometry r width positive)) (rest : Algebra ℝ)
    (trials : List (ProducerLifecycle.Invocation ℝ (ProducerTensor.geometry r width positive)))
    (order : Producer.Schedule (ProducerTensor.geometry r width positive)) :
    InputPointerBinding base r.input ∧ InputPointerBinding base alloc.data ∧ InputPointerBinding base alloc.scales ∧
    ∃ before final, ProducerLifecycle.trials (ProducerTensor.geometry r width positive) trials
      (ProducerTensor.bind id r alloc.memory template)=some before ∧
      KernelProducerGrid.run (KernelScalarProfile.mathematical selection.tree rest) realInteger8
        (ProducerTensor.geometry r width positive) order.order before=some final ∧
      RowPost r caller live alloc (ProducerTensor.memory id r alloc.data alloc.scales alloc.memory final)
        (fun g k => KernelProducerTargets.mathematicalByte (ProducerTensor.originalValues id r caller g) r.K k)
        (fun g => KernelProducerTargets.mathematicalScale (ProducerTensor.originalValues id r caller g) r.K) :=
  producerReal base r laws caller live inputLive alloc dataLaws scaleLaws template width positive covers selection rest trials order

theorem producerFPEquiv (f : KernelNativeModel.FloatingInstructions) (base : Nat → Nat) (r : RowRequest)
    (laws : TensorLaws base r.input) (caller : Memory Word32 Word16) (live : Nat → Prop)
    (inputLive : live r.input.info.storage) (alloc : RowAllocation r caller live)
    (dataLaws : TensorLaws base alloc.data) (scaleLaws : TensorLaws base alloc.scales)
    (template : State Word32) (width : Nat) (positive : 0<width) (covers : r.K≤width)
    (selection : Producer.Selection (ProducerTensor.geometry r width positive)) (rest : Algebra Word32)
    (trials : List (ProducerLifecycle.Invocation Word32 (ProducerTensor.geometry r width positive)))
    (order : Producer.Schedule (ProducerTensor.geometry r width positive)) :
    InputPointerBinding base r.input ∧ InputPointerBinding base alloc.data ∧ InputPointerBinding base alloc.scales ∧
    ∃ before final, ProducerLifecycle.trials (ProducerTensor.geometry r width positive) trials
      (ProducerTensor.bind KernelScalarProfile.pack16 r alloc.memory template)=some before ∧
      KernelProducerGrid.run (KernelScalarProfile.native f selection.tree rest) KernelNativeModel.integer8
        (ProducerTensor.geometry r width positive) order.order before=some final ∧
      RowPost r caller live alloc
        (ProducerTensor.memory KernelScalarProfile.unpack16 r alloc.data alloc.scales alloc.memory final)
        (fun g k => KernelProducerTargets.nativeByte f (ProducerTensor.geometry r width positive) selection
          (ProducerTensor.originalValues KernelScalarProfile.pack16 r caller g) k)
        (fun g => KernelProducerTargets.nativeStoredScale f (ProducerTensor.geometry r width positive) selection
          (ProducerTensor.originalValues KernelScalarProfile.pack16 r caller g)) :=
  producerNative f base r laws caller live inputLive alloc dataLaws scaleLaws template width positive covers selection rest trials order

theorem composedCorrect (base : Nat → Nat) (r : ComposedRequest)
    (safe : LaunchSafe (rows r.fpa.info) (cols r.b.info) (cols r.fpa.info))
    (laws : TensorLaws base r.fpa) (caller : Memory ℝ ℝ) (live : Nat → Prop) (inputs : r.live live)
    (alloc : RowAllocation r.row caller live) (template : State ℝ) (width : Nat) (positive : 0<width) (covers : r.row.K≤width)
    (selection : Producer.Selection (ProducerTensor.geometry r.row width positive)) (rest : Algebra ℝ)
    (order : Producer.Schedule (ProducerTensor.geometry r.row width positive)) :
    ∃ producerFinal, KernelProducerGrid.run (KernelScalarProfile.mathematical selection.tree rest) realInteger8
      (ProducerTensor.geometry r.row width positive) order.order (ProducerTensor.bind id r.row alloc.memory template)=some producerFinal ∧
      let produced:=ProducerTensor.memory id r.row alloc.data alloc.scales alloc.memory producerFinal
      let request:=ProducerTensor.producedRequest r safe caller live inputs alloc
      let joinedLive:=Admission.composedLive r caller live alloc
      ∀ (output : GemmAllocation request.original produced joinedLive)
        (gemmLaws : GemmLaws base request.original) (outputLaws : TensorLaws base output.output)
        (callerDevice : Nat) (trials : List (Host.Invocation (ComposedTensor.shape r))) (chosen : Host.Invocation (ComposedTensor.shape r)),
        Binding id base request produced joinedLive output template ∧
        Simulation id request produced joinedLive output template (KernelScalarProfile.mathematical selection.tree rest)
          mathematicalIntegers KernelIntegerIdeal.operations.toOps (·+·) (·+·)
          (Grid.zeroCell (request.original.buffers output.output.info.storage).outPrecision 0) chosen ∧
        ComposedPost id request output.output caller live callerDevice
          (fun cell => Grid.zeroCell (request.original.buffers output.output.info.storage).outPrecision
            (target (ComposedTensor.shape r) (Gemm.realInput (ComposedTensor.shape r) (ComposedTensor.realInput r caller)) cell.1 cell.2))
          (Tensor.realRun selection.tree rest request callerDevice produced joinedLive output template trials chosen) :=
  composedReal base r safe laws caller live inputs alloc template width positive covers selection rest order

theorem composedFPEquiv (f : KernelNativeModel.FloatingInstructions) (base : Nat → Nat) (r : ComposedRequest)
    (safe : LaunchSafe (rows r.fpa.info) (cols r.b.info) (cols r.fpa.info))
    (laws : TensorLaws base r.fpa) (caller : Memory Word32 Word16) (live : Nat → Prop) (inputs : r.live live)
    (alloc : RowAllocation r.row caller live) (template : State Word32) (width : Nat) (positive : 0<width) (covers : r.row.K≤width)
    (selection : Producer.Selection (ProducerTensor.geometry r.row width positive)) (rest : Algebra Word32)
    (order : Producer.Schedule (ProducerTensor.geometry r.row width positive)) :
    ∃ producerFinal, KernelProducerGrid.run (KernelScalarProfile.native f selection.tree rest) KernelNativeModel.integer8
      (ProducerTensor.geometry r.row width positive) order.order (ProducerTensor.bind KernelScalarProfile.pack16 r.row alloc.memory template)=some producerFinal ∧
      let produced:=ProducerTensor.memory KernelScalarProfile.unpack16 r.row alloc.data alloc.scales alloc.memory producerFinal
      let request:=ProducerTensor.producedRequest r safe caller live inputs alloc
      let joinedLive:=Admission.composedLive r caller live alloc
      ∀ (output : GemmAllocation request.original produced joinedLive)
        (gemmLaws : GemmLaws base request.original) (outputLaws : TensorLaws base output.output)
        (callerDevice : Nat) (trials : List (Host.Invocation (ComposedTensor.shape r))) (chosen : Host.Invocation (ComposedTensor.shape r)),
        Binding KernelScalarProfile.pack16 base request produced joinedLive output template ∧
        Simulation KernelScalarProfile.pack16 request produced joinedLive output template
          (KernelScalarProfile.native f selection.tree rest) signed32 (KernelNativeModel.operations f).toOps
          (fun x y => KernelScalarProfile.pack16 (f.atomic16 (KernelScalarProfile.unpack16 x) (KernelScalarProfile.unpack16 y)))
          f.atomic32 (Grid.zeroCell (request.original.buffers output.output.info.storage).outPrecision (FP.Value.zero .fp32)) chosen ∧
        ComposedPost KernelScalarProfile.pack16 request output.output caller live callerDevice
          (fun cell => Grid.nativeWord f selection.tree rest chosen.plans .half (request.original.buffers output.output.info.storage).outPrecision
            (ComposedTensor.shape r) chosen.config (request.original.buffers output.output.info.storage).bColumnMajor
            (ComposedTensor.nativeInput f r caller width positive selection) chosen.schedule cell)
          (Tensor.nativeRun f selection.tree rest request callerDevice produced joinedLive output template trials chosen) :=
  composedNative f base r safe laws caller live inputs alloc template width positive covers selection rest order

#axiomsClean correct
#axiomsClean fpEquiv
#axiomsClean producerCorrect
#axiomsClean producerFPEquiv
#axiomsClean composedCorrect
#axiomsClean composedFPEquiv
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Public.TensorBridge
