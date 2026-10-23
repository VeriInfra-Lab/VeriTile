import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.PublicAPI

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Public
open VeriTile.Triton FP.Structural Direct KernelExecution
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 28000
variable {F H : Type} [Inhabited F]

def ComposedPost (pack : H → F) (request : Tensor.Request) (out : TensorView) (caller : Memory F H)
    (live : Nat → Prop) (callerDevice : Nat) (reference : MatrixCell request.original.shape → FP.Structural.Cell F)
    (result : Tensor.Result F) : Prop :=
  result.tensor=out ∧ GemmDescriptor request.original result.tensor ∧ result.currentDevice=callerDevice ∧
  (∀ cell : MatrixCell request.original.shape,
    result.memory (logicalAddress result.tensor (cell.1.val*request.original.shape.N+cell.2.val))=reference cell) ∧
  (∀ p,live p.1 → ¬OutputRegion request.original.shape (request.original.buffers result.tensor.info.storage) p →
    result.memory p=Tensor.encode pack (caller p))

local instance : Inhabited Word32 := ⟨FP.Value.zero .fp32⟩

/-- Public composed native endpoint. The producer executes on the actual
prepared backing. GEMM's live set includes every original storage and both
producer allocations; its physical admission is combined with source
simulation and the independent original-caller target/frame. -/
theorem composedNative (f : KernelNativeModel.FloatingInstructions) (base : Nat → Nat) (r : ComposedRequest)
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
          (Tensor.nativeRun f selection.tree rest request callerDevice produced joinedLive output template trials chosen) := by
  obtain ⟨producerFinal,he,hpost⟩:=ComposedTensor.native_complete f base r safe laws caller live inputs alloc template width positive covers selection rest order
  rw [Admission.producer_prepared_binding KernelScalarProfile.pack16 r.row caller live (r.fpaLive live inputs) alloc template] at he
  refine ⟨producerFinal,he,?_⟩
  dsimp only
  intro output gemmLaws outputLaws callerDevice trials chosen
  let admit : Admission.Gemm base (ProducerTensor.producedRequest r safe caller live inputs alloc) _
      (Admission.composedLive r caller live alloc) output :=
    ⟨Admission.composed_inputs r safe caller live inputs alloc,gemmLaws,outputLaws⟩
  refine ⟨binding KernelScalarProfile.pack16 base _ _ _ output admit template,
    simulation KernelScalarProfile.pack16 base _ _ _ output admit template _ _ _ _ _ rfl rfl _ chosen,?_⟩
  obtain ⟨identity,descriptor,device,values,frame⟩:=hpost (Admission.composedLive r caller live alloc) output callerDevice trials chosen
  exact ⟨identity,descriptor,device,values,fun p hp outside => frame p hp (Or.inl hp) outside⟩

theorem composedReal (base : Nat → Nat) (r : ComposedRequest)
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
          (Tensor.realRun selection.tree rest request callerDevice produced joinedLive output template trials chosen) := by
  obtain ⟨producerFinal,he,hpost⟩:=ComposedTensor.real_complete base r safe laws caller live inputs alloc template width positive covers selection rest order
  rw [Admission.producer_prepared_binding id r.row caller live (r.fpaLive live inputs) alloc template] at he
  refine ⟨producerFinal,he,?_⟩
  dsimp only
  intro output gemmLaws outputLaws callerDevice trials chosen
  let admit : Admission.Gemm base (ProducerTensor.producedRequest r safe caller live inputs alloc) _
      (Admission.composedLive r caller live alloc) output :=
    ⟨Admission.composed_inputs r safe caller live inputs alloc,gemmLaws,outputLaws⟩
  refine ⟨binding id base _ _ _ output admit template,
    simulation id base _ _ _ output admit template _ _ _ _ _ rfl rfl _ chosen,?_⟩
  obtain ⟨identity,descriptor,device,values,frame⟩:=hpost (Admission.composedLive r caller live alloc) output callerDevice trials chosen
  exact ⟨identity,descriptor,device,values,fun p hp outside => frame p hp (Or.inl hp) outside⟩

#axiomsClean composedNative
#axiomsClean composedReal
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Public
