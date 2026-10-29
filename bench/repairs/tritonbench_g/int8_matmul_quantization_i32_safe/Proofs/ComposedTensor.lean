import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.ProducerTensor
import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.InputCongruence

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.ComposedTensor
open VeriTile.Triton FP.Structural Direct KernelExecution
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
variable {F H : Type} [Inhabited F]

def shape (r : ComposedRequest) : Shape := Direct.shape r.fpa.info r.b.info

def input (pack : H → F) (r : ComposedRequest) (caller : Memory F H)
    (A : Nat → Nat → Byte) (scales : Nat → H) : Gemm.Input F where
  A := A
  B := fun k n => read8 caller (logicalAddress r.b (k*(shape r).N+n))
  aScale := fun row => pack (scales row)
  bScale := fun n => (Tensor.encode pack (caller (logicalAddress r.bScale n))).read .real

/-- Pointwise original-Tensor substitution of proved producer stores.
Only stored bytes/half words and the producer frame occur in the premises. -/
theorem input_agree (pack : H → F) (r : ComposedRequest)
    (safe : LaunchSafe (rows r.fpa.info) (cols r.b.info) (cols r.fpa.info))
    (caller : Memory F H) (live : Nat → Prop) (inputs : r.live live) (alloc : RowAllocation r.row caller live)
    (produced : Memory F H) (A : Nat → Nat → Byte) (scales : Nat → H)
    (data : ∀ g : Fin r.row.G, ∀ k : Fin r.row.K, produced (logicalAddress alloc.data (g.val*r.row.K+k.val))=.inl (A g k))
    (scale : ∀ g : Fin r.row.G, produced (logicalAddress alloc.scales g.val)=.inr (.inr (scales g)))
    (frame : ∀ p,live p.1 → produced p=caller p) (cell : MatrixCell (shape r)) :
    Gemm.Agree (shape r) cell.1 cell.2
      (Tensor.originalInput pack (ProducerTensor.producedRequest r safe caller live inputs alloc).original produced)
      (input pack r caller A scales) := by
  have identities:=Direct.producedRequest_original r caller live inputs alloc
  constructor
  · intro k hk
    change read8 produced (logicalAddress (Direct.producedRequest r caller live inputs alloc).a
      (cell.1.val*r.row.K+k))=A cell.1 k
    rw [identities.1]
    unfold read8 Int8DequantMatmul.read8
    rw [data cell.1 ⟨k,hk⟩]
  · intro k hk
    change read8 produced (logicalAddress r.b (k*(shape r).N+cell.2.val))=
      read8 caller (logicalAddress r.b (k*(shape r).N+cell.2.val))
    unfold read8 Int8DequantMatmul.read8
    rw [frame _ (inputs r.b (by simp [ComposedRequest.views]))]
  · change (Tensor.encode pack (produced (logicalAddress (Direct.producedRequest r caller live inputs alloc).aScale cell.1.val))).read .fp16 = _
    rw [identities.2,scale cell.1]
    simp [Tensor.encode,input]
  · change (Tensor.encode pack (produced (logicalAddress r.bScale cell.2.val))).read .real = _
    rw [frame _ (inputs r.bScale (by simp [ComposedRequest.views]))]
    rfl

local instance : Inhabited Word32 := ⟨FP.Value.zero .fp32⟩
def nativeInput (f : KernelNativeModel.FloatingInstructions) (r : ComposedRequest) (caller : Memory Word32 Word16)
    (width : Nat) (positive : 0<width) (selection : Producer.Selection (ProducerTensor.geometry r.row width positive)) : Gemm.Input Word32 :=
  input KernelScalarProfile.pack16 r caller
    (fun g k => KernelProducerTargets.nativeByte f (ProducerTensor.geometry r.row width positive) selection
      (ProducerTensor.originalValues KernelScalarProfile.pack16 r.row caller g) k)
    (fun g => KernelProducerTargets.nativeStoredScale f (ProducerTensor.geometry r.row width positive) selection
      (ProducerTensor.originalValues KernelScalarProfile.pack16 r.row caller g))

/-- Whole composed native Tensor endpoint: the actual producer execution
builds the memory consumed by GEMM. The output graph uses only original
caller tensors and independently defined producer byte/half formulas. -/
theorem native_complete (f : KernelNativeModel.FloatingInstructions) (base : Nat → Nat) (r : ComposedRequest)
    (safe : LaunchSafe (rows r.fpa.info) (cols r.b.info) (cols r.fpa.info))
    (laws : TensorLaws base r.fpa) (caller : Memory Word32 Word16) (live : Nat → Prop) (inputs : r.live live)
    (alloc : RowAllocation r.row caller live) (template : State Word32) (width : Nat) (positive : 0<width) (covers : r.row.K≤width)
    (selection : Producer.Selection (ProducerTensor.geometry r.row width positive)) (rest : Algebra Word32)
    (order : Producer.Schedule (ProducerTensor.geometry r.row width positive)) :
    ∃ producerFinal, KernelProducerGrid.run (KernelScalarProfile.native f selection.tree rest) KernelNativeModel.integer8
      (ProducerTensor.geometry r.row width positive) order.order (ProducerTensor.bind KernelScalarProfile.pack16 r.row caller template)=some producerFinal ∧
      let produced:=ProducerTensor.memory KernelScalarProfile.unpack16 r.row alloc.data alloc.scales alloc.memory producerFinal
      let request:=ProducerTensor.producedRequest r safe caller live inputs alloc
      ∀ (gemmLive : Nat → Prop) (output : GemmAllocation request.original produced gemmLive)
        (callerDevice : Nat) (trials : List (Host.Invocation (shape r))) (chosen : Host.Invocation (shape r)),
        let result:=Tensor.nativeRun f selection.tree rest request callerDevice produced gemmLive output template trials chosen
        result.tensor=output.output ∧ GemmDescriptor request.original result.tensor ∧ result.currentDevice=callerDevice ∧
          (∀ cell : MatrixCell (shape r),result.memory (logicalAddress result.tensor (cell.1.val*(shape r).N+cell.2.val))=
            Grid.nativeWord f selection.tree rest chosen.plans .half (request.original.buffers output.output.info.storage).outPrecision
              (shape r) chosen.config (request.original.buffers output.output.info.storage).bColumnMajor
              (nativeInput f r caller width positive selection) chosen.schedule cell) ∧
          (∀ p,live p.1 → gemmLive p.1 → ¬OutputRegion (shape r) (request.original.buffers result.tensor.info.storage) p →
            result.memory p=Tensor.encode KernelScalarProfile.pack16 (caller p)) := by
  obtain ⟨producerFinal,he,hq,hs,hframe⟩ := ProducerTensor.native_complete f base r.row laws caller live alloc template width positive covers selection rest order
  refine ⟨producerFinal,he,?_⟩
  dsimp only
  intro gemmLive output callerDevice trials chosen
  obtain ⟨identity,descriptor,device,values,frame⟩ := Tensor.native_complete f selection.tree rest
    (ProducerTensor.producedRequest r safe caller live inputs alloc) callerDevice _ gemmLive output template trials chosen
  refine ⟨identity,descriptor,device,?_,?_⟩
  · intro cell
    refine (values cell).trans ?_
    apply Gemm.cell_fold_congr
    exact input_agree KernelScalarProfile.pack16 r safe caller live inputs alloc _ _ _ hq hs hframe cell
  · intro p hp hg outside
    rw [frame p hg outside,hframe p hp]

def realInput (r : ComposedRequest) (caller : Memory ℝ ℝ) : Gemm.Input ℝ :=
  input id r caller
    (fun g k => KernelProducerTargets.mathematicalByte (ProducerTensor.originalValues id r.row caller g) r.row.K k)
    (fun g => KernelProducerTargets.mathematicalScale (ProducerTensor.originalValues id r.row caller g) r.row.K)

theorem target_congr (s : Shape) (left right : Gemm.Input ℝ) (cell : MatrixCell s)
    (same : Gemm.Agree s cell.1 cell.2 left right) :
    target s (Gemm.realInput s left) cell.1 cell.2=target s (Gemm.realInput s right) cell.1 cell.2 := by
  simp only [target,Gemm.realInput]
  rw [same.aScale,same.bScale]
  congr 2
  apply Int8DequantMatmul.rsum_congr
  intro k hk
  rw [same.A k hk,same.B k hk]

/-- Independent ideal Real composed formula; native equality to it is not
assumed or claimed. The actual produced Tensor memory is still passed on. -/
theorem real_complete (base : Nat → Nat) (r : ComposedRequest)
    (safe : LaunchSafe (rows r.fpa.info) (cols r.b.info) (cols r.fpa.info))
    (laws : TensorLaws base r.fpa) (caller : Memory ℝ ℝ) (live : Nat → Prop) (inputs : r.live live)
    (alloc : RowAllocation r.row caller live) (template : State ℝ) (width : Nat) (positive : 0<width) (covers : r.row.K≤width)
    (selection : Producer.Selection (ProducerTensor.geometry r.row width positive)) (rest : Algebra ℝ)
    (order : Producer.Schedule (ProducerTensor.geometry r.row width positive)) :
    ∃ producerFinal, KernelProducerGrid.run (KernelScalarProfile.mathematical selection.tree rest) realInteger8
      (ProducerTensor.geometry r.row width positive) order.order (ProducerTensor.bind id r.row caller template)=some producerFinal ∧
      let produced:=ProducerTensor.memory id r.row alloc.data alloc.scales alloc.memory producerFinal
      let request:=ProducerTensor.producedRequest r safe caller live inputs alloc
      ∀ (gemmLive : Nat → Prop) (output : GemmAllocation request.original produced gemmLive)
        (callerDevice : Nat) (trials : List (Host.Invocation (shape r))) (chosen : Host.Invocation (shape r)),
        let result:=Tensor.realRun selection.tree rest request callerDevice produced gemmLive output template trials chosen
        result.tensor=output.output ∧ GemmDescriptor request.original result.tensor ∧ result.currentDevice=callerDevice ∧
          (∀ cell : MatrixCell (shape r),result.memory (logicalAddress result.tensor (cell.1.val*(shape r).N+cell.2.val))=
            Grid.zeroCell (request.original.buffers output.output.info.storage).outPrecision
              (target (shape r) (Gemm.realInput (shape r) (realInput r caller)) cell.1 cell.2)) ∧
          (∀ p,live p.1 → gemmLive p.1 → ¬OutputRegion (shape r) (request.original.buffers result.tensor.info.storage) p →
            result.memory p=Tensor.encode id (caller p)) := by
  obtain ⟨producerFinal,he,hq,hs,hframe⟩ := ProducerTensor.real_complete base r.row laws caller live alloc template width positive covers selection rest order
  refine ⟨producerFinal,he,?_⟩
  dsimp only
  intro gemmLive output callerDevice trials chosen
  obtain ⟨identity,descriptor,device,values,frame⟩ := Tensor.real_complete selection.tree rest
    (ProducerTensor.producedRequest r safe caller live inputs alloc) callerDevice _ gemmLive output template trials chosen
  refine ⟨identity,descriptor,device,?_,?_⟩
  · intro cell
    refine (values cell).trans ?_
    apply congrArg (Grid.zeroCell _)
    apply target_congr
    exact input_agree id r safe caller live inputs alloc _ _ _ hq hs hframe cell
  · intro p hp hg outside
    rw [frame p hg outside,hframe p hp]

#axiomsClean input_agree
#axiomsClean native_complete
#axiomsClean real_complete
#specNonCircular nativeInput avoiding [KernelProducerGrid.run, Tensor.nativeRun, ProducerTensor.memory]
#specNonCircular realInput avoiding [KernelProducerGrid.run, Tensor.realRun, ProducerTensor.memory]
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.ComposedTensor
