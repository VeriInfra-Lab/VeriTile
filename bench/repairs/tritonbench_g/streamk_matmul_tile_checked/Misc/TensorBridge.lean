import bench.repairs.tritonbench_g.streamk_matmul_tile_checked.Correct
import bench.repairs.tritonbench_g.streamk_matmul_tile_checked.Misc.NativeCorrect

/-! Optional correspondence with the existing Python input checks. -/
namespace VeriTile.Bench.Repairs.StreamK.TileChecked.TensorBridge
open Whole001 Checked TensorModel KernelMemory
open VeriTile.Torch VeriTile.Triton
noncomputable section
set_option maxHeartbeats 12000000

theorem python_checks_iff_preconditions (a b : ArgumentMetadata) (config : Contract.Config) :
    Contract.checkInputs a b config=.ok () ↔ Contract.AdmittedArguments a b config :=
  Contract.checkInputs_iff a b config

/-- Independent mathematical result on the actual caller Tensor, together with
input-storage and exterior-memory preservation. No kernel execution appears in
this predicate. Real memory is an arbitrary ideal interpretation of FP32 slots. -/
def RealResult (a b out : FP32.Tensor) (p : Source.Parameters) (v : ByteExecution.Layout p)
    (before after : Nat → ℝ) : Prop :=
  (∀ cell : Returned.Cell p,
    logicalReal after out (Returned.index p cell).val=matrixProduct before a b cell.1.val cell.2.val) ∧
  (∀ address,(a.storage.base≤address ∧ address<a.storage.base+4*a.storage.elements) ∨
      (b.storage.base≤address ∧ address<b.storage.base+4*b.storage.elements) → after address=before address) ∧
  (∀ address,ByteMemory.Outside (ByteExecution.cView v) address →
    ByteMemory.Outside (ByteExecution.sView v) address → after address=before address)

/-- Kernel-only theorem. The wrapper supplies the exact input contract; output
and scratch bindings describe the caller-provided memory, not allocator or
runtime correctness. Source execution success is proved in the conclusion. -/
theorem kernel_correct (a b out : FP32.Tensor) (la : FP32.TensorLaws a) (lb : FP32.TensorLaws b)
    (config : Contract.Config) (h : Contract.InputContract a.metadata b.metadata config)
    (v : ByteExecution.Layout (params a b config h))
    (buffers : Buffers a b (params a b config h) v out) (before : Nat → ℝ)
    (initial : Source.State ℝ (params a b config h) exterior)
    (inputs : TensorBinding.Inputs before (legacy a) (legacy b) config h.prior.profile initial)
    (memory : ValueBinding v initial before) :
    Spec.Real (∃ final,
      runValues (params a b config h) Arithmetic.real v initial before=some final ∧
      RealResult a b out (params a b config h) v before final.2 ∧
      final.1.inputA=initial.inputA ∧ final.1.inputB=initial.inputB ∧ final.1.exterior=initial.exterior ∧
      (¬Checked.Contract.Empty a.metadata b.metadata →
        Contract.StaticTiles config ∧ ABI.Valid (params a b config h) ∧
        Index.Conformance (legacy a) (legacy b) config h.prior.profile Arithmetic.real before)) := by
  let p := params a b config h
  let logical := Source.call p Arithmetic.real initial
  let after := ValueMemory.rendered before v logical
  have input := TensorBinding.source_inputs before (legacy a) (legacy b) config
    (legacy_laws a la h.prior.inputs.2.2.2.2.2.2.2.2.2.1)
    (legacy_laws b lb h.prior.inputs.2.2.2.2.2.2.2.2.2.2.1) h.prior initial inputs
  have values : ∀ cell : Returned.Cell p,logical.output (Returned.index p cell)=
      matrixProduct before a b cell.1.val cell.2.val := by
    intro cell
    have result := Returned.real_contract p initial cell
    change logical.output (Returned.index p cell)=_ at result
    rw [input.1,input.2] at result
    simpa only [Reference.matrixProduct,TensorBinding.left,TensorBinding.right,TensorBinding.parameters,
      Checked.Contract.plan,TensorBinding.logicalValue,TensorBinding.physicalAddress,legacy,
      params,p,logicalReal,matrixProduct,FP32.Storage.address,Nat.mul_one] using result
  refine ⟨(logical,after),values_run p Arithmetic.real v buffers.memory initial before memory,
    ⟨?_,?_,?_⟩,(Source.call_frame p Arithmetic.real initial).1,
    (Source.call_frame p Arithmetic.real initial).2.1,(Source.call_frame p Arithmetic.real initial).2.2,?_⟩
  · intro cell
    rw [←real_output_read v out buffers.output after (Returned.index p cell)]
    exact (ValueMemory.render_at _ v.outputBase logical.output (Returned.index p cell)).trans (values cell)
  · exact values_input_frame a b v out buffers logical before
  · exact values_frame v logical before
  · intro nonempty
    exact ⟨Contract.static_nonempty a.metadata b.metadata config h nonempty,
      Envelope.valid (legacy a) (legacy b) config h.prior nonempty,
      Index.conformance (legacy a) (legacy b) config h.prior nonempty Arithmetic.real before⟩



/-- Input-only ordered native reference. FP32 words include exceptional values.
The same fixed MMA ISA model is used on both sides; scalar half conversion and
atomic addition use their established primitive models. No numerical rewrite
or finite-operand assumption is introduced. -/
def target (ref : Native.Reference) (before : ByteMemory.Read) (a b : FP32.Tensor)
    (la : FP32.TensorLaws a) (lb : FP32.TensorLaws b) (config : Contract.Config)
    (h : Contract.InputContract a.metadata b.metadata config) : Returned.Cell (params a b config h) → FP32.Word :=
  Returned.value (params a b config h) (Native.operations (Native.referenceInstructions ref))
    (fun i => FP32.logicalValue (TensorBinding.readWord before) a la i.val)
    (fun i => FP32.logicalValue (TensorBinding.readWord before) b lb i.val)

def NativeResult (ref : Native.Reference) (before after : ByteMemory.Read) (a b out : FP32.Tensor)
    (la : FP32.TensorLaws a) (lb : FP32.TensorLaws b) (config : Contract.Config)
    (h : Contract.InputContract a.metadata b.metadata config)
    (v : ByteExecution.Layout (params a b config h))
    (output : OutputBinding (params a b config h) v out) : Prop :=
  (∀ cell : Returned.Cell (params a b config h),
    FP32.logicalValue (TensorBinding.readWord after) out output.laws (Returned.index _ cell).val=
      target ref before a b la lb config h cell) ∧
  (∀ cell : Returned.Cell (params a b config h),∀ j,j<4 →
    after (v.outputBase+4*(Returned.index _ cell).val+j)=
      ByteMemory.byte (target ref before a b la lb config h cell).bits j) ∧
  (∀ address,(a.storage.base≤address ∧ address<a.storage.base+4*a.storage.elements) ∨
      (b.storage.base≤address ∧ address<b.storage.base+4*b.storage.elements) → after address=before address) ∧
  (∀ address,ByteMemory.Outside (ByteExecution.cView v) address →
    ByteMemory.Outside (ByteExecution.sView v) address → after address=before address)

/-- Exact native kernel result, actual Tensor observation and four output bytes
per cell. The full-tile FP16 round trip and stream atomic order are retained;
no host success, whole lowering, arithmetic result or experiment is a premise. -/
theorem kernel_native (ref : Native.Reference) (a b out : FP32.Tensor)
    (la : FP32.TensorLaws a) (lb : FP32.TensorLaws b) (config : Contract.Config)
    (h : Contract.InputContract a.metadata b.metadata config)
    (v : ByteExecution.Layout (params a b config h))
    (buffers : Buffers a b (params a b config h) v out) (before : ByteMemory.Read)
    (initial : Source.State FP32.Word (params a b config h) exterior)
    (inputs : TensorBinding.Inputs (TensorBinding.readWord before) (legacy a) (legacy b) config h.prior.profile initial)
    (memory : ByteBinding v initial before) :
    ∃ final,
      runBytes (params a b config h) (Native.operations (Native.referenceInstructions ref)) v initial before=some final ∧
      NativeResult ref before final.2 a b out la lb config h v buffers.output ∧
      final.1.inputA=initial.inputA ∧ final.1.inputB=initial.inputB ∧ final.1.exterior=initial.exterior ∧
      (¬Checked.Contract.Empty a.metadata b.metadata →
        Contract.StaticTiles config ∧ ABI.Valid (params a b config h) ∧
        Index.Conformance (legacy a) (legacy b) config h.prior.profile
          (Native.operations (Native.referenceInstructions ref)) (TensorBinding.readWord before)) := by
  let p := params a b config h
  let ops := Native.operations (Native.referenceInstructions ref)
  let logical := Source.call p ops initial
  let after := ByteExecution.rendered before v logical
  have input := TensorBinding.source_inputs (TensorBinding.readWord before) (legacy a) (legacy b) config
    (legacy_laws a la h.prior.inputs.2.2.2.2.2.2.2.2.2.1)
    (legacy_laws b lb h.prior.inputs.2.2.2.2.2.2.2.2.2.2.1) h.prior initial inputs
  have values : ∀ cell : Returned.Cell p,logical.output (Returned.index p cell)=
      target ref before a b la lb config h cell := by
    intro cell
    have result := Returned.value_contract p ops initial cell
    change logical.output (Returned.index p cell)=_ at result
    rw [input.1,input.2] at result
    have leftEq : TensorBinding.left (TensorBinding.readWord before) (legacy a) (legacy b) config h.prior.profile=
        (fun i => FP32.logicalValue (TensorBinding.readWord before) a la i.val) := by
      funext i
      exact word_logical _ a la h.prior.inputs.2.2.2.2.2.2.2.2.2.2.2.1 i.val
    have rightEq : TensorBinding.right (TensorBinding.readWord before) (legacy a) (legacy b) config h.prior.profile=
        (fun i => FP32.logicalValue (TensorBinding.readWord before) b lb i.val) := by
      funext i
      exact word_logical _ b lb h.prior.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1 i.val
    rw [leftEq,rightEq] at result
    exact result

  refine ⟨(logical,after),bytes_run p ops v buffers.memory initial before memory,
    ⟨?_,?_,?_,?_⟩,(Source.call_frame p ops initial).1,
    (Source.call_frame p ops initial).2.1,(Source.call_frame p ops initial).2.2,?_⟩
  · intro cell
    rw [←native_output_read v out buffers.output (TensorBinding.readWord after) (Returned.index p cell)]
    apply congrArg (fun bits : BitVec 32 => (⟨bits⟩ : FP32.Word))
    change ByteMemory.read32 after (v.outputBase+4*(Returned.index p cell).val)=_
    exact (ByteMemory.render_read _ (ByteExecution.cView v) (ByteMemory.extend logical.output)
      (Returned.index p cell).val (Returned.index p cell).isLt).trans
      (congrArg (fun x : FP32.Word => x.bits)
        ((ByteMemory.extend_at logical.output (Returned.index p cell)).trans (values cell)))
  · intro cell j hj
    change ByteMemory.render _ (ByteExecution.cView v) (ByteMemory.extend logical.output)
      ((ByteExecution.cView v).base+4*(Returned.index p cell).val+j)=_
    rw [ByteMemory.render_at _ (ByteExecution.cView v) _ _ j (Returned.index p cell).isLt hj,
      ByteMemory.extend_at,values cell]
  · exact bytes_input_frame a b v out buffers logical before
  · exact bytes_frame v logical before
  · intro nonempty
    exact ⟨Contract.static_nonempty a.metadata b.metadata config h nonempty,
      Envelope.valid (legacy a) (legacy b) config h.prior nonempty,
      Index.conformance (legacy a) (legacy b) config h.prior nonempty ops (TensorBinding.readWord before)⟩


#axiomsClean kernel_correct
#axiomsClean kernel_native
#axiomsClean python_checks_iff_preconditions
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.TensorBridge
