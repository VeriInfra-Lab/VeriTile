import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Checked.Execution

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Checked
noncomputable section
set_option maxHeartbeats 8000000

/-- Native scalar/K8 conformance composes over all source K blocks and the
entire checked host call. No source success, whole-dot or result equality is
a numerical premise. All FP32 caller words, including exceptional words, fit. -/
theorem native_words (native : PrimitiveOps) (instruction : InstructionProfile) (ports : LocalPorts native instruction)
    (allocate : Allocator σ) (abi : LocalABI Word σ native.operations) (args : Host.Arguments)
    (law : AllocationLaw allocate abi live) (trials : Nat) (failures : Failures)
    (before after : DeviceState σ) (out : Tensor)
    (aBound : args.A.Bounded) (bBound : args.B.Bounded)
    (aLive : live before.payload args.A.storage) (bLive : live before.payload args.B.storage)
    (success : call allocate abi (.tensors args) trials failures before = .ok out after) :
    MetadataContract args ∧ (∃ ready,
      allocate.empty args.A.rows args.B.columns .fp16 args.A.device before = .ok out ready ∧
      Host.Allocation live args.A.rows args.B.columns .fp16 args.A.device before.payload out ready.payload) ∧
    (∀ row column, row < args.A.rows → column < args.B.columns →
      abi.observe after.payload (out.address row column) = reference (fpReference instruction) args (abi.observe before.payload) row column) ∧
    after.currentDevice = before.currentDevice := by
  obtain ⟨admitted,allocated,words,device⟩ := values allocate abi args law trials failures before after out aBound bBound aLive bLive success
  refine ⟨admitted,allocated,?_,device⟩
  intro row column hr hc
  exact (words row column hr hc).trans
    (congrArg (fun ops => reference ops args (abi.observe before.payload) row column) (operations_eq ports))

/-- Both actual output bytes, observed independently from typed words, plus
the exact allocated return descriptor and consumed/restored device state. -/
specification fp_equiv (native : PrimitiveOps) (instruction : InstructionProfile) (ports : LocalPorts native instruction)
    (allocate : Allocator σ) (abi : LocalABI Word σ native.operations) (args : Host.Arguments)
    (law : AllocationLaw allocate abi live) (raw : ByteABI abi backing unit)
    (inputA : TensorBinding backing unit args.A) (inputB : TensorBinding backing unit args.B)
    (bytes : AllocationByteLaw allocate abi raw live) (trials : Nat) (failures : Failures)
    (before after : DeviceState σ) (out : Tensor)
    (aBound : args.A.Bounded) (bBound : args.B.Bounded)
    (aLive : live before.payload args.A.storage) (bLive : live before.payload args.B.storage)
    (success : call allocate abi (.tensors args) trials failures before = .ok out after) :
    MetadataContract args ∧ (∃ ready,
      allocate.empty args.A.rows args.B.columns .fp16 args.A.device before = .ok out ready ∧
      Host.Allocation live args.A.rows args.B.columns .fp16 args.A.device before.payload out ready.payload) ∧
    (∀ row column lane, row < args.A.rows → column < args.B.columns → lane < 2 →
      raw.observe after.payload out.device (out.allocationBase.toNat+2*(out.address row column).2+lane) =
        wordByte (reference (fpReference instruction) args (abi.observe before.payload) row column) lane) ∧
    after.currentDevice = before.currentDevice := by
  obtain ⟨admitted,allocated,words,device⟩ :=
    native_words native instruction ports allocate abi args law trials failures before after out aBound bBound aLive bLive success
  obtain ⟨ready,returned,allocation⟩ := allocated
  have physical := bytes.output _ _ _ _ _ _ returned
  have width : unit out.storage = 2 := physical.tensor.width.trans (by rw [allocation.dtype]; rfl)
  refine ⟨admitted,⟨ready,returned,allocation⟩,?_,device⟩
  intro row column lane hr hc hl
  have logical := out.logical_bound allocation.bounded row column
    (by rw [allocation.rowsExact]; exact hr) (by rw [allocation.columnsExact]; exact hc)
  have capacity : unit (out.address row column).1*((out.address row column).2+1) ≤ (backing (out.address row column).1).bytes := by
    change unit out.storage*((out.address row column).2+1) ≤ (backing out.storage).bytes
    rw [width,physical.tensor.capacity]
    have bound := logical.1
    rw [allocation.dtype] at bound
    simpa only [Storage.bytes,Nat.mul_add,Nat.mul_one] using bound
  have codec := raw.codec after.payload (out.address row column) capacity lane (by simpa only [Tensor.address,width] using hl)
  rw [words row column hr hc] at codec
  simpa only [Tensor.address,Prod.fst,Prod.snd,width,physical.tensor.owner,physical.tensor.base] using codec

/-- Every caller backing byte survives validation, allocation, autotuner or
kernel-prefix failures, device entry/exit errors, zero-fill and success. -/
theorem frame (allocate : Allocator σ) (abi : LocalABI Word σ ops) (raw : ByteABI abi backing unit)
    (args : Host.Arguments) (law : AllocationLaw allocate abi live) (bytes : AllocationByteLaw allocate abi raw live)
    (state : DeviceState σ) (aBound : args.A.Bounded) (bBound : args.B.Bounded)
    (aLive : live state.payload args.A.storage) (bLive : live state.payload args.B.storage)
    (trials : Nat) (failures : Failures) (old owner address : Nat)
    (present : live state.payload old) (covered : (backing old).covers owner address) :
    raw.observe (call allocate abi (.tensors args) trials failures state).state.payload owner address =
      raw.observe state.payload owner address :=
  call_frame allocate abi raw args law bytes state aBound bBound aLive bLive trials failures old owner address present covered

#axiomsClean native_words
#axiomsClean fp_equiv
#axiomsClean frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Checked
