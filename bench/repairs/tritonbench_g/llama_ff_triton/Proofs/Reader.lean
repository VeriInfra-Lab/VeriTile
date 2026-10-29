import bench.repairs.tritonbench_g.llama_ff_triton.Correct
import bench.repairs.tritonbench_g.llama_ff_triton.FPEquiv

namespace VeriTile.Bench.Repairs.LlamaFF

noncomputable section

set_option maxRecDepth 30000

example (p : Parameters ℝ) (x : RowInput ℝ)
    (plans : Nat → Nat → DotPlan) (reduction : Reduction) (n : Nat) :
    ff_llama realOps p x plans reduction n=realTarget p x n := by
  exact callee_correct p x plans reduction n

example (p : Parameters ℝ) (b : Buffers) (decoded : b.useByte=Bool.false)
    (sep : b.Separate) (injective : Function.Injective (outputAddress b : Cell s → Address))
    (schedule : Schedule s) (initial : Memory ℝ ℝ) :
    (∀ x : Cell s,run idealStorage p b selectedPlans selectedReductions schedule.events initial (outputAddress b x)=
      .inr (.inr (targetAt p b initial x))) ∧
    (∀ q,(∀ x : Cell s,q≠outputAddress b x) →
      run idealStorage p b selectedPlans selectedReductions schedule.events initial q=initial q) := by
  exact correct p b decoded sep injective schedule initial

example (a : Ops F) (p : Parameters F) (x : RowInput F)
    (plans : Nat → Nat → DotPlan) (reduction : Reduction) (n : Nat) :
    ff_llama a p x plans reduction n=referenceValue a p x plans reduction n := by
  exact callee_native a p x plans reduction n

example (actual reference : StorageOps (BitVec 32) (BitVec 16))
    (leaves : NumericLeaves actual reference)
    (abi : LocalEventABI (BitVec 32) (BitVec 16) σ actual) (raw : RawByteABI abi)
    (p : Parameters (BitVec 32)) (b : Buffers) (domain : HostDomain s p b)
    (binding : CompilerBinding (BitVec 32) (BitVec 16) s p b) (state : σ)
    (caller : Memory (BitVec 32) (BitVec 16))
    (allocationFrame : ∀ q,q.1≠b.output.storage → abi.observe state q=caller q) :
    (∀ batch seq n,abi.observe ((kernel_ff s binding).native abi state).2
      (outputAddress b (logicalCell domain batch seq n))=
      wordTarget reference p b selectedPlans selectedReductions (abi.observe state) (logicalCell domain batch seq n)) ∧
    (∀ q,(∀ x : Cell s,q≠outputAddress b x) →
      abi.observe ((kernel_ff s binding).native abi state).2 q=abi.observe state q) ∧
    (∀ q,q.1≠b.output.storage → abi.observe ((kernel_ff s binding).native abi state).2 q=caller q) ∧
    (∀ (x : Cell s) (i : Fin (raw.width (outputAddress b x).1)),
      raw.bytes ((kernel_ff s binding).native abi state).2
        ((outputAddress b x).1,raw.width (outputAddress b x).1*(outputAddress b x).2+i.val)=
        cellByte (wordTarget reference p b selectedPlans selectedReductions (abi.observe state) x) i.val) ∧
    (∀ storage offset,(∀ x : Cell s,(storage,offset/raw.width storage)≠outputAddress b x) →
      raw.bytes ((kernel_ff s binding).native abi state).2 (storage,offset)=raw.bytes state (storage,offset)) ∧
    (returned s b domain.batch domain.seq domain.device domain.extents).storage=b.output.storage ∧
    (returned s b domain.batch domain.seq domain.device domain.extents).shape=[domain.batch,domain.seq,s.N] ∧
    (returned s b domain.batch domain.seq domain.device domain.extents).dtype=DType.fp16 ∧
    (∀ (device : Nat) (body : Nat × σ → Nat × σ) before,(withDevice device body before).1=before.1) := by
  exact whole_fp actual reference leaves abi raw p b domain binding state caller allocationFrame

#axiomsClean Contract.python_checks_iff
#axiomsClean CheckedHost.domain
#axiomsClean CheckedHost.active_span
#axiomsClean checked_real
#axiomsClean checked_fp
#axiomsClean checked_caller_bytes
#auditStmt checked_real
#auditStmt checked_fp
#specNonCircular decodedTarget avoiding [value,referenceValue,accumulation,run,Code.native]

#axiomsClean callee_correct

#axiomsClean correct

#axiomsClean reference_consistent

#axiomsClean whole_fp

#axiomsClean callee_native

#axiomsClean native_full_byte_frame

#axiomsClean native_output_bytes

#axiomsClean accumulation_reference

#axiomsClean moment_reference

#axiomsClean projection_reference

#axiomsClean value_reference

#axiomsClean eval_vectorCode

#axiomsClean Expr.code_eval

#axiomsClean cellCode_eval

#axiomsClean gridCode_eval

#axiomsClean Code.native_refines

#axiomsClean native_grid

#axiomsClean owner_cell

#axiomsClean lane_owner

#axiomsClean lane_unique

#axiomsClean lane_grid

#axiomsClean flat_pid_decode

#axiomsClean active_wrapped_loads

#axiomsClean source_tail_mask

#axiomsClean no_empty_lane

#axiomsClean logical_input_address

#axiomsClean logical_output_address

#axiomsClean schedule_empty

#axiomsClean hostCode_eq

#axiomsClean device_restored

#axiomsClean returned_shape

#axiomsClean whole_host

#axiomsClean formed_exact

#axiomsClean loop_formed

#axiomsClean IndexDomain.x_exact

#axiomsClean IndexDomain.output_exact

#axiomsClean original_i32_overflow

#axiomsClean repaired_i64_exact

#axiomsClean rowInput_current

#axiomsClean cellValue_current

#axiomsClean write_inputs

#axiomsClean run_frame

#axiomsClean run_output

#axiomsClean whole_word_and_frame

#axiomsClean ops_identity

#axiomsClean NumericLeaves.identity

#axiomsClean native_word_and_frame

#axiomsClean finRange_rsum

#axiomsClean Tree.real

#axiomsClean Reduction.real

#axiomsClean instructions_real

#axiomsClean dotBlock_real

#axiomsClean accumulation_real

#axiomsClean k_partition

#axiomsClean projection_sum

#axiomsClean moment_real

#axiomsClean moment_sum

#axiomsClean norm_real

#axiomsClean projection_real

#axiomsClean value_real

#axiomsClean bound_native_complete

#axiomsClean binding_unique

#axiomsClean binding_covers

#axiomsClean cellExpr_eval

#axiomsClean masked_map

#axiomsClean scaled_map

#axiomsClean instruction_map

#axiomsClean dotBlock_map

#axiomsClean accumulation_map

#axiomsClean square_map

#axiomsClean moment_map

#axiomsClean tree_map

#axiomsClean norm_map

#axiomsClean projection_map

#axiomsClean value_map

#auditStmt HostDomain

#print HostDomain

#axiomsClean HostDomain

#auditStmt CompilerBinding

#print CompilerBinding

#axiomsClean CompilerBinding

#auditStmt LocalEventABI

#print LocalEventABI

#axiomsClean LocalEventABI

#auditStmt NumericLeaves

#print NumericLeaves

#axiomsClean NumericLeaves

#auditStmt callee_correct

#print callee_correct

#auditStmt correct

#print correct

#auditStmt callee_native

#print callee_native

#auditStmt whole_fp

#print whole_fp

#specNonCircular realTarget avoiding [value,referenceValue,accumulation,run,Code.native]

#specNonCircular referenceValue avoiding [value,accumulation,moment,projection,run,Code.native]

#specNonCircular wordTarget avoiding [cellValue,run,gridCode,Code.native,boundExecute,hostCode]

end

end VeriTile.Bench.Repairs.LlamaFF
