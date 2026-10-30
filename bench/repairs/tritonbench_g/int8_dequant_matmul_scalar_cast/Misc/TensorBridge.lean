import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.IOPhysical
import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.KernelExecution
import bench.repairs.tritonbench_g.int8_dequant_matmul.Proofs.DirectReal
import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.QuoteSchedule
import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.RealKernelResult
import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.NativeKernelResult

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.TensorBridge
open Direct
noncomputable section
set_option maxHeartbeats 3000000

/-- Successful full-grid kernel event execution, the independent mathematical
target on original TensorView inputs, and every non-output element. This is the
ideal Real interpretation, not an accuracy equality for rounded native output. -/
theorem correct_model (base : Nat → Nat) (r : Request) (laws : RequestLaws base r)
    (out : Nat) (c : Config) (p : Prepared r out c)
    (selected : Selection (requestShape r) c) (events : List (Commit (requestShape r) c))
    (order : events.Perm (allCommits (requestShape r) c)) (initial : Memory ℝ ℝ)
    (zero : Zeroed realOps (requestShape r) c (requestBuffers r out) initial) :
    ∃ final,
      execute realOps r out c selected (1/16129) events initial=((),final) ∧
      OperandPointers base r ∧
      (∀ cell : MatrixCell (requestShape r),
        final (outputAddress (requestShape r) (requestBuffers r out) cell)=
          .inr (.inr (target (requestShape r) (logicalInput r initial) cell.1 cell.2))) ∧
      (∀ address : Address,
        (∀ cell : MatrixCell (requestShape r),
          address≠outputAddress (requestShape r) (requestBuffers r out) cell) →
        final address=initial address) := by
  let q:=Schedule.ofPermutation (requestShape r) c events order
  refine ⟨(execute realOps r out c selected (1/16129) events initial).2,?_,
    operand_pointers base r laws,?_,?_⟩
  · rw [execution_eq]
  · intro cell
    have result:=output_word realOps r out c p selected (1/16129) q initial zero cell
    rw [real_reference,target_original base r laws] at result
    exact result
  · intro address outside
    exact frame realOps r out c selected (1/16129) events initial address outside

/-- The complete physical source commit schedule succeeds. Its values, pointers,
masks and FP16 commit operations are extracted from actual quotation execution,
and the resulting memory equals the independent Real target on original views. -/
theorem correct (base : Nat → Nat) (r : Request) (laws : RequestLaws base r)
    (out : Nat) (c : Config) (p : Prepared r out c)
    (selected : Selection (requestShape r) c) (events : List (PhysicalCommit (requestShape r) c))
    (order : (events.map PhysicalCommit.logical).Perm (allCommits (requestShape r) c))
    (initial : Memory ℝ ℝ)
    (zero : Zeroed realOps (requestShape r) c (requestBuffers r out) initial) :
    ∃ final,
      Quote.sourceGrid (Quote.idealInteger (requestShape r) c selected)
        (requestShape r) c (hasBias r) (requestBuffers r out) (1/16129) events initial=some final ∧
      execute realOps r out c selected (1/16129) (events.map PhysicalCommit.logical) initial=((),final) ∧
      OperandPointers base r ∧
      (∀ cell : MatrixCell (requestShape r),
        final (outputAddress (requestShape r) (requestBuffers r out) cell)=
          .inr (.inr (target (requestShape r) (logicalInput r initial) cell.1 cell.2))) ∧
      (∀ address : Address,
        (∀ cell : MatrixCell (requestShape r),
          address≠outputAddress (requestShape r) (requestBuffers r out) cell) →
        final address=initial address) := by
  obtain ⟨final,execution,pointers,values,framing⟩:=correct_model base r laws out c p selected
    (events.map PhysicalCommit.logical) order initial zero
  refine ⟨final,?_,execution,pointers,values,framing⟩
  rw [Quote.sourceGrid_eq _ _ _ _ _ _ selected rfl]
  change some (gridRun Quote.integerRealOps _ _ _ _ _ _ _ _)=some final
  rw [integer_grid_real]
  rw [execution_eq] at execution
  exact congrArg (fun p : Unit×Memory ℝ ℝ=>some p.2) execution

/-- Precisely the bytes written by active output lanes. -/
def OutputByte (r : Request) (out : Nat) (address : Address) : Prop :=
  ∃ cell : MatrixCell (requestShape r), ∃ i : Fin 2,
    address=(out,2*(cell.1.val*(requestShape r).N+cell.2.val)+i.val)

theorem byte_index_outside (r : Request) (out : Nat) (dtype : Nat → DType)
    (kind : dtype out=.fp16) (address : Address) (outside : ¬OutputByte r out address) :
    ∀ cell : MatrixCell (requestShape r),
      (address.1,address.2/(dtype address.1).bytes)≠
        outputAddress (requestShape r) (requestBuffers r out) cell := by
  intro cell same
  have storage:=congrArg Prod.fst same
  have index:=congrArg Prod.snd same
  simp only [outputAddress,requestBuffers,buffers,Prod.fst,Prod.snd,Nat.zero_add] at storage index
  rw [storage,kind] at index
  change address.2/2=cell.1.val*(requestShape r).N+cell.2.val at index
  apply outside
  refine ⟨cell,⟨address.2%2,Nat.mod_lt _ (by decide)⟩,?_⟩
  apply Prod.ext storage
  simp only [Prod.snd]
  rw [←index]
  omega

theorem fp_equiv_model (base : Nat → Nat) (r : Request) (laws : RequestLaws base r)
    (out : Nat) (c : Config) (p : Prepared r out c)
    (profile : FixedISAProfile) (selected : Selection (requestShape r) c)
    (events : List (Commit (requestShape r) c))
    (order : events.Perm (allCommits (requestShape r) c))
    (abi : LocalEventABI (BitVec 32) Word32 Word16 σ (fpOps profile))
    (raw : RawByteABI abi) (initial : σ)
    (input : PublicInput abi raw (requestShape r) (requestBuffers r out) (hasBias r) initial)
    (zero : Zeroed (fpOps profile) (requestShape r) c (requestBuffers r out) (abi.observe initial)) :
    ∃ final,
      (program r out c selected Source.runtimeDivisor events).native abi initial=((),final) ∧
      OperandPointers base r ∧
      (∀ cell : MatrixCell (requestShape r),
        abi.observe final (outputAddress (requestShape r) (requestBuffers r out) cell)=
          .inr (.inr (logicalWordReference (fpOps profile) (requestShape r) c selected
            Source.runtimeDivisor (Schedule.ofPermutation (requestShape r) c events order)
            (originalWords (fpOps profile) r (abi.observe initial)) cell))) ∧
      (∀ cell : MatrixCell (requestShape r), ∀ i : Fin 2,
        raw.bytes final (out,2*(cell.1.val*(requestShape r).N+cell.2.val)+i.val)=
          (logicalWordReference (fpOps profile) (requestShape r) c selected Source.runtimeDivisor
            (Schedule.ofPermutation (requestShape r) c events order)
            (originalWords (fpOps profile) r (abi.observe initial)) cell).bits.extractLsb' (8*i.val) 8) ∧
      (∀ address : Address,¬OutputByte r out address → raw.bytes final address=raw.bytes initial address) := by
  let q:=Schedule.ofPermutation (requestShape r) c events order
  let result:=(program r out c selected Source.runtimeDivisor events).native abi initial
  have execution:=native_execution abi r out c selected Source.runtimeDivisor events initial
  change (result.1,abi.observe result.2)=_ at execution
  have memory:=congrArg Prod.snd execution
  simp only [Prod.snd] at memory
  have values : ∀ cell : MatrixCell (requestShape r),
      abi.observe result.2 (outputAddress (requestShape r) (requestBuffers r out) cell)=
        .inr (.inr (logicalWordReference (fpOps profile) (requestShape r) c selected
          Source.runtimeDivisor q (originalWords (fpOps profile) r (abi.observe initial)) cell)) := by
    intro cell
    rw [memory]
    have value:=output_word (fpOps profile) r out c p selected Source.runtimeDivisor q
      (abi.observe initial) zero cell
    rw [word_original base r laws] at value
    exact value
  refine ⟨result.2,?_,operand_pointers base r laws,values,?_,?_⟩
  · exact Prod.ext (Subsingleton.elim _ _) rfl
  · intro cell i
    have kind : raw.dtype out=.fp16:=input.outKind
    rw [raw.decode,kind]
    simp only [DType.bytes,Nat.mul_add_div (by decide : 0<2),Nat.div_eq_of_lt i.isLt,
      Nat.add_zero,Nat.mul_add_mod,Nat.mul_mod_right,Nat.zero_add,Nat.mod_eq_of_lt i.isLt]
    have value:=values cell
    simp only [outputAddress,requestBuffers,buffers,Nat.zero_add] at value
    rw [value]
    rfl
  · intro address outside
    rw [raw.decode,raw.decode,memory]
    rw [frame (fpOps profile) r out c selected Source.runtimeDivisor events (abi.observe initial)
      (address.1,address.2/(raw.dtype address.1).bytes)
      (byte_index_outside r out raw.dtype input.outKind address outside)]

#axiomsClean byte_index_outside
#axiomsClean fp_equiv_model

/-- Actual quotation-derived store descriptors and native fixed-leaf events
produce the same complete output words, both bytes, and non-output byte frame. -/
theorem fp_equiv (base : Nat → Nat) (r : Request) (laws : RequestLaws base r)
    (out : Nat) (c : Config) (p : Prepared r out c)
    (profile : FixedISAProfile) (selected : Selection (requestShape r) c)
    (events : List (PhysicalCommit (requestShape r) c))
    (order : (events.map PhysicalCommit.logical).Perm (allCommits (requestShape r) c))
    (abi : LocalEventABI (BitVec 32) Word32 Word16 σ (fpOps profile))
    (raw : RawByteABI abi) (initial : σ)
    (input : PublicInput abi raw (requestShape r) (requestBuffers r out) (hasBias r) initial)
    (zero : Zeroed (fpOps profile) (requestShape r) c (requestBuffers r out) (abi.observe initial)) :
    let commits:=events.map PhysicalCommit.logical
    ∃ final,
      Quote.sourceGrid (Quote.nativeInteger profile (requestShape r) c selected)
        (requestShape r) c (hasBias r) (requestBuffers r out) Source.runtimeDivisor events
        (abi.observe initial)=some (abi.observe final) ∧
      (program r out c selected Source.runtimeDivisor commits).native abi initial=((),final) ∧
      OperandPointers base r ∧
      (∀ cell : MatrixCell (requestShape r),
        abi.observe final (outputAddress (requestShape r) (requestBuffers r out) cell)=
          .inr (.inr (logicalWordReference (fpOps profile) (requestShape r) c selected
            Source.runtimeDivisor (Schedule.ofPermutation (requestShape r) c commits order)
            (originalWords (fpOps profile) r (abi.observe initial)) cell))) ∧
      (∀ cell : MatrixCell (requestShape r), ∀ i : Fin 2,
        raw.bytes final (out,2*(cell.1.val*(requestShape r).N+cell.2.val)+i.val)=
          (logicalWordReference (fpOps profile) (requestShape r) c selected Source.runtimeDivisor
            (Schedule.ofPermutation (requestShape r) c commits order)
            (originalWords (fpOps profile) r (abi.observe initial)) cell).bits.extractLsb' (8*i.val) 8) ∧
      (∀ address : Address,¬OutputByte r out address → raw.bytes final address=raw.bytes initial address) := by
  obtain ⟨final,native,pointers,values,bytes,framing⟩:=fp_equiv_model base r laws out c p profile selected
    (events.map PhysicalCommit.logical) order abi raw initial input zero
  refine ⟨final,?_,native,pointers,values,bytes,framing⟩
  rw [Quote.sourceGrid_eq _ _ _ _ _ _ selected rfl]
  have bridge:=native_execution abi r out c selected Source.runtimeDivisor
    (events.map PhysicalCommit.logical) initial
  change (((program r out c selected Source.runtimeDivisor (events.map PhysicalCommit.logical)).native abi initial).1,
    abi.observe ((program r out c selected Source.runtimeDivisor (events.map PhysicalCommit.logical)).native abi initial).2)=_ at bridge
  rw [native,execution_eq] at bridge
  exact congrArg (fun p : Unit×Memory Word32 Word16=>some p.2) bridge.symm

#axiomsClean fp_equiv
#axiomsClean correct_model
#axiomsClean correct
#specNonCircular Direct.logicalInput avoiding [execute,program]
#specNonCircular Direct.logicalWordReference avoiding [execute,program]
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.TensorBridge
