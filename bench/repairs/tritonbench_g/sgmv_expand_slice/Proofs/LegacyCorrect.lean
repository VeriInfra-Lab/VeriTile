import bench.repairs.tritonbench_g.sgmv_expand_slice.Kernels
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.Mathematics
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.CopyBack
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.Metadata
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.Integer
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.Schedule
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.HostCallReal
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.HostSnapshotWitness
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.HostEntry
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.PreparedBounds
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.LegacyCorrect
open VeriTile.Triton FP.Structural HostSnapshots HostPrepared

/-! The main semantics is the actual typed source evaluated with exact Real
arithmetic and identity floating casts. `typed_source` proves every job of that
source; `whole_call` composes the same six finite entry snapshots, all metadata
validation, ordered overlaps, successful local runtime calls, context exit,
and logical slice copies. Kernel success and final-output equality are proved,
not hypotheses. Native arithmetic is not imported by this Real proof family.

The byte component is a separate, universally quantified footprint observer;
it never pretends to encode arbitrary reals injectively into finite bytes.
Native caller-byte values are proved in FPEquiv. All outcomes retain the byte
frame, including pre-copy exceptions and partial-copy failures. Caller aliases
are permitted, and HostEntry binds all six reads to one entry byte state.

Domain: the original accepted dtype/shape/contiguity conditions, independent
input/output row counts, arbitrary ordered segment overlaps, both ADD_INPUTS
branches, all metadata widths, empty/sentinel/zero-column skips and K tails.
Ordinary clone/view/copy/allocator/runtime contracts are local API boundaries.
PreparedBounds derives i64 pointer/dimension/grid bounds from allocated storage.

Historical alternative: SourceObligation below still names the distinct erased
BlockState interpreter. It is retained unproved for audit continuity and is not
a dependency of the new typed-source main theorem. No equivalence between the
two interpreters is asserted or assumed. -/

noncomputable abbrev target := mathematicalResult

/-- Historical, unused alternative interpreter obligation. No proof is claimed. -/
def SourceObligation (abi : SourceABI) (args : KernelArgs) (shape : Shape)
    (segments : List Segment) (X : Nat → Nat → ℝ) (W : Nat → Nat → Nat → ℝ)
    (entry : Nat → Nat → ℝ) (initial : BlockState) : Prop :=
  ∃ final, runRealJobs abi args (jobs shape.columns segments) initial = some final ∧
    ∀ r c, r < shape.outputRows → c < shape.outputColumns →
      final.readMem args.output (r*args.outputRowStride+c*args.outputColumnStride) =
        target shape X W abi.addInputs segments entry r c

theorem target_frame (shape : Shape) (segments : List Segment)
    (X : Nat → Nat → ℝ) (W : Nat → Nat → Nat → ℝ) (entry : Nat → Nat → ℝ)
    (add : Bool) (r c : Nat) (outside : ∀ s ∈ segments, ¬ActiveCell shape s r c) :
    target shape X W add segments entry r c = entry r c :=
  mathematicalResult_frame shape X W add segments entry r c outside


/-- Actual typed-source execution against the independent exact matrix formula. -/
theorem typed_source (rest : Algebra ℝ) (abi : SourceABI) (shape : Shape) (batches : Nat)
    (entry : (role : Host.Role) → View HostReal.Carrier role) (caller : List Window)
    (domain : Domain abi shape batches entry) (p : Prepared entry caller) (lists : Lists p)
    (segments : List Segment) (valid : validateSegments shape (raw batches lists) = some segments) :
    ∃ final, runSourceJobs (SourceBody.install (RealBackend.algebra rest) RealBackend.dot)
        abi (PreparedMemory.arguments shape (strides p)) (jobs shape.columns segments)
        (HostReal.initial abi shape batches p lists) = some final ∧
      SourceSuffix.OutputMemory abi (PreparedMemory.arguments shape (strides p)) shape final
        (mathematicalResult shape (HostReal.input shape entry) (HostReal.weight shape entry) abi.addInputs segments (HostReal.output shape entry)) ∧
      SourceGrid.ReadFrame (PreparedMemory.arguments shape (strides p)) (HostReal.initial abi shape batches p lists) final := by
  exact HostReal.source_result rest abi shape batches entry caller domain p lists segments valid

/-- Full successful API call. Only ordinary local API availability is assumed;
source execution and every returned logical output are conclusions. -/
theorem whole_call (api : HostControl.API HostReal.Carrier) (copies : HostCopies.API)
    (rest : Algebra ℝ) (abi : SourceABI) (shape : Shape) (batches : Nat)
    (entry : (role : Host.Role) → View HostReal.Carrier role) (caller : List Window)
    (domain : Domain abi shape batches entry)
    (setup : HostControl.PreparedCall shape batches entry caller)
    (preparation : HostControl.prepare api shape batches entry caller = .ok setup)
    (launches : ∀ batch, api.launch batch = .ok ()) (leave : api.leave = .ok ())
    (copied : ∀ batch requested, copies.call batch requested = .ok ())
    (view : OutputView) (bytePayload : Nat → Nat → StoredWord view.width) (bytes : Bytes) :
    let result := HostCallReal.call api copies rest abi shape batches entry caller view bytePayload bytes
    result.physical.outcome = .returnedNone ∧ result.physical.object = view.object ∧ result.physical.backing = view.backing ∧
    (∃ logical, result.logical = some logical ∧ ∀ r c, r < shape.outputRows → c < shape.outputColumns →
      logical r c = mathematicalResult shape (HostReal.input shape entry) (HostReal.weight shape entry)
        abi.addInputs setup.segments (HostReal.output shape entry) r c) ∧
    (∀ a b, ¬AllowedByte shape setup.segments view a b → result.physical.memory a b = bytes a b) := by
  exact HostCallReal.successful api copies rest abi shape batches entry caller domain setup preparation launches leave copied view bytePayload bytes

/-- All API outcomes preserve every caller byte outside active destinations. -/
theorem whole_call_frame (api : HostControl.API HostReal.Carrier) (copies : HostCopies.API)
    (rest : Algebra ℝ) (abi : SourceABI) (shape : Shape) (batches : Nat)
    (entry : (role : Host.Role) → View HostReal.Carrier role) (caller : List Window)
    (segments : List Segment)
    (valid : validateSegments shape (HostMetadata.rawSegments batches
      (entry .starts).values (entry .lengths).values (entry .adapters).values) = some segments)
    (view : OutputView) (bytePayload : Nat → Nat → StoredWord view.width) (bytes : Bytes)
    (a b : Nat) (outside : ¬AllowedByte shape segments view a b) :
    (HostCallReal.call api copies rest abi shape batches entry caller view bytePayload bytes).physical.memory a b = bytes a b := by
  exact HostCallReal.frame api copies rest abi shape batches entry caller segments valid view bytePayload bytes a b outside

/-- All six local view reads use one entry byte state. The exact Real logical
observer remains separate from the universally arbitrary byte-copy payload. -/
theorem whole_call_from_entry_bytes (api : HostControl.API HostReal.Carrier) (copies : HostCopies.API)
    (rest : Algebra ℝ) (abi : SourceABI) (shape : Shape) (batches : Nat) (bytes : Bytes)
    (views : (role : Host.Role) → HostEntry.Descriptor HostReal.Carrier role)
    (domain : Domain abi shape batches (HostEntry.entry views bytes))
    (setup : HostControl.PreparedCall shape batches (HostEntry.entry views bytes) (HostEntry.callerWindows views))
    (preparation : HostControl.prepare api shape batches (HostEntry.entry views bytes) (HostEntry.callerWindows views) = .ok setup)
    (launches : ∀ batch, api.launch batch = .ok ()) (leave : api.leave = .ok ())
    (copied : ∀ batch requested, copies.call batch requested = .ok ())
    (view : OutputView) (bytePayload : Nat → Nat → StoredWord view.width) :
    let result := HostCallReal.call api copies rest abi shape batches (HostEntry.entry views bytes)
      (HostEntry.callerWindows views) view bytePayload bytes
    result.physical.outcome = .returnedNone ∧ result.physical.object = view.object ∧ result.physical.backing = view.backing ∧
    (∃ logical, result.logical = some logical ∧ ∀ r c, r < shape.outputRows → c < shape.outputColumns →
      logical r c = mathematicalResult shape (HostReal.input shape (HostEntry.entry views bytes))
        (HostReal.weight shape (HostEntry.entry views bytes)) abi.addInputs setup.segments
        (HostReal.output shape (HostEntry.entry views bytes)) r c) ∧
    (∀ a b, ¬AllowedByte shape setup.segments view a b → result.physical.memory a b = bytes a b) := by
  exact whole_call api copies rest abi shape batches (HostEntry.entry views bytes) (HostEntry.callerWindows views)
    domain setup preparation launches leave copied view bytePayload bytes

#axiomsClean whole_call_from_entry_bytes

abbrev invalid_metadata_unchanged := @HostCallReal.invalid_metadata_unchanged
abbrev same_call_entry := @HostEntry.same_call
abbrev actual_pointer_words := @PreparedBounds.source_pointer_words
abbrev actual_dimensions_and_pid := @PreparedBounds.dimensions_and_pid

#axiomsClean typed_source
#axiomsClean whole_call
#axiomsClean whole_call_frame
#axiomsClean invalid_metadata_unchanged
#axiomsClean same_call_entry
#axiomsClean actual_pointer_words
#axiomsClean actual_dimensions_and_pid

#axiomsClean target_frame
#axiomsClean clamped_output_offset
#axiomsClean matrix_intermediates_i64
#axiomsClean weight_address_i64
#axiomsClean one_dimensional_coordinate
#axiomsClean Host.returned_identity
#axiomsClean Host.before_copy_failure
#axiomsClean Host.failed_copy_frame
#axiomsClean Host.nonempty_copy_witness
#axiomsClean actual_copyback_readback
#axiomsClean actual_copyback_frame
#axiomsClean actual_failed_copy_frame
#axiomsClean validated_segments_fit
#axiomsClean metadata32_widen
#axiomsClean Integer.matrix_pointer_exact
#axiomsClean Integer.weight_pointer_exact
#axiomsClean Integer.byte_address_exact
#axiomsClean Integer.accepted_pid_i64

end VeriTile.Bench.Repairs.SgmvExpandSlice.LegacyCorrect
