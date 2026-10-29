import bench.repairs.tritonbench_g.sgmv_expand_slice.Kernels
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.NativeGraph
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.CopyBack
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.Metadata
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.Integer
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.Schedule
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.RangeReuse
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.HostCallNative
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.HostEntry
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.WordCodec
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.PreparedBounds
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.LegacyFPEquiv
open VeriTile.Triton FP.Structural HostSnapshots HostPrepared NativeBackend

/-! Whole-call source-to-native recurrence and caller-word readback.
Default input/output are FP32; weights remain FP16/BF16. Every FP16/BF16/FP32
input/output combination and all eight metadata-width combinations are retained.
The independent native recurrence preserves casts, padded zeros, K-block order,
eight m16n8k16 incoming-accumulator operations, final weight cast, mixed-format
promotion, observed add operand order and output conversion. All raw words,
including NaNs and signed zeros, remain in the domain.

`whole_call` executes the actual typed source from the same six finite snapshots,
validates all metadata before launches, serializes overlapping batches, exits
the context before copyback, and reads the actual final private state into the
supplied output view. Its conclusions include every caller logical word, output
object/backing identity, None return and the full caller-byte frame. No kernel
success, final-output or whole-memory law is a numerical premise.

Deferred numerical ports are exactly selected-format conversion, add, one
m16n8k16 instruction, and the two explicit FP32-zero-to-low conversion laws.
Ordinary local tensor-view/clone/copy/allocator/compiler/runtime contracts are
separate boundaries. The explicit WordCodec.forABI is inhabited and matches the
output width; its bit identity is not a claim of tested GPU conformance.
No GPU execution was available. PreparedBounds connects the actual clone strides
to i64 pointer and dimension/grid bounds from ordinary allocated storage. -/

abbrev target := Native.result
abbrev numericalPorts := Native.PrimitiveAgreement

/-- A local typed storage read; low storage uses exactly sixteen raw bits. -/
def readWord (format : Native.Format) (cell : VeriTile.Triton.FP.Structural.Cell (BitVec 32)) :
    Native.Word format := match format with
  | .half => (cell.read .fp16).setWidth 16
  | .bfloat => (cell.read .bf16).setWidth 16
  | .single => cell.read .real

/-- Source-to-native result contract. The concrete prepared instance is proved below. -/
def SourceObligation (backend : VeriTile.Triton.FP.Structural.Algebra (BitVec 32))
    (abi : SourceABI) (args : KernelArgs) (p : Native.Profile) (shape : Shape)
    (segments : List Segment) (ops : Native.Primitives) (input : Native.Inputs p)
    (entry : Nat → Nat → Native.Word p.output)
    (initial : VeriTile.Triton.FP.Structural.State (BitVec 32)) : Prop :=
  ∃ final, runSourceJobs backend abi args (jobs shape.columns segments) initial = some final ∧
    ∀ r c, r < shape.outputRows → c < shape.outputColumns →
      readWord p.output (final.mem args.output (r*args.outputRowStride+c*args.outputColumnStride)) =
        target ops p shape segments input entry r c

/-- Pure congruence from exact local primitive agreements; this theorem does
not claim that the actual source runner or native device satisfies the target. -/
theorem local_primitives_compose (left right : Native.Primitives)
    (h : numericalPorts left right) (p : Native.Profile) (shape : Shape)
    (segments : List Segment) (input : Native.Inputs p)
    (entry : Nat → Nat → Native.Word p.output) :
    target left p shape segments input entry = target right p shape segments input entry :=
  Native.result_of_local_primitives left right h p shape segments input entry


theorem readWord_output (format : ScalarFormat)
    (cell : VeriTile.Triton.FP.Structural.Cell (BitVec 32)) :
    readWord (NativeBackend.scalar format) cell =
      NativeBackend.unpack (NativeBackend.scalar format)
        (VeriTile.Triton.FP.Structural.toFloat (SourceSuffix.storageType format)
          (cell.read (SourceSuffix.storageType format).toTileDType)) := by
  cases format <;> rfl

/-- Every concrete prepared source job succeeds and every bounded output word
matches the independent native recurrence. -/
theorem prepared_source_obligation
    (rest : VeriTile.Triton.FP.Structural.Algebra NativeBackend.Raw)
    (ops : Native.Primitives) (zeros : NativeBackend.ZeroConversions ops)
    (abi : SourceABI) (shape : Shape) (strides : PreparedMemory.Strides)
    (input : Native.Inputs (NativeBackend.profile abi))
    (entry : Nat → Nat → Native.Word (NativeBackend.scalar abi.output))
    (raw : List RawSegment) (extraLengths : List Int) (segments : List Segment)
    (validated : validateSegments shape raw = some segments)
    (xlayout : GridStride.Contiguous shape.tokens shape.inner strides.inputRow strides.inputInner)
    (wlayout : SnapshotLayout.WeightContiguous shape.adapters shape.columns shape.inner
      strides.weightAdapter strides.weightColumn strides.weightInner)
    (ylayout : GridStride.Contiguous shape.outputRows shape.outputColumns strides.outputRow strides.outputColumn)
    (capacity : shape.slice+shape.columns ≤ shape.outputColumns) :
    SourceObligation
      (SourceBody.install (NativeBackend.algebra rest ops (NativeBackend.profile abi))
        (NativeBackend.dot ops (NativeBackend.weight abi.weight))) abi (PreparedMemory.arguments shape strides)
      (NativeBackend.profile abi) shape segments ops input entry
      (NativeBackend.prepared abi shape input entry raw extraLengths) := by
  obtain ⟨final,execution,values,_⟩ := NativeBackend.prepared_result rest ops zeros abi shape strides
    input entry raw extraLengths segments validated xlayout wlayout ylayout capacity
  refine ⟨final,execution,?_⟩
  intro r c hr hc
  change readWord (NativeBackend.scalar abi.output)
    (final.mem (PreparedMemory.arguments shape strides).output
      (r*strides.outputRow+c*strides.outputColumn)) = _
  rw [readWord_output]
  change NativeBackend.unpack (NativeBackend.scalar abi.output)
    (SourceSuffix.readOutput abi (PreparedMemory.arguments shape strides) final r c) = _
  rw [values r c hr hc,NativeBackend.unpack_pack]

theorem whole_call (abi : SourceABI) (api : HostControl.API (HostNative.Carrier abi)) (copies : HostCopies.API)
    (rest : Algebra Raw) (ops : Native.Primitives) (zeros : ZeroConversions ops)
    (shape : Shape) (batches : Nat)
    (entry : (role : Host.Role) → View (HostNative.Carrier abi) role) (caller : List Window)
    (domain : Domain abi shape batches entry)
    (setup : HostControl.PreparedCall shape batches entry caller)
    (preparation : HostControl.prepare api shape batches entry caller = .ok setup)
    (launches : ∀ batch, api.launch batch = .ok ()) (leave : api.leave = .ok ())
    (copied : ∀ batch requested, copies.call batch requested = .ok ())
    (view : OutputView) (columns : view.columns = shape.outputColumns)
    (codec : CallerLogical.Codec (Native.Word (profile abi).output) view) (bytes : Bytes)
    (entryRead : ∀ r c, r < shape.outputRows → c < shape.outputColumns →
      CallerLogical.read view codec bytes r c = HostNative.output abi shape entry r c) :
    let result := HostCallNative.call abi api copies rest ops shape batches entry caller view codec bytes
    result.outcome = .returnedNone ∧ result.object = view.object ∧ result.backing = view.backing ∧
    (∀ r c, r < shape.outputRows → c < shape.outputColumns →
      CallerLogical.read view codec result.memory r c =
        Native.result ops (profile abi) shape setup.segments (HostNative.input abi shape entry) (HostNative.output abi shape entry) r c) ∧
    (∀ a b, ¬AllowedByte shape setup.segments view a b → result.memory a b = bytes a b) := by
  exact HostCallNative.successful abi api copies rest ops zeros shape batches entry caller domain setup preparation launches leave copied view columns codec bytes entryRead
theorem whole_call_frame (abi : SourceABI) (api : HostControl.API (HostNative.Carrier abi)) (copies : HostCopies.API)
    (rest : Algebra Raw) (ops : Native.Primitives) (shape : Shape) (batches : Nat)
    (entry : (role : Host.Role) → View (HostNative.Carrier abi) role) (caller : List Window)
    (segments : List Segment)
    (valid : validateSegments shape (HostMetadata.rawSegments batches
      (entry .starts).values (entry .lengths).values (entry .adapters).values) = some segments)
    (view : OutputView) (codec : CallerLogical.Codec (Native.Word (profile abi).output) view) (bytes : Bytes)
    (a b : Nat) (outside : ¬AllowedByte shape segments view a b) :
    (HostCallNative.call abi api copies rest ops shape batches entry caller view codec bytes).memory a b = bytes a b := by
  exact HostCallNative.frame abi api copies rest ops shape batches entry caller segments valid view codec bytes a b outside
/-- Concrete public binding: every role is read from the same caller bytes,
all caller windows participate in clone freshness, and the output ABI chooses
its two/four-byte codec. The only read premise is the local supplied-view law. -/
theorem whole_call_from_entry_bytes (abi : SourceABI)
    (api : HostControl.API (HostNative.Carrier abi)) (copies : HostCopies.API)
    (rest : Algebra Raw) (ops : Native.Primitives) (zeros : ZeroConversions ops)
    (shape : Shape) (batches : Nat) (bytes : Bytes)
    (views : (role : Host.Role) → HostEntry.Descriptor (HostNative.Carrier abi) role)
    (domain : Domain abi shape batches (HostEntry.entry views bytes))
    (setup : HostControl.PreparedCall shape batches (HostEntry.entry views bytes) (HostEntry.callerWindows views))
    (preparation : HostControl.prepare api shape batches (HostEntry.entry views bytes) (HostEntry.callerWindows views) = .ok setup)
    (launches : ∀ batch, api.launch batch = .ok ()) (leave : api.leave = .ok ())
    (copied : ∀ batch requested, copies.call batch requested = .ok ())
    (view : OutputView) (rows : view.rows = shape.outputRows) (columns : view.columns = shape.outputColumns)
    (width : view.width = scalarBytes abi.output)
    (outputWindow : ∀ r c b, r < view.rows → c < view.columns → b < view.width →
      (views .output).window.contains view.backing (view.byte r c b))
    (entryRead : ∀ r c, r < shape.outputRows → c < shape.outputColumns →
      CallerLogical.read view (WordCodec.forABI abi view width) bytes r c =
        HostNative.output abi shape (HostEntry.entry views bytes) r c) :
    let result := HostCallNative.call abi api copies rest ops shape batches
      (HostEntry.entry views bytes) (HostEntry.callerWindows views) view (WordCodec.forABI abi view width) bytes
    result.outcome = .returnedNone ∧ result.object = view.object ∧ result.backing = view.backing ∧
    (∀ r c, r < shape.outputRows → c < shape.outputColumns →
      CallerLogical.read view (WordCodec.forABI abi view width) result.memory r c =
        Native.result ops (profile abi) shape setup.segments (HostNative.input abi shape (HostEntry.entry views bytes))
          (HostNative.output abi shape (HostEntry.entry views bytes)) r c) ∧
    (∀ a b, ¬AllowedByte shape setup.segments view a b → result.memory a b = bytes a b) := by
  exact whole_call abi api copies rest ops zeros shape batches (HostEntry.entry views bytes)
    (HostEntry.callerWindows views) domain setup preparation launches leave copied view columns
    (WordCodec.forABI abi view width) bytes entryRead

#axiomsClean whole_call_from_entry_bytes

abbrev invalid_metadata_unchanged := @HostCallNative.invalid_metadata_unchanged
abbrev same_call_entry := @HostEntry.same_call
abbrev actual_pointer_words := @PreparedBounds.source_pointer_words
abbrev actual_dimensions_and_pid := @PreparedBounds.dimensions_and_pid

#axiomsClean readWord_output
#axiomsClean prepared_source_obligation
#axiomsClean whole_call
#axiomsClean whole_call_frame
#axiomsClean invalid_metadata_unchanged
#axiomsClean same_call_entry
#axiomsClean actual_pointer_words
#axiomsClean actual_dimensions_and_pid
#axiomsClean WordCodec.forABI

#axiomsClean local_primitives_compose
#axiomsClean authorized_copy_frame
#axiomsClean partial_copy_frame
#axiomsClean source_jobs_append
#axiomsClean actual_copyback_readback
#axiomsClean actual_copyback_frame
#axiomsClean actual_failed_copy_frame
#axiomsClean validated_segments_fit
#axiomsClean Integer.matrix_pointer_exact
#axiomsClean Integer.weight_pointer_exact
#axiomsClean Integer.byte_address_exact
#axiomsClean RangeReuse.range_values_int

end VeriTile.Bench.Repairs.SgmvExpandSlice.LegacyFPEquiv
