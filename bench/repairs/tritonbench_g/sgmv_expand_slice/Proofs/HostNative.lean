import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.HostPrepared
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.NativePrepared
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.CallerLogical

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.HostNative
open VeriTile.Triton FP.Structural HostSnapshots HostPrepared NativeBackend
noncomputable section

def Carrier (abi : SourceABI) : Host.Role → Type
  | .input => Native.Word (profile abi).input
  | .weight => Native.Word (profile abi).weight.format
  | .output => Native.Word (profile abi).output
  | _ => Int

instance (abi : SourceABI) (role : Host.Role) : Inhabited (Carrier abi role) := by
  cases role <;> unfold Carrier <;> infer_instance

def input (abi : SourceABI) (shape : Shape) (entry : (role : Host.Role) → View (Carrier abi) role) :
    Native.Inputs (profile abi) :=
  ⟨matrix shape.inner (entry .input).values,HostPrepared.weight shape.columns shape.inner (entry .weight).values⟩
def output (abi : SourceABI) (shape : Shape) (entry : (role : Host.Role) → View (Carrier abi) role) :=
  matrix shape.outputColumns (entry .output).values

def cloneInput (abi : SourceABI) (shape : Shape) {entry : (role : Host.Role) → View (Carrier abi) role}
    {caller : List Window} (p : Prepared entry caller) : Native.Inputs (profile abi) :=
  ⟨matrix shape.inner p.x.values,HostPrepared.weight shape.columns shape.inner p.w.values⟩
def cloneOutput (abi : SourceABI) (shape : Shape) {entry : (role : Host.Role) → View (Carrier abi) role}
    {caller : List Window} (p : Prepared entry caller) := matrix shape.outputColumns p.y.values

theorem clone_input_entry (abi : SourceABI) (shape : Shape) {entry : (role : Host.Role) → View (Carrier abi) role}
    {caller : List Window} (p : Prepared entry caller) : cloneInput abi shape p = input abi shape entry := by
  simp only [cloneInput,input,p.x.preserved,p.w.preserved]
theorem clone_output_entry (abi : SourceABI) (shape : Shape) {entry : (role : Host.Role) → View (Carrier abi) role}
    {caller : List Window} (p : Prepared entry caller) : cloneOutput abi shape p = output abi shape entry := by
  simp only [cloneOutput,output,p.y.preserved]

def initial (abi : SourceABI) (shape : Shape) (batches : Nat)
    {entry : (role : Host.Role) → View (Carrier abi) role} {caller : List Window}
    (p : Prepared entry caller) (lists : Lists p) : State Raw :=
  prepared abi shape (cloneInput abi shape p) (cloneOutput abi shape p)
    (raw batches lists) (lists.lengths.val.drop batches)

def read (abi : SourceABI) (shape : Shape) (strides : PreparedMemory.Strides) (state : State Raw)
    (r c : Nat) : Native.Word (profile abi).output :=
  unpack (scalar abi.output) (SourceSuffix.readOutput abi (PreparedMemory.arguments shape strides) state r c)

/-- The same returned six snapshots determine the source memory, source
strides, validation records, and independent entry-snapshot native target. -/
theorem source_result (rest : Algebra Raw) (ops : Native.Primitives) (zeros : ZeroConversions ops)
    (abi : SourceABI) (shape : Shape) (batches : Nat)
    (entry : (role : Host.Role) → View (Carrier abi) role) (caller : List Window)
    (domain : Domain abi shape batches entry) (p : Prepared entry caller) (lists : Lists p)
    (segments : List Segment) (valid : validateSegments shape (raw batches lists) = some segments) :
    ∃ final, runSourceJobs (SourceBody.install (algebra rest ops (profile abi)) (dot ops (NativeBackend.weight abi.weight)))
        abi (PreparedMemory.arguments shape (strides p)) (jobs shape.columns segments)
        (initial abi shape batches p lists) = some final ∧
      (∀ r c, r < shape.outputRows → c < shape.outputColumns →
        read abi shape (strides p) final r c =
          Native.result ops (profile abi) shape segments (input abi shape entry) (output abi shape entry) r c) ∧
      SourceGrid.ReadFrame (PreparedMemory.arguments shape (strides p)) (initial abi shape batches p lists) final := by
  obtain ⟨hx,hw,hy⟩ := layouts abi shape batches entry caller domain p
  obtain ⟨final,run,values,frame⟩ := prepared_result rest ops zeros abi shape (strides p)
    (cloneInput abi shape p) (cloneOutput abi shape p) (raw batches lists) (lists.lengths.val.drop batches)
    segments valid hx hw hy domain.capacity
  refine ⟨final,run,?_,frame⟩
  intro r c hr hc
  unfold read
  rw [values r c hr hc,unpack_pack,clone_input_entry,clone_output_entry]

theorem target_frame (ops : Native.Primitives) (p : Native.Profile) (shape : Shape)
    (segments : List Segment) (input : Native.Inputs p) (entry : Nat → Nat → Native.Word p.output)
    (r c : Nat) (outside : ∀ s ∈ segments, ¬ActiveCell shape s r c) :
    Native.result ops p shape segments input entry r c = entry r c := by
  induction segments generalizing entry with
  | nil => rfl
  | cons s ss ih =>
      change Native.result ops p shape ss input (Native.segmentStep ops p shape s input entry) r c = entry r c
      rw [ih _ (by intro t ht; exact outside t (List.mem_cons_of_mem _ ht))]
      exact if_neg (outside s (by simp))

/-- Readback copies the actual source final state's words. The entry-view
relation is local logical interpretation; it assumes no final output law. -/
theorem caller_result (rest : Algebra Raw) (ops : Native.Primitives) (zeros : ZeroConversions ops)
    (abi : SourceABI) (shape : Shape) (batches : Nat)
    (entry : (role : Host.Role) → View (Carrier abi) role) (caller : List Window)
    (domain : Domain abi shape batches entry) (p : Prepared entry caller) (lists : Lists p)
    (segments : List Segment) (valid : validateSegments shape (raw batches lists) = some segments)
    (view : OutputView) (columns : view.columns = shape.outputColumns)
    (codec : CallerLogical.Codec (Native.Word (profile abi).output) view) (bytes : Bytes)
    (entryRead : ∀ r c, r < shape.outputRows → c < shape.outputColumns →
      CallerLogical.read view codec bytes r c = output abi shape entry r c) :
    ∃ final, runSourceJobs (SourceBody.install (algebra rest ops (profile abi)) (dot ops (NativeBackend.weight abi.weight)))
        abi (PreparedMemory.arguments shape (strides p)) (jobs shape.columns segments)
        (initial abi shape batches p lists) = some final ∧
      let result := Host.returned view
        (copyBackWrites shape view segments (fun r c => codec.encode (read abi shape (strides p) final r c))) bytes
      (∀ r c, r < shape.outputRows → c < shape.outputColumns →
        CallerLogical.read view codec result.memory r c =
          Native.result ops (profile abi) shape segments (input abi shape entry) (output abi shape entry) r c) ∧
      (∀ a b, ¬AllowedByte shape segments view a b → result.memory a b = bytes a b) ∧
      result.object = view.object ∧ result.backing = view.backing ∧ result.outcome = .returnedNone := by
  obtain ⟨final,run,values,_⟩ := source_result rest ops zeros abi shape batches entry caller domain p lists segments valid
  refine ⟨final,run,?_,?_,rfl,rfl,rfl⟩
  · intro r c hr hc
    by_cases active : ∃ s ∈ segments, ActiveCell shape s r c
    · obtain ⟨s,hs,ha⟩ := active
      rw [CallerLogical.active_read shape view codec segments (by simpa [columns] using domain.capacity)
        _ bytes s hs r c ha]
      exact values r c hr hc
    · have outside : ∀ s ∈ segments, ¬ActiveCell shape s r c := by simpa using active
      rw [CallerLogical.inactive_read shape view codec segments (by simpa [columns] using domain.capacity)
        _ bytes r c (by simpa [columns] using hc) outside,
        entryRead r c hr hc,target_frame ops (profile abi) shape segments _ _ r c outside]
  · intro a b outside
    exact actual_copyback_frame shape view segments _ bytes a b outside

#axiomsClean clone_input_entry
#axiomsClean clone_output_entry
#axiomsClean source_result
#axiomsClean target_frame
#axiomsClean caller_result
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.HostNative
