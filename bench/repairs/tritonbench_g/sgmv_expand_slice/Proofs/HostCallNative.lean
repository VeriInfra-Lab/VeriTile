import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.HostNative
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.HostControl
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.HostCopies

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.HostCallNative
open VeriTile.Triton FP.Structural HostSnapshots HostPrepared NativeBackend
noncomputable section

/-- The complete post-shape-check call: enter, six clones, three tolist calls,
full validation, ordered actual source launches, leave, active-only copies,
and None return. Every pre-copy failure returns the original caller bytes. -/
def call (abi : SourceABI) (api : HostControl.API (HostNative.Carrier abi)) (copies : HostCopies.API)
    (rest : Algebra Raw) (ops : Native.Primitives) (shape : Shape) (batches : Nat)
    (entry : (role : Host.Role) → View (HostNative.Carrier abi) role) (caller : List Window)
    (view : OutputView) (codec : CallerLogical.Codec (Native.Word (profile abi).output) view) (bytes : Bytes) : Host.Result :=
  match HostControl.prepare api shape batches entry caller with
  | .error stage => Host.failed view stage [] bytes
  | .ok setup =>
      match HostControl.privatePhase api.launch api.leave
        (SourceBody.install (algebra rest ops (profile abi)) (dot ops (weight abi.weight)))
        abi (PreparedMemory.arguments shape (strides setup.snapshots)) shape setup.segments
        (HostNative.initial abi shape batches setup.snapshots setup.lists) with
      | .error stage => Host.failed view stage [] bytes
      | .ok final => HostCopies.result view bytes
          (HostCopies.run copies shape view
            (fun r c => codec.encode (HostNative.read abi shape (strides setup.snapshots) final r c)) 0 setup.segments)

theorem successful (abi : SourceABI) (api : HostControl.API (HostNative.Carrier abi)) (copies : HostCopies.API)
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
    let result := call abi api copies rest ops shape batches entry caller view codec bytes
    result.outcome = .returnedNone ∧ result.object = view.object ∧ result.backing = view.backing ∧
    (∀ r c, r < shape.outputRows → c < shape.outputColumns →
      CallerLogical.read view codec result.memory r c =
        Native.result ops (profile abi) shape setup.segments (HostNative.input abi shape entry) (HostNative.output abi shape entry) r c) ∧
    (∀ a b, ¬AllowedByte shape setup.segments view a b → result.memory a b = bytes a b) := by
  obtain ⟨final,execution,logical,frame,object,backing,outcome⟩ :=
    HostNative.caller_result rest ops zeros abi shape batches entry caller domain setup.snapshots setup.lists
      setup.segments setup.validated view columns codec bytes entryRead
  have hphase := HostControl.privatePhase_complete api.launch
    (SourceBody.install (algebra rest ops (profile abi)) (dot ops (weight abi.weight))) abi
    (PreparedMemory.arguments shape (strides setup.snapshots)) shape setup.segments
    (HostNative.initial abi shape batches setup.snapshots setup.lists) final launches execution
  have hcall : call abi api copies rest ops shape batches entry caller view codec bytes =
      Host.returned view (copyBackWrites shape view setup.segments
        (fun r c => codec.encode (HostNative.read abi shape (strides setup.snapshots) final r c))) bytes := by
    unfold call
    simp only [preparation,leave,hphase,HostCopies.complete copies shape view _ 0 setup.segments copied]
    rfl
  dsimp only
  rw [hcall]
  exact ⟨outcome,object,backing,logical,frame⟩

/-- Every outcome has the byte frame, including any clone/tolist/launch/leave
failure and a partially committed local copy. The reference segments come
from the entry metadata, independently of the returned private allocations. -/
theorem frame (abi : SourceABI) (api : HostControl.API (HostNative.Carrier abi)) (copies : HostCopies.API)
    (rest : Algebra Raw) (ops : Native.Primitives) (shape : Shape) (batches : Nat)
    (entry : (role : Host.Role) → View (HostNative.Carrier abi) role) (caller : List Window)
    (segments : List Segment)
    (valid : validateSegments shape (HostMetadata.rawSegments batches
      (entry .starts).values (entry .lengths).values (entry .adapters).values) = some segments)
    (view : OutputView) (codec : CallerLogical.Codec (Native.Word (profile abi).output) view) (bytes : Bytes)
    (a b : Nat) (outside : ¬AllowedByte shape segments view a b) :
    (call abi api copies rest ops shape batches entry caller view codec bytes).memory a b = bytes a b := by
  unfold call
  cases preparation : HostControl.prepare api shape batches entry caller with
  | error stage => rfl
  | ok setup =>
      simp only
      have same : setup.segments = segments := by
        have h := setup.validated
        rw [raw_entry,valid] at h
        exact Option.some.inj h.symm
      cases phase : HostControl.privatePhase api.launch api.leave
        (SourceBody.install (algebra rest ops (profile abi)) (dot ops (weight abi.weight))) abi
        (PreparedMemory.arguments shape (strides setup.snapshots)) shape setup.segments
        (HostNative.initial abi shape batches setup.snapshots setup.lists) with
      | error stage => rfl
      | ok final =>
          apply HostCopies.frame
          simpa only [same] using outside

#axiomsClean successful
#axiomsClean frame

theorem invalid_metadata_unchanged (abi : SourceABI) (api : HostControl.API (HostNative.Carrier abi))
    (copies : HostCopies.API) (rest : Algebra Raw) (ops : Native.Primitives) (shape : Shape) (batches : Nat)
    (entry : (role : Host.Role) → View (HostNative.Carrier abi) role) (caller : List Window)
    (invalid : validateSegments shape (HostMetadata.rawSegments batches
      (entry .starts).values (entry .lengths).values (entry .adapters).values) = none)
    (view : OutputView) (codec : CallerLogical.Codec (Native.Word (profile abi).output) view) (bytes : Bytes) :
    (call abi api copies rest ops shape batches entry caller view codec bytes).memory = bytes := by
  unfold call
  cases preparation : HostControl.prepare api shape batches entry caller with
  | error stage => rfl
  | ok setup =>
      have h := setup.validated
      rw [raw_entry,invalid] at h
      cases h

#axiomsClean invalid_metadata_unchanged
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.HostCallNative
