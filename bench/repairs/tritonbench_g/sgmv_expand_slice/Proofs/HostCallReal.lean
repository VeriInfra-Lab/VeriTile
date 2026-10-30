import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.HostReal
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.HostControl
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.HostCopies

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.HostCallReal
open VeriTile.Triton FP.Structural HostSnapshots HostPrepared
noncomputable section

/-- Exact-Real logical observation and a separate parametric byte-footprint
observation. No injection or decoding of arbitrary reals into finite bytes is
postulated. Failed copies have no claimed completed logical output. -/
structure Result where
  physical : Host.Result
  logical : Option (Nat → Nat → ℝ)

/-- The byte payload is universally arbitrary: the byte-frame theorem is
independent of arithmetic and applies in particular to the actual native
payloads. Exact-Real values are observed only through the logical component. -/
def call (api : HostControl.API HostReal.Carrier) (copies : HostCopies.API)
    (rest : Algebra ℝ) (abi : SourceABI) (shape : Shape) (batches : Nat)
    (entry : (role : Host.Role) → View HostReal.Carrier role) (caller : List Window)
    (view : OutputView) (bytePayload : Nat → Nat → StoredWord view.width) (bytes : Bytes) : Result :=
  match HostControl.prepare api shape batches entry caller with
  | .error stage => ⟨Host.failed view stage [] bytes,none⟩
  | .ok setup =>
      match HostControl.privatePhase api.launch api.leave
        (SourceBody.install (RealBackend.algebra rest) RealBackend.dot) abi
        (PreparedMemory.arguments shape (strides setup.snapshots)) shape setup.segments
        (HostReal.initial abi shape batches setup.snapshots setup.lists) with
      | .error stage => ⟨Host.failed view stage [] bytes,none⟩
      | .ok final =>
          let copied := HostCopies.run copies shape view bytePayload 0 setup.segments
          ⟨HostCopies.result view bytes copied,
            match copied.outcome with
            | .failed _ => none
            | .returnedNone => some (HostReal.copy shape
                (SourceSuffix.readOutput abi (PreparedMemory.arguments shape (strides setup.snapshots)) final)
                setup.segments (HostReal.output shape entry))⟩

theorem successful (api : HostControl.API HostReal.Carrier) (copies : HostCopies.API)
    (rest : Algebra ℝ) (abi : SourceABI) (shape : Shape) (batches : Nat)
    (entry : (role : Host.Role) → View HostReal.Carrier role) (caller : List Window)
    (domain : Domain abi shape batches entry)
    (setup : HostControl.PreparedCall shape batches entry caller)
    (preparation : HostControl.prepare api shape batches entry caller = .ok setup)
    (launches : ∀ batch, api.launch batch = .ok ()) (leave : api.leave = .ok ())
    (copied : ∀ batch requested, copies.call batch requested = .ok ())
    (view : OutputView) (bytePayload : Nat → Nat → StoredWord view.width) (bytes : Bytes) :
    let result := call api copies rest abi shape batches entry caller view bytePayload bytes
    result.physical.outcome = .returnedNone ∧ result.physical.object = view.object ∧ result.physical.backing = view.backing ∧
    (∃ logical, result.logical = some logical ∧ ∀ r c, r < shape.outputRows → c < shape.outputColumns →
      logical r c = mathematicalResult shape (HostReal.input shape entry) (HostReal.weight shape entry)
        abi.addInputs setup.segments (HostReal.output shape entry) r c) ∧
    (∀ a b, ¬AllowedByte shape setup.segments view a b → result.physical.memory a b = bytes a b) := by
  obtain ⟨final,execution,logical⟩ := HostReal.caller_result rest abi shape batches entry caller domain
    setup.snapshots setup.lists setup.segments setup.validated
  have hphase := HostControl.privatePhase_complete api.launch
    (SourceBody.install (RealBackend.algebra rest) RealBackend.dot) abi
    (PreparedMemory.arguments shape (strides setup.snapshots)) shape setup.segments
    (HostReal.initial abi shape batches setup.snapshots setup.lists) final launches execution
  dsimp only
  unfold call
  simp only [preparation,leave,hphase,HostCopies.complete copies shape view bytePayload 0 setup.segments copied]
  refine ⟨rfl,rfl,rfl,⟨_,rfl,logical⟩,?_⟩
  intro a b outside
  exact actual_copyback_frame shape view setup.segments bytePayload bytes a b outside

theorem frame (api : HostControl.API HostReal.Carrier) (copies : HostCopies.API)
    (rest : Algebra ℝ) (abi : SourceABI) (shape : Shape) (batches : Nat)
    (entry : (role : Host.Role) → View HostReal.Carrier role) (caller : List Window)
    (segments : List Segment)
    (valid : validateSegments shape (HostMetadata.rawSegments batches
      (entry .starts).values (entry .lengths).values (entry .adapters).values) = some segments)
    (view : OutputView) (bytePayload : Nat → Nat → StoredWord view.width) (bytes : Bytes)
    (a b : Nat) (outside : ¬AllowedByte shape segments view a b) :
    (call api copies rest abi shape batches entry caller view bytePayload bytes).physical.memory a b = bytes a b := by
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
        (SourceBody.install (RealBackend.algebra rest) RealBackend.dot) abi
        (PreparedMemory.arguments shape (strides setup.snapshots)) shape setup.segments
        (HostReal.initial abi shape batches setup.snapshots setup.lists) with
      | error stage => rfl
      | ok final =>
          apply HostCopies.frame
          simpa only [same] using outside

#axiomsClean successful
#axiomsClean frame

theorem invalid_metadata_unchanged (api : HostControl.API HostReal.Carrier) (copies : HostCopies.API)
    (rest : Algebra ℝ) (abi : SourceABI) (shape : Shape) (batches : Nat)
    (entry : (role : Host.Role) → View HostReal.Carrier role) (caller : List Window)
    (invalid : validateSegments shape (HostMetadata.rawSegments batches
      (entry .starts).values (entry .lengths).values (entry .adapters).values) = none)
    (view : OutputView) (bytePayload : Nat → Nat → StoredWord view.width) (bytes : Bytes) :
    (call api copies rest abi shape batches entry caller view bytePayload bytes).physical.memory = bytes := by
  unfold call
  cases preparation : HostControl.prepare api shape batches entry caller with
  | error stage => rfl
  | ok setup =>
      have h := setup.validated
      rw [raw_entry,invalid] at h
      cases h

#axiomsClean invalid_metadata_unchanged
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.HostCallReal
