import bench.repairs.tritonbench_g.rms_matmul_rbe.Proofs.Program
import bench.repairs.tritonbench_g.rms_matmul_rbe.Proofs.CheckedNative
namespace VeriTile.Bench.Repairs.RmsMatmulRbe
noncomputable section

theorem fp_callee (a : Ops F) (p : Parameters F) (x : RowInput F)
    (plans : Nat → Nat → DotPlan) (reduction : Reduction) (rotary : Bool) (row n : Nat) :
    rms_matmul_rbe a p x plans reduction rotary row n=referenceValue a p x plans reduction rotary row n :=
  value_reference a p x plans reduction rotary row n

theorem fp_qkv
    (actual reference : StorageOps (BitVec 32) (BitVec 16))
    (leaves : NumericLeaves actual reference)
    (abi : LocalEventABI (BitVec 32) (BitVec 16) σ actual)
    (p : Parameters (BitVec 32)) (b : Buffers) (sep : b.Separate)
    (injective : Function.Injective (outputAddress b : Cell s → Address))
    (binding : CompilerBinding (BitVec 32) (BitVec 16) s p b) (state : σ) :
    (∀ x : Cell s,abi.observe (boundExecute actual abi binding state).2 (outputAddress b x)=
      wordTarget reference p b selectedPlans selectedReductions (abi.observe state) x) ∧
    (∀ q,(∀ x : Cell s,q≠outputAddress b x) →
      abi.observe (boundExecute actual abi binding state).2 q=abi.observe state q) := by
  exact bound_native_complete actual reference leaves abi p b sep injective binding state
theorem fp_equiv
    (actual reference : StorageOps (BitVec 32) (BitVec 16))
    (leaves : NumericLeaves actual reference)
    (abi : LocalEventABI (BitVec 32) (BitVec 16) σ actual)
    (p : Parameters (BitVec 32)) (b : Buffers) (domain : HostDomain s p b)
    (binding : CompilerBinding (BitVec 32) (BitVec 16) s p b) (state : σ)
    (caller : Memory (BitVec 32) (BitVec 16))
    (allocationFrame : ∀ q,q.1≠(b.output 0).storage → abi.observe state q=caller q) :
    (∀ x : Cell s,abi.observe ((hostCode s binding).native abi state).2 (outputAddress b x)=
      wordTarget reference p b selectedPlans selectedReductions (abi.observe state) x) ∧
    (∀ q,(∀ x : Cell s,q≠outputAddress b x) →
      abi.observe ((hostCode s binding).native abi state).2 q=abi.observe state q) ∧
    (∀ q,q.1≠(b.output 0).storage → (∀ x : Cell s,q≠outputAddress b x) →
      abi.observe ((hostCode s binding).native abi state).2 q=caller q) ∧
    (returned s b domain.device domain.extents 1).storage=domain.suppliedK.storage ∧
    (returned s b domain.device domain.extents 2).storage=domain.suppliedV.storage := by
  exact whole_host actual reference leaves abi p b domain binding state caller allocationFrame
theorem fp_frame (actual : StorageOps (BitVec 32) (BitVec 16))
    (abi : LocalEventABI (BitVec 32) (BitVec 16) σ actual) (raw : RawByteABI abi)
    (binding : CompilerBinding (BitVec 32) (BitVec 16) s p b) (state : σ)
    (storage offset : Nat) (outside : ∀ x : Cell s,(storage,offset/raw.width storage)≠outputAddress b x) :
    raw.bytes ((hostCode s binding).native abi state).2 (storage,offset)=raw.bytes state (storage,offset) := by
  exact native_full_byte_frame actual abi raw binding state storage offset outside
theorem fp_output_bytes (actual reference : StorageOps (BitVec 32) (BitVec 16))
    (leaves : NumericLeaves actual reference)
    (abi : LocalEventABI (BitVec 32) (BitVec 16) σ actual) (raw : RawByteABI abi)
    (p : Parameters (BitVec 32)) (b : Buffers) (domain : HostDomain s p b)
    (binding : CompilerBinding (BitVec 32) (BitVec 16) s p b) (state : σ)
    (x : Cell s) (i : Fin (raw.width (outputAddress b x).1)) :
    raw.bytes ((hostCode s binding).native abi state).2
      ((outputAddress b x).1,raw.width (outputAddress b x).1*(outputAddress b x).2+i.val)=
      cellByte (wordTarget reference p b selectedPlans selectedReductions (abi.observe state) x) i.val := by
  exact native_output_bytes actual reference leaves abi raw p b domain binding state x i

#axiomsClean fp_callee
#axiomsClean fp_qkv
#axiomsClean fp_equiv
#axiomsClean fp_frame
#axiomsClean fp_output_bytes

/-- Actual returned physical bytes, original-input word reference and complete
caller backing frame. All numerical hypotheses are scalar or one fixed K8
MMA instruction; compiler, allocation, view and storage APIs are structural. -/
specification rms_matmul_rbe_repaired_fp_returns
    (actual reference : StorageOps (BitVec 32) (BitVec 16)) (leaves : NumericLeaves actual reference)
    (abi : LocalEventABI (BitVec 32) (BitVec 16) σ actual)
    (i : Contract.Inputs) (h : Contract.InputContract i) (a : Checked.Allocation i) (scalar : ℝ → BitVec 32)
    (c : Checked.Compiler (BitVec 32) (BitVec 16) i a scalar) (returns : Checked.Returns i a)
    (physical : Checked.PhysicalABI abi i a returns) (before prepared : σ)
    (allocationWords : ∀ q,q.1≠a.q.base → abi.observe prepared q=abi.observe before q)
    (allocationBytes : ∀ t,t∈i.reads++[i.k,i.v] → ∀ byte,byte<t.bytes →
      physical.toRawByteABI.bytes prepared (t.base,byte)=physical.toRawByteABI.bytes before (t.base,byte)) :
    (Contract.checkInputs i=.ok () ↔ Contract.InputContract i) ∧
    (∀ phase,((Checked.entry abi c returns prepared).1 phase).metadata.shape=
      [(Checked.shape i).B,(Checked.shape i).M,(Checked.shape i).heads,(Checked.shape i).headDim]) ∧
    ((Checked.entry abi c returns prepared).1 0).metadata.dtype=Contract.fp16 ∧
    ((Checked.entry abi c returns prepared).1 1).metadata.dataPtr=i.k.metadata.dataPtr ∧
    ((Checked.entry abi c returns prepared).1 2).metadata.dataPtr=i.v.metadata.dataPtr ∧
    (∀ phase,((Checked.entry abi c returns prepared).1 phase).metadata.device=i.x.metadata.device) ∧
    (∀ phase,((Checked.entry abi c returns prepared).1 phase).metadata.dtype=
      if phase.val=0 then Contract.fp16 else i.qw.metadata.dtype) ∧
    (∀ x : Cell (Checked.shape i),∀ lane,lane<Contract.width (Checked.output i a.q x.1).metadata.dtype →
      physical.physicalByte (Checked.entry abi c returns prepared).2
        ((Checked.pointer64 (returns.tensor x.1) (Checked.flat3 (Checked.shape i).M (Checked.shape i).N
          x.2.1 x.2.2.1 x.2.2.2)).toNat+lane)=
      cellByte (wordTarget reference (Checked.parameters i scalar) (Checked.buffers i a.q)
        selectedPlans selectedReductions (abi.observe before) x) lane) ∧
    (∀ t,t∈i.reads → ∀ byte,byte<t.bytes →
      physical.physicalByte (Checked.entry abi c returns prepared).2 (t.base+byte)=physical.physicalByte before (t.base+byte)) ∧
    (∀ t,t∈i.reads++[i.k,i.v] → ∀ byte,byte<t.bytes →
      (∀ x : Cell (Checked.shape i),(t.base,byte/physical.toRawByteABI.width t.base)≠outputAddress (Checked.buffers i a.q) x) →
      physical.physicalByte (Checked.entry abi c returns prepared).2 (t.base+byte)=physical.physicalByte before (t.base+byte)) := by
  obtain ⟨hs,ht,hk,hv,hd⟩ := Checked.native_metadata abi i h a scalar c returns prepared
  exact ⟨Contract.checkInputs_iff i,hs,ht,hk,hv,hd,Checked.returned_dtypes i h a returns,
    Checked.native_return_bytes actual reference leaves abi i h a scalar c returns physical before prepared allocationWords,
    Checked.native_inputs_frame actual abi i h a scalar c returns physical before prepared allocationBytes,
    Checked.native_backing_frame actual abi i a scalar c returns physical before prepared allocationBytes⟩

specification rms_matmul_rbe_native_addresses (i : Contract.Inputs) (h : Contract.InputContract i)
    (a : Checked.Allocation i) (active : 0<(Checked.shape i).B ∧ 0<(Checked.shape i).M) :
    Checked.ActiveBounds (Checked.shape i) ∧ Checked.LoadsValid i ∧
    (∀ x : Cell (Checked.shape i),Checked.PointerValid (Checked.output i a.q x.1)
      (Checked.flat3 (Checked.shape i).M (Checked.shape i).N x.2.1 x.2.2.1 x.2.2.2)) ∧
    (∀ row,row<tilesM (Checked.shape i)*16 →
      -(2^63:Int) ≤ i.scalars.start+(row:Int) ∧ i.scalars.start+(row:Int)<(2^63:Int)) ∧
    (∀ x : Cell (Checked.shape i),
      (Checked.pointer64 (Checked.output i a.q x.1) (Checked.flat3 (Checked.shape i).M (Checked.shape i).N
        x.2.1 x.2.2.1 x.2.2.2)).toNat=(Checked.output i a.q x.1).base+
        Contract.width (Checked.output i a.q x.1).metadata.dtype*(outputAddress (Checked.buffers i a.q) x).2) :=
  ⟨Checked.active_bounds i h a active,Checked.loads_valid i h a,Checked.output_pointer_valid i h a,
    Checked.padded_position i h active,Checked.output_address64 i h a⟩

#axiomsClean rms_matmul_rbe_repaired_fp_returns
#axiomsClean rms_matmul_rbe_native_addresses
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsMatmulRbe
