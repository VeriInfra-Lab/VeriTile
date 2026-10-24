import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.Checked.Execution

namespace VeriTile.Bench.Repairs.TritonLinearActivation.Checked
noncomputable section
set_option maxHeartbeats 8000000

/-- Native full-call word semantics. Numeric hypotheses are only the nine
scalar/cast families and the actual fixed K8/K16 instruction lane family. -/
specification fp_equiv (ts : Fin 4 → Observed) (flag : Bool)
    (primitive : PrimitiveOps) (profile : LocalProfile) (leaves : NumericLeaves primitive profile)
    (allocate : DeviceState σ → Result σ Unit) (abi : LocalEventABI Word σ primitive.arithmetic)
    (rt : Tail) (h : Allocation r caller (before : Memory Word)) (layout : Layout h)
    (binding : PhysicalBinding h layout ts)
    (allocLaw : AllocationLaw abi h allocate) (tailLaw : TailContract rt h)
    (failures : Failures) (trials : List (Fin 98)) (c : Fin 98)
    (state after : DeviceState σ) (out : Tensor) (entry : abi.observe state.payload = before)
    (success : call ts flag allocate abi rt h failures trials c state = .ok out after) :
    Domain r ts flag ∧ out = h.returned ∧
    (∀ m n, m < r.batch.rows → n < r.n →
      abi.observe after.payload (h.storage 0,outputOffset r.shape m n) =
        (fpArithmetic profile).encode r.mode
          (activation (fpArithmetic profile) r.activation (callerPre (fpArithmetic profile) r c before m n))) ∧
    (∀ m n, m < r.batch.rows → n < r.n → r.hasSaved = Bool.true →
      abi.observe after.payload (r.storage 3,outputOffset r.shape m n) =
        savedValue (fpArithmetic profile) r.mode (callerPre (fpArithmetic profile) r c before m n)) ∧
    (∀ p, p.1 ∈ caller → ¬(r.hasSaved = Bool.true ∧ p.1 = r.storage 3 ∧ p.2 < r.batch.rows*r.n) →
      abi.observe after.payload p = before p) ∧
    after.currentDevice = state.currentDevice ∧ rt.save h.context = .ok () := by
  obtain ⟨admit,returned,memory,device,saved⟩ :=
    call_success ts flag binding.identity allocate abi rt h allocLaw tailLaw failures trials c state after out entry success
  have memory := memory.trans (congrArg (fun arithmetic => h.finalMemory arithmetic c) leaves.arithmetic_eq)
  have valid := domain_valid r ts binding.identity admit
  exact ⟨admit,returned,by simpa only [memory] using h.final_output (fpArithmetic profile) c valid,
    by simpa only [memory] using h.final_saved (fpArithmetic profile) c valid,
    by simpa only [memory] using h.final_frame (fpArithmetic profile) c valid,device,saved⟩

/-- All actual bytes of the returned output storage, observed independently
from the logical word map and connected only by a single-cell codec law. -/
theorem output_bytes (ts : Fin 4 → Observed) (flag : Bool)
    (primitive : PrimitiveOps) (profile : LocalProfile) (leaves : NumericLeaves primitive profile)
    (allocate : DeviceState σ → Result σ Unit) (abi : LocalEventABI Word σ primitive.arithmetic)
    (rt : Tail) (h : Allocation r caller (before : Memory Word)) (layout : Layout h)
    (binding : PhysicalBinding h layout ts) (raw : ByteABI abi layout.backing layout.mode)
    (allocLaw : AllocationLaw abi h allocate) (tailLaw : TailContract rt h)
    (failures : Failures) (trials : List (Fin 98)) (c : Fin 98)
    (state after : DeviceState σ) (out : Tensor) (entry : abi.observe state.payload = before)
    (success : call ts flag allocate abi rt h failures trials c state = .ok out after)
    (m n j : Nat) (hm : m < r.batch.rows) (hn : n < r.n) (byte : j < byteWidth r.mode) :
    raw.observe after.payload (layout.backing (h.storage 0)).owner
      ((layout.backing (h.storage 0)).baseByte+byteWidth r.mode*outputOffset r.shape m n+j) =
    wordByte ((fpArithmetic profile).encode r.mode
      (activation (fpArithmetic profile) r.activation (callerPre (fpArithmetic profile) r c before m n))) j := by
  have values := (fp_equiv ts flag primitive profile leaves allocate abi rt h layout binding
    allocLaw tailLaw failures trials c state after out entry success).2.2.1 m n hm hn
  rw [raw.codec after.payload r.mode (h.storage 0,outputOffset r.shape m n) layout.outputMode
    (cell_capacity _ _ _ _ (output_index r m n hm hn) layout.outputBytes) j byte,values]

/-- The caller's original FP32 saved tensor receives the rounded half scratch
through scalar FP16 decoding and FP32 copyback when source AMP is enabled. -/
theorem saved_bytes (ts : Fin 4 → Observed) (flag : Bool)
    (primitive : PrimitiveOps) (profile : LocalProfile) (leaves : NumericLeaves primitive profile)
    (allocate : DeviceState σ → Result σ Unit) (abi : LocalEventABI Word σ primitive.arithmetic)
    (rt : Tail) (h : Allocation r caller (before : Memory Word)) (layout : Layout h)
    (binding : PhysicalBinding h layout ts) (raw : ByteABI abi layout.backing layout.mode)
    (allocLaw : AllocationLaw abi h allocate) (tailLaw : TailContract rt h)
    (failures : Failures) (trials : List (Fin 98)) (c : Fin 98)
    (state after : DeviceState σ) (out : Tensor) (entry : abi.observe state.payload = before)
    (success : call ts flag allocate abi rt h failures trials c state = .ok out after)
    (m n j : Nat) (hm : m < r.batch.rows) (hn : n < r.n) (byte : j < 4) (saved : r.hasSaved = Bool.true) :
    raw.observe after.payload (layout.backing (r.storage 3)).owner
      ((layout.backing (r.storage 3)).baseByte+4*outputOffset r.shape m n+j) =
    wordByte (savedValue (fpArithmetic profile) r.mode (callerPre (fpArithmetic profile) r c before m n)) j := by
  have values := (fp_equiv ts flag primitive profile leaves allocate abi rt h layout binding
    allocLaw tailLaw failures trials c state after out entry success).2.2.2.1 m n hm hn saved
  have codec := raw.codec after.payload .fp32 (r.storage 3,outputOffset r.shape m n) (layout.savedMode saved)
    (cell_capacity .fp32 _ _ _ (output_index r m n hm hn) (layout.savedBytes saved)) j byte
  simpa only [Prod.fst,Prod.snd,byteWidth,values] using codec

/-- Full caller-backing exterior frame on success and every ordinary failure.
This includes arbitrary bytes after the saved logical prefix. -/
theorem frame (ts : Fin 4 → Observed) (flag : Bool)
    (allocate : DeviceState σ → Result σ Unit) (h : Allocation r caller (before : Memory Word))
    (layout : Layout h) (abi : LocalEventABI Word σ a) (raw : ByteABI abi layout.backing layout.mode)
    (allocLaw : AllocationByteLaw h layout abi raw allocate)
    (rt : Tail) (failures : Failures) (trials : List (Fin 98)) (c : Fin 98)
    (state : DeviceState σ) (old owner address : Nat) (live : old ∈ caller)
    (covered : (layout.backing old).covers owner address)
    (outside : ¬SavedCovers r layout.backing owner address) :
    raw.observe (call ts flag allocate abi rt h failures trials c state).state.payload owner address =
      raw.observe state.payload owner address :=
  call_frame ts flag allocate h layout abi raw allocLaw rt failures trials c state old owner address live covered outside

#axiomsClean fp_equiv
#axiomsClean output_bytes
#axiomsClean saved_bytes
#axiomsClean frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonLinearActivation.Checked
