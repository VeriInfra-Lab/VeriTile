import bench.repairs.tritonbench_g.rotary_emb.Proofs.Host

namespace VeriTile.Bench.Repairs.RotaryEmb

/-- Raw FP32 bytes at the original Q/K allocations; cosine, sine, tails and
all exterior bytes are framed. Source semantics uses actual primitive ports;
the independent target uses only the selected local PTX graph. -/
structure NativeClaim (call : Host.Metadata) (callerDevice : Nat)
    (actual target : Native.Instructions) (initial : Physical.Bytes)
    (result : Host.Returned Physical.Byte) : Prop where
  returnedNone : result.value=none
  identities : result.identities=call.identities
  device : result.deviceRestored=callerDevice
  source : Physical.load result.memory=
    Native.project (RotaryEmb.source call.shape (Native.ops actual) (Native.liftMemory (Physical.load initial)))
  values : ∀ w,call.shape.Active w → ∀ odd,
    Physical.load result.memory (dataRegion w.key) (call.shape.cell w odd)=
      Native.pairWord call.shape target w (Physical.load initial) odd
  bytes : ∀ w,call.shape.Active w → ∀ odd i,
    result.memory (dataRegion w.key) (4*call.shape.cell w odd+i.val)=
      Physical.encode (Native.pairWord call.shape target w (Physical.load initial) odd) i
  tail : ∀ key n h d,h<call.shape.heads key → d<call.shape.width → call.shape.rotary≤d → ∀ i : Fin 4,
    result.memory (dataRegion key) (4*call.shape.address key n h d+i.val)=
      initial (dataRegion key) (4*call.shape.address key n h d+i.val)
  coefficients : ∀ (sine : Bool) a,result.memory (if sine then .sin else .cos) a=initial (if sine then .sin else .cos) a
  frame : ∀ r a,(∀ w,call.shape.Active w → ¬Writes call.shape w r (a/4)) → result.memory r a=initial r a

/-- Structural full-source result has no numerical correspondence hypothesis. -/
theorem whole_native (call : Host.Metadata) (domain : Host.Admitted call)
    (callerDevice : Nat) (actual : Native.Instructions) (initial : Physical.Bytes) :
    NativeClaim call callerDevice actual actual initial (Host.selected call callerDevice actual initial) := by
  have memory := Host.selected_memory call callerDevice actual initial
  refine ⟨rfl,rfl,rfl,?_,?_,?_,?_,?_,?_⟩
  · rw [memory]
    exact Physical.native_source call.shape domain.shape actual initial
  · intro w active odd
    rw [memory]
    exact Physical.native_result call.shape domain.shape actual initial w active odd
  · intro w active odd i
    rw [memory]
    exact Physical.native_bytes call.shape domain.shape actual initial w active odd i
  · intro key n h d hh hd tail i
    rw [memory]
    exact Physical.native_tail call.shape domain.shape actual initial key n h d hh hd tail i
  · intro sine a
    rw [memory]
    exact Physical.native_coefficients call.shape actual initial sine a
  · intro r a outside
    rw [memory]
    exact Physical.native_frame call.shape actual initial r a outside

/-- Three local instruction equalities are the entire numerical bridge.
No output, whole-kernel correctness, readback or coverage law is a premise. -/
theorem whole_fp (call : Host.Metadata) (domain : Host.Admitted call)
    (callerDevice : Nat) (actual target : Native.Instructions)
    (numeric : Native.Obligations actual target) (initial : Physical.Bytes) :
    NativeClaim call callerDevice actual target initial (Host.selected call callerDevice actual initial) := by
  rw [←Native.instructions_eq actual target numeric]
  exact whole_native call domain callerDevice actual initial

#axiomsClean whole_native
#axiomsClean whole_fp
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.RotaryEmb
