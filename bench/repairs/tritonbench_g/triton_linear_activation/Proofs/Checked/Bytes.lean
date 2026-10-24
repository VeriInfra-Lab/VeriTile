import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.Checked.Footprint
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.FPReference
import bench.repairs.tritonbench_g.swiglu_triton.Proofs.Memory

namespace VeriTile.Bench.Repairs.TritonLinearActivation.Checked
noncomputable section
open Classical
set_option maxHeartbeats 6000000
abbrev Backing := SwigluTriton.Memory.Allocation
abbrev Byte := BitVec 8

def byteWidth : Mode → Nat | .fp32 => 4 | .autocastFP16 => 2
def wordByte (x : Word) (i : Nat) : Byte := BitVec.ofNat 8 (x.raw/256^i)

def Covers (backing : Nat → Backing) (mode : Mode) (p : Address) (owner address : Nat) : Prop :=
  owner = (backing p.1).owner ∧
  (backing p.1).baseByte+byteWidth mode*p.2 ≤ address ∧
  address < (backing p.1).baseByte+byteWidth mode*p.2+byteWidth mode

/-- Actual storage descriptors and ordinary allocator capacities. These
fields contain no kernel output, memory equality or successful execution law. -/
structure Layout (h : Allocation r caller (before : Memory α)) where
  backing : Nat → Backing
  mode : Nat → Mode
  outputOwner : (backing (h.storage 0)).owner = r.device
  outputMode : mode (h.storage 0) = r.mode
  outputBytes : byteWidth r.mode*(r.batch.rows*r.n) ≤ (backing (h.storage 0)).bytes
  castOwner : r.mode = .autocastFP16 → ∀ i : Fin 3, (backing (h.storage (castSlot i))).owner = r.device
  castMode : r.mode = .autocastFP16 → ∀ i : Fin 3, mode (h.storage (castSlot i)) = .autocastFP16
  castBytes : r.mode = .autocastFP16 → ∀ i : Fin 3, 2*r.count (inputSlot i) ≤ (backing (h.storage (castSlot i))).bytes
  scratchOwner : r.mode = .autocastFP16 → r.hasSaved = Bool.true → (backing (h.storage 4)).owner = r.device
  scratchMode : r.mode = .autocastFP16 → r.hasSaved = Bool.true → mode (h.storage 4) = .autocastFP16
  scratchBytes : r.mode = .autocastFP16 → r.hasSaved = Bool.true → 2*(r.batch.rows*r.n) ≤ (backing (h.storage 4)).bytes
  savedMode : r.hasSaved = Bool.true → mode (r.storage 3) = .fp32
  savedBytes : r.hasSaved = Bool.true → 4*(r.batch.rows*r.n) ≤ (backing (r.storage 3)).bytes
  fresh : ∀ i old, old ∈ caller → (backing old).separate (backing (h.storage i))

def SavedCovers (r : Request) (backing : Nat → Backing) (owner address : Nat) : Prop :=
  r.hasSaved = Bool.true ∧ owner = (backing (r.storage 3)).owner ∧
  (backing (r.storage 3)).baseByte ≤ address ∧
  address < (backing (r.storage 3)).baseByte+4*(r.batch.rows*r.n)

theorem fresh_outside (layout : Layout (h : Allocation r caller (before : Memory α)))
    (slot : Fin 6) (old owner address : Nat) (live : old ∈ caller)
    (covered : (layout.backing old).covers owner address) (mode : Mode) (i count : Nat)
    (hi : i < count) (capacity : byteWidth mode*count ≤ (layout.backing (h.storage slot)).bytes) :
    ¬ Covers layout.backing mode (h.storage slot,i) owner address := by
  intro hit
  have sep := layout.fresh slot old live
  have positive : 0 < byteWidth mode := by cases mode <;> decide
  have last : byteWidth mode*i+byteWidth mode ≤ byteWidth mode*count := by
    have := Nat.mul_le_mul_left (byteWidth mode) (Nat.succ_le_of_lt hi)
    simpa [Nat.mul_add] using this
  simp only [Covers] at hit
  simp only [SwigluTriton.Memory.Allocation.covers] at covered
  rcases sep with different | before | after
  · exact different (covered.1.symm.trans hit.1)
  · omega
  · omega

theorem allowed_outside (layout : Layout (h : Allocation r caller (before : Memory α)))
    (old owner address : Nat) (live : old ∈ caller)
    (covered : (layout.backing old).covers owner address)
    (outside : ¬SavedCovers r layout.backing owner address) :
    ∀ mode p, Allowed h mode p → ¬ Covers layout.backing mode p owner address := by
  intro mode p allowed
  rcases p with ⟨storage,index⟩
  rcases allowed with output | casts | scratch | saved
  · rcases output with ⟨rfl,hi,rfl⟩
    exact fresh_outside layout 0 old owner address live covered _ index _ hi layout.outputBytes
  · obtain ⟨mode,i,rfl,hi,rfl⟩ := casts
    exact fresh_outside layout (castSlot i) old owner address live covered _ index _ hi (layout.castBytes mode i)
  · rcases scratch with ⟨mode,flag,rfl,hi,rfl⟩
    exact fresh_outside layout 4 old owner address live covered _ index _ hi (layout.scratchBytes mode flag)
  · rcases saved with ⟨flag,rfl,hi,rfl⟩
    intro hit
    apply outside
    simp only [Covers,byteWidth] at hit
    exact ⟨flag,hit.1,by omega,by omega⟩

/-- Raw byte observation is independent of typed word observation. The only
connection is an ordinary one-cell codec law; the only frame assumption is
one primitive event's exact byte span. -/
structure ByteABI (abi : LocalEventABI Word σ a) (backing : Nat → Backing) (modes : Nat → Mode) where
  observe : σ → Nat → Nat → Byte
  codec : ∀ state mode p, modes p.1 = mode →
    byteWidth mode*(p.2+1) ≤ (backing p.1).bytes → ∀ i, i < byteWidth mode →
      observe state (backing p.1).owner ((backing p.1).baseByte+byteWidth mode*p.2+i) =
        wordByte (abi.observe state p) i
  frame : ∀ (β : Type) (e : Event Word β) state owner address,
    (match e with | .store mode p _ => ¬Covers backing mode p owner address | _ => True) →
    observe (abi.step e state).2 owner address = observe state owner address

theorem native_byte_frame (abi : LocalEventABI Word σ a) (raw : ByteABI abi backing modes)
    (code : Code Word β) (state : σ) (safe : Code.Safe allowed code)
    (owner address : Nat) (outside : ∀ mode p, allowed mode p → ¬Covers backing mode p owner address) :
    raw.observe (code.native abi state).2 owner address = raw.observe state owner address := by
  induction code generalizing state with
  | pure x => rfl
  | event e next ih =>
    simp only [Code.native]
    rw [ih _ _ (safe.2 _)]
    apply raw.frame
    cases e <;> try trivial
    exact outside _ _ safe.1

/-- Every finite native prefix has the same exterior frame, including a
prefix interrupted after output/saved stores or in AMP copyback. -/
def runPrefix (abi : LocalEventABI α σ a) : Nat → Code α β → σ → σ
  | 0,_,state => state
  | _+1,.pure _,state => state
  | fuel+1,.event e next,state =>
    let step := abi.step e state
    runPrefix abi fuel (next step.1) step.2

theorem prefix_byte_frame (abi : LocalEventABI Word σ a) (raw : ByteABI abi backing modes)
    (code : Code Word β) (state : σ) (safe : Code.Safe allowed code) (fuel : Nat)
    (owner address : Nat) (outside : ∀ mode p, allowed mode p → ¬Covers backing mode p owner address) :
    raw.observe (runPrefix abi fuel code state) owner address = raw.observe state owner address := by
  induction fuel generalizing code state with
  | zero => rfl
  | succ fuel ih =>
    cases code with
    | pure x => rfl
    | event e next =>
      simp only [runPrefix]
      rw [ih _ _ (safe.2 _)]
      apply raw.frame
      cases e <;> try trivial
      exact outside _ _ safe.1

theorem whole_byte_frame (h : Allocation r caller (before : Memory Word)) (layout : Layout h)
    (abi : LocalEventABI Word σ a) (raw : ByteABI abi layout.backing layout.mode)
    (state : σ) (trials : List (Fin 98)) (c : Fin 98) (old owner address : Nat)
    (live : old ∈ caller) (covered : (layout.backing old).covers owner address)
    (outside : ¬SavedCovers r layout.backing owner address) :
    raw.observe ((h.autotunedCode trials c).native abi state).2 owner address = raw.observe state owner address :=
  native_byte_frame abi raw _ state (safe_autotuned h trials c) owner address
    (allowed_outside layout old owner address live covered outside)

#axiomsClean native_byte_frame
#axiomsClean prefix_byte_frame
#axiomsClean whole_byte_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonLinearActivation.Checked
