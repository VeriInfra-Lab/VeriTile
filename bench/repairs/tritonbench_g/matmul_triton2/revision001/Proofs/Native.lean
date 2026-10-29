import bench.repairs.tritonbench_g.matmul_triton2.revision001.Proofs.Host
namespace VeriTile.Bench.Repairs.MatmulTriton2.Revision001.Native
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
abbrev Word := FP.Value .fp32
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- One output lane of a single fixed-k8 hardware instruction. Both operands
contain exactly eight source FP32 payloads consumed in the instruction's TF32
input mode; seed and result are FP32. No block/tile/full-K oracle is available.
The hardware's internal accumulation order is not replaced by a binary fold. -/
abbrev MMA8 := (Fin 8 → Word) → (Fin 8 → Word) → Word → Word
structure Instructions where
  zero : Word
  sync8 : MMA8
  warpGroup8 : MMA8
structure Reference where
  sync8 : MMA8
  warpGroup8 : MMA8
def referenceInstructions (r : Reference) : Instructions :=
  ⟨FP.Value.zero .fp32,r.sync8,r.warpGroup8⟩
inductive Family where
  | sync | warpGroup
  deriving DecidableEq
def instruction (I : Instructions) : Family → MMA8
  | .sync => I.sync8
  | .warpGroup => I.warpGroup8

/-- Independent numerical references are bound at the fixed instruction
boundary. Their actual selected rounding, exceptional values and TF32 behavior
remain explicit numerical work; this package does not invent an internal tree. -/
structure Correspondence (actual : Instructions) (reference : Reference) : Prop where
  zero : actual.zero=FP.Value.zero .fp32
  sync8 : ∀ a b seed, actual.sync8 a b seed=reference.sync8 a b seed
  warpGroup8 : ∀ a b seed, actual.warpGroup8 a b seed=reference.warpGroup8 a b seed

def lane {n : Nat} (zero : Word) (a : Fin n → Word) (group : Nat) (j : Fin 8) : Word :=
  if h:group*8+j.val<n then a ⟨group*8+j.val,h⟩ else zero
def chunk (I : Instructions) (family : Family) {n : Nat}
    (a b : Fin n → Word) (z : Word) (group : Nat) : Word :=
  instruction I family (lane I.zero a group) (lane I.zero b group) z
def dot (I : Instructions) (family : Family) {n : Nat}
    (a b : Fin n → Word) (z : Word) : Word :=
  (List.range ((n+7)/8)).foldl (chunk I family a b) z
def primitives (I : Instructions) (family : Family) : Core.Primitives Word where
  zero := I.zero
  dot := dot I family

theorem lane_index (k : Nat) (j : Fin k) :
    j.val/8*8+j.val%8=j.val ∧ j.val%8<8 ∧ j.val/8<(k+7)/8 := by
  have hm := Nat.mod_lt j.val (by decide : 0<8)
  have hd := Nat.div_add_mod j.val 8
  constructor
  · omega
  constructor
  · exact hm
  · have hq := Nat.div_add_mod (k+7) 8
    have hr := Nat.mod_lt (k+7) (by decide : 0<8)
    omega

theorem lane_unique (g h : Nat) (i j : Fin 8) (he : g*8+i.val=h*8+j.val) :
    g=h ∧ i=j := by
  have hi := i.isLt
  have hj := j.isLt
  have hg : g=h := by omega
  refine ⟨hg,Fin.ext ?_⟩
  omega

theorem correspondence_consistent (reference : Reference) :
    Correspondence (referenceInstructions reference) reference := ⟨rfl,fun _ _ _ => rfl,fun _ _ _ => rfl⟩

theorem instruction_eq (actual : Instructions) (reference : Reference) (numeric : Correspondence actual reference)
    (family : Family) : instruction actual family=instruction (referenceInstructions reference) family := by
  cases family
  · funext a b seed; exact numeric.sync8 a b seed
  · funext a b seed; exact numeric.warpGroup8 a b seed

theorem chunk_eq (actual : Instructions) (reference : Reference) (numeric : Correspondence actual reference)
    (family : Family) {n : Nat} (a b : Fin n → Word) (z : Word) (group : Nat) :
    chunk actual family a b z group=chunk (referenceInstructions reference) family a b z group := by
  simp only [chunk,instruction_eq actual reference numeric,numeric.zero,referenceInstructions]
theorem dot_eq (actual : Instructions) (reference : Reference) (numeric : Correspondence actual reference)
    (family : Family) {n : Nat} (a b : Fin n → Word) (z : Word) :
    dot actual family a b z=dot (referenceInstructions reference) family a b z := by
  unfold dot
  have he : chunk actual family a b=chunk (referenceInstructions reference) family a b := by
    funext z g; exact chunk_eq actual reference numeric family a b z g
  rw [he]
theorem primitives_eq (actual : Instructions) (reference : Reference) (numeric : Correspondence actual reference)
    (family : Family) : primitives actual family=primitives (referenceInstructions reference) family := by
  have hd : @dot actual family=@dot (referenceInstructions reference) family := by
    funext n a b z; exact dot_eq actual reference numeric family a b z
  simp only [primitives,numeric.zero,hd,referenceInstructions]

def config (choice : Fin 8) : Host.Config :=
  ![⟨128,256,64,8⟩,⟨64,256,32,8⟩,⟨128,128,32,8⟩,⟨128,64,32,8⟩,
    ⟨64,128,32,8⟩,⟨128,32,32,8⟩,⟨64,32,32,8⟩,⟨32,64,32,8⟩] choice
def family (choice : Fin 8) : Family := if choice.val<6 then .warpGroup else .sync
def numWarps (choice : Fin 8) : Nat := ![8,4,4,4,4,4,2,2] choice
def numStages (choice : Fin 8) : Nat := ![3,4,4,4,4,4,5,5] choice

theorem config_legal (choice : Fin 8) :
    0<(config choice).blockRows ∧ 0<(config choice).blockColumns ∧
    0<(config choice).blockInner ∧ 0<(config choice).groupRows ∧
    (config choice).blockInner%8=0 := by fin_cases choice <;> decide

theorem all_k8_lanes_in_range (choice : Fin 8) (g : Nat)
    (hg : g<((config choice).blockInner+7)/8) (j : Fin 8) :
    g*8+j.val<(config choice).blockInner := by
  fin_cases choice <;> norm_num [config] at hg ⊢ <;> omega

theorem host_eq (actual : Instructions) (reference : Reference) (numeric : Correspondence actual reference)
    (family : Family) (rest : Algebra Word) (allocate : Host.Allocator Word)
    (x : Host.Arguments) (s : State Word) :
    Host.call rest (primitives actual family) allocate x s=
      Host.call rest (primitives (referenceInstructions reference) family) allocate x s := by
  rw [primitives_eq actual reference numeric family]

#axiomsClean lane_index
#axiomsClean lane_unique
#axiomsClean correspondence_consistent
#axiomsClean instruction_eq
#axiomsClean chunk_eq
#axiomsClean dot_eq
#axiomsClean primitives_eq
#axiomsClean config_legal
#axiomsClean all_k8_lanes_in_range
#axiomsClean host_eq
end
end VeriTile.Bench.Repairs.MatmulTriton2.Revision001.Native
