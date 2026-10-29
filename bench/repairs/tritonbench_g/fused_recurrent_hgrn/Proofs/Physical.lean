import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.Bounds
import bench.repairs.tritonbench_g.swiglu_triton.Proofs.Memory

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn.Physical
open Checked
open VeriTile.Bench.Repairs.SwigluTriton.Memory (Key)
abbrev API := VeriTile.Bench.Repairs.SwigluTriton.Memory.Ops
noncomputable section
open Classical

structure Cell where
  role : Role
  offset : Nat
  deriving DecidableEq

def key (runtime : Runtime c U S V) (x : Cell) : Key :=
  ⟨(runtime.tensor x.role).device,(runtime.tensor x.role).base+4*x.offset⟩

def Active (c : Shape) (U S V : Bool) (x : Cell) : Prop :=
  present U S V x.role=Bool.true ∧ x.offset<extent c x.role

theorem metadata (c : Shape) (U S V : Bool) (runtime : Runtime c U S V)
    (accepted : check c U S V runtime=Bool.true) (x : Cell) (active : Active c U S V x) :
    (runtime.tensor x.role).base=(runtime.tensor x.role).allocationBase+4*(runtime.tensor x.role).storageOffset ∧
    (runtime.tensor x.role).base%4=0 ∧
    4*(runtime.tensor x.role).storageOffset+4*extent c x.role ≤ (runtime.tensor x.role).allocationBytes ∧
    (runtime.tensor x.role).allocationBase+(runtime.tensor x.role).allocationBytes ≤ 2^64 := by
  have facts := accepted
  simp only [check,Bool.and_eq_true] at facts
  have member : x.role∈roles := by cases x.role <;> simp [roles]
  have tensor := List.all_eq_true.mp facts.1.1 x.role member
  simp only [active.1,↓reduceIte,decide_eq_true_eq] at tensor
  have result : extent c x.role=0 ∨
      ((runtime.tensor x.role).base=(runtime.tensor x.role).allocationBase+4*(runtime.tensor x.role).storageOffset ∧
       (runtime.tensor x.role).base%4=0 ∧
       4*(runtime.tensor x.role).storageOffset+4*extent c x.role ≤ (runtime.tensor x.role).allocationBytes ∧
       (runtime.tensor x.role).allocationBase+(runtime.tensor x.role).allocationBytes ≤ 2^64) :=
    tensor.2.2.2.2.2.2.2.2.2
  exact result.resolve_left (by have := active.2; omega)

theorem key_bounds (c : Shape) (U S V : Bool) (runtime : Runtime c U S V)
    (accepted : check c U S V runtime=Bool.true) (x : Cell) (active : Active c U S V x) :
    (runtime.tensor x.role).allocationBase ≤ (key runtime x).address ∧
    (key runtime x).address+4 ≤ (runtime.tensor x.role).allocationBase+(runtime.tensor x.role).allocationBytes ∧
    (key runtime x).address+4 ≤ 2^64 ∧ (key runtime x).address%4=0 := by
  have facts := metadata c U S V runtime accepted x active
  have offset := active.2
  simp only [key]
  refine ⟨by omega,by omega,by omega,?_⟩
  simp [Nat.add_mod,facts.2.1]

def regionCells (r : Role) (count : Nat) : List Cell := (List.range count).map fun a => ⟨r,a⟩
def cells (c : Shape) (S forward : Bool) : List Cell :=
  if forward then regionCells .o c.rows ++ (if S then regionCells .ht c.slots else [])
  else regionCells .dx c.rows ++ regionCells .dg c.rows

theorem region_mem (r : Role) (count : Nat) (x : Cell) :
    x∈regionCells r count ↔ x.role=r ∧ x.offset<count := by
  cases x
  simp [regionCells,and_comm,eq_comm]

theorem cells_mem (c : Shape) (S phase : Bool) (x : Cell) : x∈cells c S phase ↔
    (if phase then (x.role=.o ∧ x.offset<c.rows) ∨ (S=Bool.true ∧ x.role=.ht ∧ x.offset<c.slots)
    else (x.role=.dx ∨ x.role=.dg) ∧ x.offset<c.rows) := by
  cases phase <;> cases S <;> cases hr : x.role <;> simp [cells,region_mem,hr]

theorem cells_active (c : Shape) (U S V phase : Bool) (x : Cell) (member : x∈cells c S phase) :
    written x.role ∧ Active c U S V x := by
  have facts := (cells_mem c S phase x).mp member
  cases phase
  · rcases facts with ⟨hr,ha⟩
    rcases hr with hr|hr <;> simp [written,Active,present,extent,stateRole,ha,hr]
  · rcases facts with ⟨hr,ha⟩|⟨hS,hr,ha⟩ <;>
      simp_all [written,Active,present,extent,stateRole]

theorem key_disjoint (c : Shape) (U S V : Bool) (runtime : Runtime c U S V)
    (accepted : check c U S V runtime=Bool.true) (x y : Cell)
    (ax : Active c U S V x) (ay : Active c U S V y) (wx : written x.role) (wy : written y.role)
    (different : x≠y) : (key runtime x).disjoint (key runtime y) := by
  by_cases roles : x.role=y.role
  · have offsets : x.offset≠y.offset := by
      intro equal
      apply different
      cases x;cases y
      simp_all
    simp only [key,Key.disjoint]
    rw [roles]
    by_cases before : x.offset<y.offset
    · exact Or.inr (Or.inl (by omega))
    · exact Or.inr (Or.inr (by omega))
  · have bx := key_bounds c U S V runtime accepted x ax
    have by' := key_bounds c U S V runtime accepted y ay
    have fresh := runtime.freshOutputs x.role y.role wx wy roles
    rcases fresh with owner|before|after
    · exact Or.inl owner
    · exact Or.inr (Or.inl (by omega))
    · exact Or.inr (Or.inr (by omega))

/-- Physical finite-store normal form of the closed source interpreter. -/
def commit {M α β : Type} (api : API M α β) (c : Shape) (U S V : Bool)
    (runtime : Runtime c U S V) (phase : Bool) (values : Role → Nat → α) (m : M) : M :=
  VeriTile.Bench.Repairs.SwigluTriton.Memory.stores api (key runtime)
    (fun x => values x.role x.offset) (cells c S phase) m

theorem readback {M α β : Type} (api : API M α β) (c : Shape) (U S V : Bool)
    (runtime : Runtime c U S V) (accepted : check c U S V runtime=Bool.true)
    (phase : Bool) (values : Role → Nat → α) (m : M) (x : Cell) (member : x∈cells c S phase) :
    api.read (commit api c U S V runtime phase values m) (key runtime x)=values x.role x.offset := by
  have xactive := cells_active c U S V phase x member
  apply VeriTile.Bench.Repairs.SwigluTriton.Memory.readback
    api (key runtime) (fun x => values x.role x.offset) (cells c S phase) m x member
  · intro y hy equal
    have yactive := cells_active c U S V phase y hy
    have same : y=x := by
      by_contra different
      exact Key.disjoint_ne (key_disjoint c U S V runtime accepted y x yactive.2 xactive.2
        yactive.1 xactive.1 different) equal
    rw [same]
  · intro y hy different
    have yactive := cells_active c U S V phase y hy
    exact key_disjoint c U S V runtime accepted x y xactive.2 yactive.2 xactive.1 yactive.1
      (by intro same; subst y; exact different rfl)

def Outside (c : Shape) (U S V : Bool) (runtime : Runtime c U S V)
    (phase : Bool) (owner address : Nat) : Prop :=
  ∀ x,x∈cells c S phase → ¬(key runtime x).covers owner address

theorem frame {M α β : Type} (api : API M α β) (c : Shape) (U S V : Bool)
    (runtime : Runtime c U S V) (phase : Bool) (values : Role → Nat → α) (m : M) (owner address : Nat)
    (outside : Outside c U S V runtime phase owner address) :
    api.observe (commit api c U S V runtime phase values m) owner address=api.observe m owner address :=
  VeriTile.Bench.Repairs.SwigluTriton.Memory.bytes_frame api _ _ _ m owner address outside

theorem forward_keeps {M α β : Type} (api : API M α β) (c : Shape) (U S V : Bool)
    (runtime : Runtime c U S V) (accepted : check c U S V runtime=Bool.true)
    (values : Role → Nat → α) (m : M) (x : Cell) (active : Active c U S V x)
    (readonly : x.role=.x ∨ x.role=.g ∨ x.role=.h0) :
    api.read (commit api c U S V runtime Bool.true values m) (key runtime x)=api.read m (key runtime x) := by
  apply VeriTile.Bench.Repairs.SwigluTriton.Memory.read_frame
  intro y member
  have writes := (cells_mem c S Bool.true y).mp member
  have role : y.role=.o ∨ y.role=.ht := by rcases writes with h|h;exact Or.inl h.1;exact Or.inr h.2.1
  have writer := key_bounds c U S V runtime accepted y (cells_active c U S V Bool.true y member).2
  have reader := key_bounds c U S V runtime accepted x active
  have fresh := runtime.freshForward y.role role x.role readonly
  rcases fresh with different|before|after
  · exact Or.inl (Ne.symm different)
  · exact Or.inr (Or.inr (by omega))
  · exact Or.inr (Or.inl (by omega))

theorem backward_keeps_outputs {M α β : Type} (api : API M α β) (c : Shape) (U S V : Bool)
    (runtime : Runtime c U S V) (accepted : check c U S V runtime=Bool.true)
    (values : Role → Nat → α) (m : M) (x : Cell) (active : Active c U S V x)
    (readonly : x.role=.o ∨ x.role=.ht) :
    api.read (commit api c U S V runtime Bool.false values m) (key runtime x)=api.read m (key runtime x) := by
  apply VeriTile.Bench.Repairs.SwigluTriton.Memory.read_frame
  intro y member
  have facts := (cells_active c U S V Bool.false y member)
  have writer := (cells_mem c S Bool.false y).mp member
  have wx : written x.role := by rcases readonly with h|h <;> simp [written,h]
  have different : x≠y := by
    intro same
    have roles := congrArg Cell.role same
    rcases readonly with h|h <;> rcases writer.1 with g|g <;> simp_all
  exact key_disjoint c U S V runtime accepted x y active facts.2 wx facts.1 different

/-- Entire caller backing allocations are readonly at each phase, including
padding and storage before/after a contiguous view. Incoming cotangents may
alias saved outputs because both belong to the backward readonly set. -/
def Readonly (phase : Bool) (r : Role) : Prop :=
  if phase then r=.x ∨ r=.g ∨ r=.h0
  else r=.x ∨ r=.g ∨ r=.o ∨ r=.ht ∨ r=.h0 ∨ r=.incoming ∨ r=.dht

theorem caller_frame {M α β : Type} (api : API M α β) (c : Shape) (U S V : Bool)
    (runtime : Runtime c U S V) (accepted : check c U S V runtime=Bool.true)
    (phase : Bool) (values : Role → Nat → α) (m : M) (r : Role) (readonly : Readonly phase r)
    (address : Nat) (lo : (runtime.tensor r).allocationBase ≤ address)
    (hi : address < (runtime.tensor r).allocationBase+(runtime.tensor r).allocationBytes) :
    api.observe (commit api c U S V runtime phase values m) (runtime.tensor r).device address=
      api.observe m (runtime.tensor r).device address := by
  apply frame api c U S V runtime phase values m (runtime.tensor r).device address
  intro x hx covered
  have writer := key_bounds c U S V runtime accepted x (cells_active c U S V phase x hx).2
  have writes := (cells_mem c S phase x).mp hx
  have fresh : separate (runtime.tensor x.role) (runtime.tensor r) := by
    cases phase
    · exact runtime.freshBackward x.role writes.1 r readonly
    · have role : x.role=.o ∨ x.role=.ht := by
        rcases writes with h|h
        · exact Or.inl h.1
        · exact Or.inr h.2.1
      exact runtime.freshForward x.role role r readonly
  rcases covered with ⟨owner,low,high⟩
  rcases fresh with other|before|after
  · exact other owner.symm
  · omega
  · omega

#axiomsClean caller_frame
#axiomsClean key_bounds
#axiomsClean key_disjoint
#axiomsClean readback
#axiomsClean frame
#axiomsClean forward_keeps
#axiomsClean backward_keeps_outputs
end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn.Physical
