import bench.repairs.tritonbench_g.index_select_bwd.Proofs.Kernel
import bench.repairs.tritonbench_g.index_select_bwd.Proofs.Physical

namespace VeriTile.Bench.Repairs.IndexSelectBwd.KernelBinding
open VeriTile.Triton FP.Structural Logical Physical
open VeriTile.Torch.FP32 (logicalValue)
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- The exact slice pointers passed to one launch. The selected row is signed
metadata checked before dispatch; there is no index load or atomic in this kernel. -/
def destinationOffset (r : Request) (h : InputContract r) (i : Fin r.N) (k lane : Nat) : Nat :=
  (Native.selected r h i).val*r.C+(k*512+lane)
def incomingOffset (r : Request) (i : Fin r.N) (k lane : Nat) : Nat :=
  i.val*r.C+(k*512+lane)

/-- The two source regions are views of one original byte-memory map. Only
active slice words are pinned; all other cells and registers remain arbitrary.
The current map includes preceding duplicate-row updates when one occurred. -/
def state (r : Request) (h : InputContract r) (m : Memory)
    (i : Fin r.N) (k : Nat) (surrounding : State Word) : State Word :=
  { surrounding with
    mem := fun region lane =>
      if lane < min 512 (r.C-k*512) then
        if region = Program.destinationRegion then
          Cell.mk .real (words m r.destination (destinationOffset r h i k lane))
        else if region = Program.incomingRegion then
          Cell.mk .real (words m r.incoming (incomingOffset r i k lane))
        else surrounding.mem region lane
      else surrounding.mem region lane }

def current (r : Request) (h : InputContract r) (objects : Objects r)
    (m : Memory) (i : Fin r.N) (k lane : Nat) : Word :=
  logicalValue (read m) objects.destination objects.destinationLaws
    (destinationOffset r h i k lane)
def incoming (r : Request) (objects : Objects r)
    (m : Memory) (i : Fin r.N) (k lane : Nat) : Word :=
  logicalValue (read m) objects.incoming objects.incomingLaws (incomingOffset r i k lane)

theorem inputs (r : Request) (h : InputContract r) (objects : Objects r)
    (m : Memory) (i : Fin r.N) (k : Nat) (surrounding : State Word) :
    KernelProof.InputValues (Program.layout r.C k) (state r h m i k surrounding)
      (current r h objects m i k) (incoming r objects m i k) := by
  intro lane active
  change lane < min 512 (r.C-k*512) at active
  have bound : k*512+lane<r.C := by omega
  let column : Fin r.C := ⟨k*512+lane,bound⟩
  have left := Physical.destination_direct r h objects m (Native.selected r h i,column)
  have right := Physical.incoming_direct r h objects m (i,column)
  constructor
  · simp only [state,Program.layout,if_pos active]
    exact left
  · simp only [state,Program.layout,if_pos active,
      if_neg (show Program.incomingRegion≠Program.destinationRegion by decide)]
    exact right

/-- Checked tensor extents bound every clamped lane address and the native
int64 lane conversion. Physical storage observations rule out pointer wrap. -/
theorem accesses (r : Request) (h : InputContract r) (storage : StorageBinding r)
    (i : Fin r.N) (k lane : Nat) (hk : k<(r.C+511)/512) (hl : lane<512) :
    let safe := if lane < (min 512 (r.C-k*512)) then lane else 0
    destinationOffset r h i k safe<r.R*r.C ∧ incomingOffset r i k safe<r.N*r.C ∧
    (BitVec.ofInt 64 (lane : Int)).toInt=(lane : Int) ∧
    (BitVec.ofNat 64 (Native.address r.destination 4 (destinationOffset r h i k safe))).toNat=
      Native.address r.destination 4 (destinationOffset r h i k safe) ∧
    (BitVec.ofNat 64 (Native.address r.incoming 4 (incomingOffset r i k safe))).toNat=
      Native.address r.incoming 4 (incomingOffset r i k safe) := by
  have bounds := (Native.all_indices r h i k lane hk hl).2.2
  dsimp only at bounds ⊢
  refine ⟨bounds.2.1,bounds.2.2.1,Native.source_lane_cast lane hl,?_,?_⟩
  · exact Native.pointer r.destination 4 (r.R*r.C) _ storage.native.destination (by decide) bounds.2.1
  · exact Native.pointer r.incoming 4 (r.N*r.C) _ storage.native.incoming (by decide) bounds.2.2.1

def SafeLane (r : Request) (h : InputContract r) (i : Fin r.N) (k lane : Nat) : Prop :=
  let safe := if lane < min 512 (r.C-k*512) then lane else 0
  destinationOffset r h i k safe<r.R*r.C ∧ incomingOffset r i k safe<r.N*r.C ∧
  (BitVec.ofInt 64 (lane : Int)).toInt=(lane : Int) ∧
  (BitVec.ofNat 64 (Native.address r.destination 4 (destinationOffset r h i k safe))).toNat=
    Native.address r.destination 4 (destinationOffset r h i k safe) ∧
  (BitVec.ofNat 64 (Native.address r.incoming 4 (incomingOffset r i k safe))).toNat=
    Native.address r.incoming 4 (incomingOffset r i k safe)

/-- The source's selected row agrees with actual int64 metadata bytes, and
every clamped pointer has a checked native representation. -/
def CheckedCorrespondence (r : Request) (h : InputContract r) (m : Memory)
    (i : Fin r.N) (k : Nat) : Prop :=
  k<(r.C+511)/512 ∧ ((Native.selected r h i).val : Int)=indexAt r m i ∧
    ∀ lane,lane<512 → SafeLane r h i k lane

theorem checked_correspondence (r : Request) (h : InputContract r) (m : Memory)
    (storage : StorageBinding r) (observed : IndexBinding r m)
    (i : Fin r.N) (k : Nat) (hk : k<(r.C+511)/512) :
    CheckedCorrespondence r h m i k := by
  have positive : 0<r.C := by omega
  refine ⟨hk,?_,fun lane hl => accesses r h storage i k lane hk hl⟩
  rw [←observed i i.isLt]
  exact (Native.selected_value r h i ⟨0,positive⟩).2

#axiomsClean inputs
#axiomsClean accesses
#axiomsClean checked_correspondence
end
end VeriTile.Bench.Repairs.IndexSelectBwd.KernelBinding
