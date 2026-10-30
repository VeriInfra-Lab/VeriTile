import bench.tritonbench_g.fast_layernorm.Proofs.KernelInputs
import bench.tritonbench_g.layernorm_fwd_triton.Proofs.Memory

/-! Narrow output-Tensor interpretation of the already proved regional source
stores. Placement contains addresses, finite store coverage and output/output
separation only. It contains no output values or execution-success premise. -/
namespace VeriTile.Bench.Repairs.FastLayernorm.KernelStorage
open VeriTile.Triton FP.Structural VeriTile.Torch.FP32
open VeriTile.Bench.Repairs.LayernormFwdTriton
noncomputable section
open scoped Classical

structure Placement (active : RegionName → Nat → Prop) where
  tensor : RegionName → Tensor
  laws : ∀ r,TensorLaws (tensor r)
  checked : ∀ r,Contract.TensorContract (tensor r).metadata Bool.true
  cells : List (RegionName × Nat)
  unique : cells.Nodup
  coverage : ∀ r i,(r,i)∈cells ↔ active r i
  bounds : ∀ r i,active r i → i<(tensor r).metadata.numel
  apart : ∀ a∈cells,∀ b∈cells,a≠b →
    Memory.Apart
      ((dataView (tensor a.1) (laws a.1) ((checked a.1).dense rfl)).address a.2)
      ((dataView (tensor b.1) (laws b.1) ((checked b.1).dense rfl)).address b.2)

def Placement.address (p : Placement active) (r : RegionName) (i : Nat) : Nat :=
  (dataView (p.tensor r) (p.laws r) ((p.checked r).dense rfl)).address i

theorem actual_pointer (p : Placement active) (r : RegionName) (nonempty : 0<(p.tensor r).metadata.numel) :
    BitVec.ofNat 64 (p.address r 0)=(p.tensor r).dataPtr :=
  VeriTile.Torch.FP32.actual_pointer _ _ _ nonempty

theorem logical_address (p : Placement active) (r : RegionName) (i : Nat) (hi : active r i) :
    (p.tensor r).storage.address ((p.tensor r).physicalIndex i)=p.address r i := by
  have bound := p.bounds r i hi
  rw [(p.laws r).contiguous ((p.checked r).dense rfl) i bound]
  simp only [Placement.address,DataView.address,DataView.base,dataView,
    if_neg (by omega : (p.tensor r).metadata.numel≠0),Storage.address]
  omega

variable {α β Store Byte : Type} [Inhabited α]
def writes (p : Placement active) (observe : RegionName → Nat → Cell α → β) (v : State α) : List (Memory.Write β) :=
  p.cells.map fun a => ⟨p.address a.1 a.2,observe a.1 a.2 (v.mem a.1 a.2)⟩
def commit (api : Memory.API β Store Byte) (p : Placement active) (observe : RegionName → Nat → Cell α → β)
    (v : State α) (initial : Store) : Store := Memory.run api (writes p observe v) initial

theorem commit_output (api : Memory.API β Store Byte) (p : Placement active) (observe : RegionName → Nat → Cell α → β)
    (v : State α) (initial : Store) (r : RegionName) (i : Nat) (hi : active r i) :
    api.read (commit api p observe v initial)
      ((p.tensor r).storage.address ((p.tensor r).physicalIndex i))=observe r i (v.mem r i) := by
  rw [logical_address p r i hi]
  apply Memory.run_readback
  · exact ⟨⟨p.address r i,observe r i (v.mem r i)⟩,
      List.mem_map.mpr ⟨(r,i),(p.coverage r i).mpr hi,rfl⟩,rfl,rfl⟩
  · intro e he
    obtain ⟨a,ha,rfl⟩ := List.mem_map.mp he
    by_cases eq : a=(r,i)
    · subst a
      exact Or.inl ⟨rfl,rfl⟩
    · exact Or.inr (p.apart a ha (r,i) ((p.coverage r i).mpr hi) eq)

theorem commit_frame (api : Memory.API β Store Byte) (p : Placement active) (observe : RegionName → Nat → Cell α → β)
    (v : State α) (initial : Store) (byte : Nat)
    (outside : ∀ r i,active r i → byte<p.address r i ∨ p.address r i+4≤byte) :
    api.byte (commit api p observe v initial) byte=api.byte initial byte := by
  apply Memory.run_byte_frame
  intro e he
  obtain ⟨a,ha,rfl⟩ := List.mem_map.mp he
  exact outside a.1 a.2 ((p.coverage a.1 a.2).mp ha)

def TensorImage (api : Memory.API β Store Byte) (p : Placement active) (observe : RegionName → Nat → Cell α → β)
    (v : State α) (initial : Store) : Prop :=
  (∀ r i,active r i → api.read (commit api p observe v initial)
    ((p.tensor r).storage.address ((p.tensor r).physicalIndex i))=observe r i (v.mem r i)) ∧
  (∀ byte,(∀ r i,active r i → byte<p.address r i ∨ p.address r i+4≤byte) →
    api.byte (commit api p observe v initial) byte=api.byte initial byte) ∧
  (∀ r,0<(p.tensor r).metadata.numel → BitVec.ofNat 64 (p.address r 0)=(p.tensor r).dataPtr)

theorem complete (api : Memory.API β Store Byte) (p : Placement active) (observe : RegionName → Nat → Cell α → β)
    (v : State α) (initial : Store) : TensorImage api p observe v initial :=
  ⟨commit_output api p observe v initial,commit_frame api p observe v initial,actual_pointer p⟩

/-- Attach the physical interpretation to the same constructed source state.
P is the full source execution/formula/memory/frame proposition at the call site. -/
theorem attach {H : Prop} {P : State α → Prop} (source : H ∧ ∃ v,P v)
    (api : Memory.API β Store Byte) (p : Placement active) (observe : RegionName → Nat → Cell α → β) (initial : Store) :
    H ∧ ∃ v,P v ∧ TensorImage api p observe v initial := by
  obtain ⟨h,v,hv⟩ := source
  exact ⟨h,v,hv,complete api p observe v initial⟩

theorem native_logical_output (api : Memory.API (BitVec 32) Store Byte) (p : Placement active)
    (observe : RegionName → Nat → Cell α → BitVec 32) (v : State α) (initial : Store)
    (r : RegionName) (i : Nat) (hi : active r i) :
    (logicalValue (fun a => (⟨api.read (commit api p observe v initial) a⟩ : VeriTile.Torch.FP32.Word))
      (p.tensor r) (p.laws r) i).bits=observe r i (v.mem r i) := by
  simp only [logicalValue,(p.checked r).noNeg,Bool.false_eq_true,if_false]
  exact commit_output api p observe v initial r i hi

theorem nativeAttach {H : Prop} {P : State α → Prop} (source : H ∧ ∃ v,P v)
    (api : Memory.API (BitVec 32) Store Byte) (p : Placement active)
    (observe : RegionName → Nat → Cell α → BitVec 32) (initial : Store) :
    H ∧ ∃ v,P v ∧ TensorImage api p observe v initial ∧
      (∀ r i,active r i →
        (logicalValue (fun a => (⟨api.read (commit api p observe v initial) a⟩ : VeriTile.Torch.FP32.Word))
          (p.tensor r) (p.laws r) i).bits=observe r i (v.mem r i)) := by
  obtain ⟨h,v,hv⟩ := source
  exact ⟨h,v,hv,complete api p observe v initial,native_logical_output api p observe v initial⟩

#axiomsClean commit_output
#axiomsClean commit_frame
#axiomsClean complete
#axiomsClean attach
#axiomsClean native_logical_output
#axiomsClean nativeAttach
end
end VeriTile.Bench.Repairs.FastLayernorm.KernelStorage
