import bench.tritonbench_g.layernorm_fwd_triton.Proofs.Storage
import bench.tritonbench_g.fast_layernorm.Proofs.Model

/-! Actual untyped storage and arbitrary logical shape. The caller union
retains each complete backing, including aliasing views and odd byte tails. -/
namespace VeriTile.Bench.Repairs.FastLayernorm.Storage
open VeriTile.Bench.Repairs.LayernormFwdTriton
abbrev Backing := VeriTile.Bench.Repairs.LayernormFwdTritonRevision005.Storage.Backing
abbrev Tensor := VeriTile.Bench.Repairs.LayernormFwdTritonRevision005.Storage.Tensor
abbrev Key := Nat × Nat
abbrev World (Store : Type) := Nat → Store
noncomputable section
open scoped Classical

def coordinates : List Nat → Nat → List Nat
  | [],_ => []
  | n::ns,i => i/ns.prod :: coordinates ns (i%ns.prod)

theorem coordinates_valid (shape : List Nat) (i : Nat) (hi : i<shape.prod) :
    List.Forall₂ (fun i n => i<n) (coordinates shape i) shape := by
  induction shape generalizing i with
  | nil => exact .nil
  | cons n ns ih =>
    have hp : 0<ns.prod := by
      simp only [List.prod_cons] at hi
      by_contra h
      have hz : ns.prod=0 := by omega
      simp [hz] at hi
    refine .cons ?_ (ih _ (Nat.mod_lt _ hp))
    exact (Nat.div_lt_iff_lt_mul hp).mpr (by simpa [List.prod_cons] using hi)

def contains (s : Backing) (key : Key) : Prop := s.contains key

def caller (tensors : List Tensor) (key : Key) : Prop := ∃ t∈tensors,contains t.backing key

def observe (api : Memory.API α Store Byte) (m : World Store) (key : Key) : Byte :=
  api.byte (m key.1) key.2

def logical (api : Memory.API α Store Byte) (neg : α → α) (m : World Store) (t : Tensor) (i : Nat) : α :=
  Torch.logical api neg (m t.view.device) t.view (coordinates t.view.shape i)

def realLogical (m : World (Nat → ℝ)) (t : Tensor) (i : Nat) : ℝ :=
  Torch.realLogical (m t.view.device) t.view (coordinates t.view.shape i)

def contiguousStrides : List Nat → List Nat
  | [] => []
  | _::shape => (shape.map (fun n => max 1 n)).prod :: contiguousStrides shape

/-- A contiguous logical view retains its actual data pointer, original
backing and offset. Singleton strides need not be canonical. -/
structure Dense where
  tensor : Tensor
  count : Nat
  countEq : count=tensor.view.shape.prod
  aligned : tensor.view.allocation.base%4=0
  positive : tensor.view.negative=Bool.false
  lower : 0<count → tensor.backing.base≤tensor.view.allocation.base
  span : tensor.view.allocation.base+count*4≤tensor.backing.base+tensor.backing.nbytes
  coordinates : ∀ i, i<count →
    Torch.address tensor.view (Storage.coordinates tensor.view.shape i)=tensor.view.allocation.base+4*i

abbrev Dense.owner (d : Dense) := d.tensor.backing.owner
def Dense.address (d : Dense) (i : Nat) := d.tensor.view.allocation.base+4*i

theorem dense_logical (api : Memory.API α Store Byte) (neg : α → α)
    (m : World Store) (d : Dense) (i : Nat) (hi : i<d.count) :
    logical api neg m d.tensor i=api.read (m d.owner) (d.address i) := by
  simp only [logical,Torch.logical,d.positive,Bool.false_eq_true,if_false,d.coordinates i hi,d.tensor.owner,Dense.address]

theorem dense_real (m : World (Nat → ℝ)) (d : Dense) (i : Nat) (hi : i<d.count) :
    realLogical m d.tensor i=m d.owner (d.address i) := by
  simp only [realLogical,Torch.realLogical,d.positive,Bool.false_eq_true,if_false,d.coordinates i hi,d.tensor.owner,Dense.address]

theorem dense_bytes (d : Dense) (i byte : Nat) (hi : i<d.count) (hb : byte<4) :
    contains d.tensor.backing (d.owner,d.address i+byte) := by
  have hs := d.span
  have lower := d.lower (by omega)
  unfold contains VeriTile.Bench.Repairs.LayernormFwdTritonRevision005.Storage.Backing.contains Dense.address Dense.owner
  exact ⟨rfl,by omega,by omega⟩

theorem dense_aligned (d : Dense) (i : Nat) : d.address i%4=0 := by
  simp [Dense.address,Nat.add_mod,d.aligned]

theorem dense_count_bound (d : Dense) : d.count≤2^62 := by
  have hs := d.span
  have hf := d.tensor.backing.finite
  norm_num at *
  omega

theorem dense_address_bound (d : Dense) (i : Nat) (hi : i<d.count) : d.address i+4≤2^64 := by
  have hs := d.span
  have hf := d.tensor.backing.finite
  unfold Dense.address
  omega

theorem dense_apart (d : Dense) (i j : Nat) (hne : i≠j) : Memory.Apart (d.address i) (d.address j) := by
  unfold Memory.Apart Dense.address
  omega

/-- Allocator freshness is over full owner-tagged byte windows. -/
def Disjoint (a b : Backing) : Prop :=
  a.owner≠b.owner ∨ a.base+a.nbytes≤b.base ∨ b.base+b.nbytes≤a.base

theorem disjoint_symm (a b : Backing) (h : Disjoint a b) : Disjoint b a := by
  rcases h with h|h|h
  · exact Or.inl (Ne.symm h)
  · exact Or.inr (Or.inr h)
  · exact Or.inr (Or.inl h)

theorem disjoint_byte (a : Dense) (b : Backing) (h : Disjoint a.tensor.backing b)
    (i : Nat) (hi : i<a.count) (key : Key) (hb : contains b key) (owner : key.1=a.owner) :
    key.2<a.address i ∨ a.address i+4≤key.2 := by
  have hs := a.span
  have lower := a.lower (by omega)
  rcases hb with ⟨ho,hlo,hhi⟩
  unfold Disjoint at h
  unfold Dense.address Dense.owner at *
  rcases h with h|h|h <;> omega

def runWorld (api : Memory.API α Store Byte) (owner : Nat) (writes : List (Memory.Write α))
    (m : World Store) : World Store := Function.update m owner (Memory.run api writes (m owner))

theorem runWorld_frame (api : Memory.API α Store Byte) (owner : Nat) (writes : List (Memory.Write α))
    (m : World Store) (key : Key)
    (outside : key.1=owner → ∀ e∈writes,key.2<e.address ∨ e.address+4≤key.2) :
    observe api (runWorld api owner writes m) key=observe api m key := by
  by_cases h : key.1=owner
  · subst owner
    simpa [observe,runWorld] using Memory.run_byte_frame api writes (m key.1) key.2 (outside rfl)
  · simp [observe,runWorld,h]

theorem runWorld_readback (api : Memory.API α Store Byte) (owner : Nat) (writes : List (Memory.Write α))
    (m : World Store) (address : Nat) (value : α)
    (present : ∃ e∈writes,e.address=address ∧ e.value=value)
    (only : ∀ e∈writes,e.address=address ∧ e.value=value ∨ Memory.Apart e.address address) :
    api.read (runWorld api owner writes m owner) address=value := by
  simpa [runWorld] using Memory.run_readback api writes (m owner) address value present only

theorem caller_odd_tail (api : Memory.API α Store Byte) (tensors : List Tensor)
    (before after : World Store)
    (frame : ∀ key,caller tensors key → observe api after key=observe api before key)
    (t : Tensor) (ht : t∈tensors) (offset : Nat) (ho : offset<t.backing.nbytes) :
    observe api after (t.backing.owner,t.backing.base+offset)=
    observe api before (t.backing.owner,t.backing.base+offset) := by
  exact frame _ ⟨t,ht,rfl,by omega,by omega⟩

theorem matrix_row_stride (d : Dense) (rowCount N : Nat)
    (shape : d.tensor.view.shape=[rowCount,N]) (positive : 0<N)
    (canonical : d.tensor.view.strides=contiguousStrides d.tensor.view.shape) : d.tensor.view.strides.headD 0=N := by
  rw [canonical,shape]
  simp [contiguousStrides,max_eq_right (show 1≤N by omega)]

#axiomsClean matrix_row_stride
#axiomsClean coordinates_valid
#axiomsClean dense_aligned
#axiomsClean dense_count_bound
#axiomsClean dense_address_bound
#axiomsClean runWorld_frame
#axiomsClean runWorld_readback
#axiomsClean caller_odd_tail
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.Storage
