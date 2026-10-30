import bench.repairs.tritonbench_g.matmul_triton1.Proofs.Source

namespace VeriTile.Bench.Repairs.MatmulTriton1.Kernel
open VeriTile.Triton FP.Structural
open Revision001 SharedContract TensorBinding
noncomputable section

theorem data_address (t : Tensor) (laws : Laws t) (hc : t.metadata.isContiguous = Bool.true)
    (i : Nat) (hi : i < t.metadata.numel) :
    (VeriTile.Torch.FP32.dataView t laws hc).address i = address t i := by
  simp [VeriTile.Torch.FP32.DataView.address,VeriTile.Torch.FP32.DataView.base,
    VeriTile.Torch.FP32.dataView,address,base,show t.metadata.numel ≠ 0 by omega]

theorem address_storage (t : Tensor) (laws : Laws t) (hc : t.metadata.isContiguous = Bool.true)
    (i : Nat) (hi : i < t.metadata.numel) : storageContains t (address t i) := by
  have h := laws.inBounds i hi
  rw [laws.contiguous hc i hi] at h
  simp only [storageContains,address,base,VeriTile.Torch.FP32.Storage.address]
  omega

def logicalRead {α : Type} (read : Memory α) (t : Tensor) (i : Nat) : α :=
  read (t.storage.address (t.physicalIndex i))

theorem logicalRead_address {α : Type} (read : Memory α) (t : Tensor) (laws : Laws t)
    (hc : t.metadata.isContiguous = Bool.true) (i : Nat) (hi : i < t.metadata.numel) :
    logicalRead read t i = read (address t i) := by
  unfold logicalRead
  rw [laws.contiguous hc i hi]
  congr 1
  simp [address,base,VeriTile.Torch.FP32.Storage.address,Nat.mul_add,Nat.add_assoc]

theorem logicalRead_native (read : Memory Word) (t : Tensor) (laws : Laws t)
    (hn : t.metadata.isNeg = Bool.false) (i : Nat) :
    logicalRead read t i = VeriTile.Torch.FP32.logicalValue read t laws i := by
  simp [logicalRead,VeriTile.Torch.FP32.logicalValue,hn]

structure PhysicalBinding {α : Type} (x : Launch) (read : Memory α) (s : State α) : Prop where
  left : ∀ i, i < x.a.metadata.numel → s.mem x.ra i = Cell.mk .real (read (address x.a i))
  right : ∀ i, i < x.b.metadata.numel → s.mem x.rb i = Cell.mk .real (read (address x.b i))
  output : ∀ i, i < x.c.metadata.numel → s.mem x.rc i = Cell.mk .real (read (address x.c i))

theorem Binding.physical {α : Type} (x : Launch) (read : Memory Word) (decode : Word → α)
    (s : State α) (h : Binding x read decode s) : PhysicalBinding x (decode ∘ read) s := by
  refine ⟨?_,?_,h.output⟩
  · intro i hi
    simpa only [data_address x.a x.la x.admitted.contiguous.1 i hi] using h.inputs.left i hi
  · intro i hi
    simpa only [data_address x.b x.lb x.admitted.contiguous.2 i hi] using h.inputs.right i hi

/-- One physical memory for all actual Tensor ports. Values come only from
the executed source state. Selection depends on the output address window. -/
def image {α : Type} [Inhabited α] (x : Launch) (read : Memory α)
    (mem : RegionName → Nat → Cell α) : Memory α := fun q =>
  if base x.c ≤ q ∧ q < base x.c + 4*x.c.metadata.numel ∧ (q-base x.c)%4 = 0 then
    (mem x.rc ((q-base x.c)/4)).read .real
  else read q

theorem image_at {α : Type} [Inhabited α] (x : Launch) (read : Memory α)
    (mem : RegionName → Nat → Cell α) (i : Nat) (hi : i < x.c.metadata.numel) :
    image x read mem (address x.c i) = (mem x.rc i).read .real := by
  have hd : (address x.c i-base x.c)/4 = i := by unfold address; omega
  have hm : (address x.c i-base x.c)%4 = 0 := by unfold address; omega
  have bounds : base x.c ≤ address x.c i ∧ address x.c i < base x.c+4*x.c.metadata.numel := by
    unfold address; omega
  simp [image,bounds,hd,hm]

theorem image_frame {α : Type} [Inhabited α] (x : Launch) (read : Memory α)
    (mem : RegionName → Nat → Cell α) (q : Nat)
    (outside : q < base x.c ∨ base x.c+4*x.c.metadata.numel ≤ q) : image x read mem q = read q := by
  simp [image,show ¬(base x.c ≤ q ∧ q < base x.c+4*x.c.metadata.numel ∧ (q-base x.c)%4=0) by omega]

theorem image_initial {α : Type} [Inhabited α] (x : Launch) (read : Memory α)
    (s : State α) (h : PhysicalBinding x read s) : image x read s.mem = read := by
  funext q
  unfold image
  split
  · rename_i hit
    have hi : (q-base x.c)/4 < x.c.metadata.numel := by omega
    have eq : address x.c ((q-base x.c)/4) = q := by unfold address; omega
    rw [h.output _ hi,Cell.read_same,eq]
  · rfl

/-- The projection commutes with each actual output word store. -/
theorem image_write {α : Type} [Inhabited α] (x : Launch) (read : Memory α)
    (s : State α) (i : Nat) (hi : i < x.c.metadata.numel) (value : α) :
    image x read (s.write x.rc i (.mk .real value)).mem =
      Function.update (image x read s.mem) (address x.c i) value := by
  funext q
  by_cases hit : q = address x.c i
  · subst q
    rw [image_at x read _ i hi]
    simp [State.write]
  · rw [Function.update_of_ne hit]
    unfold image
    split
    · rename_i active
      have ne : (q-base x.c)/4 ≠ i := by intro eq; apply hit; unfold address; omega
      simp [State.write,ne]
    · rfl

theorem input_storage_frame {α : Type} [Inhabited α] (x : Launch) (read : Memory α)
    (mem : RegionName → Nat → Cell α) (t : Tensor) (member : t=x.a ∨ t=x.b)
    (q : Nat) (hq : storageContains t q) : image x read mem q = read q := by
  apply image_frame
  have sep := x.storageSeparate t member q
  by_contra outside
  exact sep (by omega) (by omega) hq

structure Realization {α : Type} (x : Launch) (read post : Memory α)
    (before final : State α) (value : Index x → α) : Prop where
  beforeBinding : PhysicalBinding x read before
  afterBinding : PhysicalBinding x post final
  outputs : ∀ i : Index x, logicalRead post x.c (offset x i) = value i
  inputs : ∀ t, t=x.a ∨ t=x.b → ∀ i, i<t.metadata.numel → logicalRead post t i = logicalRead read t i
  inputStorageFrame : ∀ t, t=x.a ∨ t=x.b → ∀ q, storageContains t q → post q=read q
  frame : ∀ q, q<base x.c ∨ base x.c+4*x.c.metadata.numel ≤ q → post q=read q

theorem realize {α : Type} [Inhabited α] (x : Launch) (read : Memory α)
    (before final : State α) (value : Index x → α) (bound : PhysicalBinding x read before)
    (result : Post x before final value) :
    Realization x read (image x read final.mem) before final value := by
  have frame := input_storage_frame x read final.mem
  refine ⟨bound,⟨?_,?_,?_⟩,?_,?_,frame,image_frame x read final.mem⟩
  · intro i hi
    rw [result.2 _ _ (Or.inl x.separateA),bound.left i hi,
      frame x.a (Or.inl rfl) _ (address_storage x.a x.la x.admitted.contiguous.1 i hi)]
  · intro i hi
    rw [result.2 _ _ (Or.inl x.separateB),bound.right i hi,
      frame x.b (Or.inr rfl) _ (address_storage x.b x.lb x.admitted.contiguous.2 i hi)]
  · intro n hn
    obtain ⟨i,rfl⟩ := index_covers x n hn
    rw [image_at x read final.mem _ hn,result.1 i,Cell.read_same]
  · intro i
    rw [logicalRead_address _ x.c x.lc x.outputContiguous _ (offset_bound x i),
      image_at x read final.mem _ (offset_bound x i),result.1 i,Cell.read_same]
  · intro t member i hi
    unfold logicalRead
    apply frame t member
    have hb : t.physicalIndex i < t.storage.elements := by
      rcases member with rfl | rfl
      · exact x.la.inBounds i hi
      · exact x.lb.inBounds i hi
    simp only [storageContains,VeriTile.Torch.FP32.Storage.address]
    omega

#axiomsClean image_initial
#axiomsClean image_write
#axiomsClean realize
end
end VeriTile.Bench.Repairs.MatmulTriton1.Kernel
