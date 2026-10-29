import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperTensorMemory
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperAllocated

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper
noncomputable section
set_option maxHeartbeats 2000000
variable {S Key Value : Type} {world : World S Key Value}

/-- Disjoint byte spans, not merely unequal Python object IDs. Empty tensors
have no cells and may use the same allocator sentinel. -/
def ByteApart (a b : Descriptor) : Prop :=
  ∀ i j, i<a.extent*a.width → j<b.extent*b.width → a.base+i≠b.base+j

theorem ByteApart.symm {a b : Descriptor} (h : ByteApart a b) : ByteApart b a :=
  fun i j hi hj he => h j i hj hi he.symm

theorem ByteApart.cells {a b : Descriptor} (h : ByteApart a b)
    (i j : Nat) (hi : i<a.extent) (hj : j<b.extent) :
    LayerNormFwd.Address.byteAddress a i≠LayerNormFwd.Address.byteAddress b j :=
  h (i*a.width) (j*b.width) (Nat.mul_lt_mul_of_pos_right hi a.width_pos)
    (Nat.mul_lt_mul_of_pos_right hj b.width_pos)

def Separate (store : TensorStore S Descriptor Value) (s : S) (a b : Tensor) : Prop :=
  ByteApart (store.descriptor s a) (store.descriptor s b)

theorem Separate.transfer {store : TensorStore S Descriptor Value} {s t : S} {a b : Tensor}
    (h : Separate store s a b) (ha : store.present s a) (hb : store.present s b)
    (he : store.Extends s t) : Separate store t a b := by
  unfold Separate
  rw [(he a ha).2.1,(he b hb).2.1]
  exact h

/-- One ordinary allocator call creates storage disjoint from every live
predecessor tensor, including aliases not mentioned in the request's ID list. -/
structure FreshAllocationAPI (store : TensorStore S Descriptor Value) (api : TensorAPI world) : Prop where
  fresh : ∀ role request s output, ((api.allocate role request).run s).2=.ok output →
    ∀ old, store.present s old → Separate store ((api.allocate role request).run s).1 output old

/-- A prefix of actual successful allocations. The initial live tensors and
all newly allocated outputs persist; fresh outputs are pairwise byte-disjoint. -/
structure AllocationBatch (store : TensorStore S Descriptor Value) (initial : S)
    (outputs : List Tensor) (current : S) : Prop where
  preserved : store.Extends initial current
  present : ∀ output ∈ outputs, store.present current output
  prior : ∀ output ∈ outputs, ∀ old, store.present initial old → Separate store current output old
  pairwise : outputs.Pairwise (Separate store current)

theorem AllocationBatch.empty (store : TensorStore S Descriptor Value) (s : S) :
    AllocationBatch store s [] s :=
  ⟨store.extends_refl s,by simp,by simp,List.Pairwise.nil⟩

theorem AllocationBatch.transfer {store : TensorStore S Descriptor Value} {initial s t : S}
    {outputs : List Tensor} (h : AllocationBatch store initial outputs s) (he : store.Extends s t) :
    AllocationBatch store initial outputs t := by
  refine ⟨h.preserved.trans he,fun output ho => (he output (h.present output ho)).1,?_,?_⟩
  · intro output ho old hp
    exact (h.prior output ho old hp).transfer (h.present output ho) (h.preserved old hp).1 he
  · apply List.Pairwise.imp_of_mem _ h.pairwise
    intro a b ha hb hab
    exact hab.transfer (h.present a ha) (h.present b hb) he

theorem AllocationBatch.append {store : TensorStore S Descriptor Value} {initial s t : S}
    {outputs : List Tensor} (h : AllocationBatch store initial outputs s) (he : store.Extends s t)
    (output : Tensor) (hp : store.present t output)
    (hf : ∀ old, store.present s old → Separate store t output old) :
    AllocationBatch store initial (outputs++[output]) t := by
  have ht := h.transfer he
  refine ⟨ht.preserved,?_,?_,?_⟩
  · intro tensor hm
    rcases List.mem_append.mp hm with hm | hm
    · exact ht.present tensor hm
    · have htensor : tensor=output := List.mem_singleton.mp hm
      rw [htensor]
      exact hp
  · intro tensor hm old ho
    rcases List.mem_append.mp hm with hm | hm
    · exact ht.prior tensor hm old ho
    · have htensor : tensor=output := List.mem_singleton.mp hm
      rw [htensor]
      exact hf old (h.preserved old ho).1
  · apply List.pairwise_append.mpr
    refine ⟨ht.pairwise,by simp,?_⟩
    intro a ha b hb
    have hb : b=output := List.mem_singleton.mp hb
    rw [hb]
    exact (hf a (h.present a ha)).symm

theorem allocate_batch (store : TensorStore S Descriptor Value) (api : TensorAPI world)
    (hs : StorageAPI store api) (hf : FreshAllocationAPI store api) (role : Wrapper.Role)
    (request : Wrapper.Allocation) (initial s : S) (outputs : List Tensor)
    (hb : AllocationBatch store initial outputs s) (output : Tensor)
    (h : ((call (.allocate role) (api.allocate role request)).run s).result=.ok output) :
    AllocationBatch store initial (outputs++[output])
      ((call (.allocate role) (api.allocate role request)).run s).state := by
  have hh := call_success _ _ _ _ h
  exact hb.append (hs.allocate_preserves role request s) output
    (hs.allocate_present role request s output hh) (hf.fresh role request s output hh)

def optionTensors : Option Tensor → List Tensor | none => [] | some tensor => [tensor]

theorem allocateOptional_batch (store : TensorStore S Descriptor Value) (api : TensorAPI world)
    (hs : StorageAPI store api) (hf : FreshAllocationAPI store api) (role : Wrapper.Role)
    (condition : Bool) (request : Wrapper.Allocation) (assertStride : Bool)
    (initial s : S) (outputs : List Tensor) (hb : AllocationBatch store initial outputs s)
    (output : Option Tensor)
    (h : ((allocateOptional api role condition request assertStride).run s).result=.ok output) :
    AllocationBatch store initial (outputs++optionTensors output)
      ((allocateOptional api role condition request assertStride).run s).state := by
  cases condition
  · have ho : output=none := by simpa [allocateOptional,Pure.pure,Action.pure] using h.symm
    simpa [ho,allocateOptional,optionTensors,Pure.pure,Action.pure] using hb
  · unfold allocateOptional at h ⊢
    simp only [Bool.true_eq,ite_true,Bind.bind] at h ⊢
    obtain ⟨tensor,htensor,hrest,hstate,_⟩ := bind_success _ _ _ _ h
    have ht := allocate_batch store api hs hf role request initial s outputs hb tensor htensor
    cases hg : assertStride
    · simp only [hg,Bool.false_eq_true,ite_false] at hrest hstate ⊢
      obtain ⟨_,_,hfinal,hstate2,_⟩ := bind_success _ _ _ _ hrest
      have ho : output=some tensor := by simpa [Pure.pure,Action.pure] using hfinal.symm
      rw [ho,hstate,hstate2]
      exact ht
    · simp only [hg,ite_true] at hrest hstate ⊢
      obtain ⟨_,_,hfinal,hstate2,_⟩ := bind_success _ _ _ _ hrest
      have ho : output=some tensor := by simpa [Pure.pure,Action.pure] using hfinal.symm
      rw [ho,hstate,hstate2]
      exact ht.transfer (persists_checkStride store role tensor _)

#axiomsClean ByteApart.cells
#axiomsClean AllocationBatch.transfer
#axiomsClean AllocationBatch.append
#axiomsClean allocate_batch
#axiomsClean allocateOptional_batch
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
