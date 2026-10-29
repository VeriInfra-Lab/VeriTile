import bench.repairs.tritonbench_g.chunk_bwd_dqkg.HostContract
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.WholeFlat

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Direct
open VeriTile.Triton Whole
noncomputable section
open Classical
set_option maxHeartbeats 3000000

def ofRegion (r : RegionName) : Role :=
  if r="q" then .q else if r="k" then .k else if r="v" then .v else if r="h" then .h
  else if r="g" then .g else if r="do" then .dout else if r="dh" then .dh
  else if r="dq" then .dq else if r="dk" then .dk else .dg

@[simp] theorem ofRegion_region (r : Role) : ofRegion r.region=r := by
  cases r <;> simp [ofRegion,Role.region]

theorem region_of_mem (r : RegionName) (hr : r∈regions) : (ofRegion r).region=r := by
  rw [←region_list] at hr
  obtain ⟨role,_,rfl⟩ := List.mem_map.mp hr
  simp

/-- Ordinary fresh allocator facts. Only fresh output identities and storage
are supplied here; input separation is derived from the executable check. -/
structure Allocation (p : Metadata) where
  objects : Role → Nat
  freshPointer : Role → Nat
  aligned : ∀ r,¬IsInput r → freshPointer r%4=0
  fits : ∀ r,¬IsInput r → freshPointer r+4*extent (config p) r.region≤2^64
  freshIdentity : ∀ r s,¬IsInput r → r≠s → objects r≠objects s
  freshSeparate : ∀ r s,¬IsInput r → r≠s →
    freshPointer r+4*extent (config p) r.region≤(if IsInput s then base p s else freshPointer s) ∨
    (if IsInput s then base p s else freshPointer s)+4*extent (config p) s.region≤freshPointer r

def pointer (p : Metadata) (a : Allocation p) (r : Role) :=
  if IsInput r then base p r else a.freshPointer r

def OutputGuard (p : Metadata) (a : Allocation p) : Prop := Nonempty p →
  ∀ r,¬IsInput r → pointer p a r+4*envelope (config p)<2^64

def checkOutputs (p : Metadata) (a : Allocation p) : Except InputError Unit :=
  if OutputGuard p a then .ok () else .error .pointer

theorem checkOutputs_iff (p : Metadata) (a : Allocation p) :
    checkOutputs p a=.ok () ↔ OutputGuard p a := by simp [checkOutputs,ite_eq_iff]

theorem empty_extent (p : Metadata) (he : ¬Nonempty p) (r : RegionName) : extent (config p) r=0 := by
  have empty : (config p).B=0 ∨ (config p).H=0 ∨ (config p).T=0 := by
    unfold Nonempty at he
    omega
  rcases empty with h|h|h <;> simp [extent,Config.heads,Config.NT,h]

theorem pointer_separate (p : Metadata) (h : InputContract p) (a : Allocation p)
    (r s : Role) (ne : r≠s) :
    pointer p a r+4*extent (config p) r.region≤pointer p a s ∨
    pointer p a s+4*extent (config p) s.region≤pointer p a r := by
  by_cases ri : IsInput r
  · by_cases si : IsInput s
    · by_cases he : Nonempty p
      · simpa [pointer,ri,si,tensor_size p h r ri,tensor_size p h s si] using h.separate he r s ri si ne
      · simp only [empty_extent p he,Nat.mul_zero,Nat.add_zero]
        omega
    · have hs := a.freshSeparate s r si ne.symm
      simpa only [pointer,if_pos ri,if_neg si,or_comm] using hs
  · simpa only [pointer,if_neg ri] using a.freshSeparate r s ri ne

def flat (p : Metadata) (a : Allocation p) : FlatAlloc where
  flat := "heap"
  regions := regions
  base := fun r => pointer p a (ofRegion r)/4
  extent := extent (config p)

theorem denseAllocation (p : Metadata) (h : InputContract p) (a : Allocation p) :
    DenseAllocation (config p) (flat p a) := by
  refine ⟨rfl,rfl,?_⟩
  intro r hr s hs ne
  change r∈regions at hr
  change s∈regions at hs
  have different : ofRegion r≠ofRegion s := by
    intro eq
    have same := congrArg Role.region eq
    rw [region_of_mem r hr,region_of_mem s hs] at same
    exact ne same
  have disjoint := pointer_separate p h a (ofRegion r) (ofRegion s) different
  rw [region_of_mem r hr,region_of_mem s hs] at disjoint
  change pointer p a (ofRegion r)/4+extent (config p) r≤pointer p a (ofRegion s)/4 ∨
    pointer p a (ofRegion s)/4+extent (config p) s≤pointer p a (ofRegion r)/4
  omega

theorem pointer_aligned (p : Metadata) (h : InputContract p) (a : Allocation p)
    (r : Role) (positive : 0<extent (config p) r.region) : pointer p a r%4=0 := by
  by_cases ri : IsInput r
  · simp only [pointer,if_pos ri]
    apply (h.tensors r ri).2.2.2.2.2.2.2.2.2
    rw [tensor_size p h r ri]
    exact positive
  · simpa [pointer,ri] using a.aligned r ri

theorem output_base (p : Metadata) (a : Allocation p) (r : Role) (out : ¬IsInput r) :
    4*(flat p a).base r.region=pointer p a r := by
  have aligned := a.aligned r out
  simp only [flat,ofRegion_region,pointer,if_neg out]
  omega

def sourceLayout (p : Metadata) : RealMath.Layout :=
  {(config p).layout with
    s_k_h := (p.strides .q).getD 1 0
    s_k_t := (p.strides .q).getD 2 0
    s_v_h := (p.strides .v).getD 1 0
    s_v_t := (p.strides .v).getD 2 0
    s_h_h := (p.strides .dh).getD 1 0
    s_h_t := (p.strides .dh).getD 2 0}

theorem sourceLayout_eq (p : Metadata) (h : InputContract p) (he : Nonempty p) :
    sourceLayout p=(config p).layout := by
  have q := (h.tensors .q (by simp [IsInput,inputRoles])).2.1
  have v := (h.tensors .v (by simp [IsInput,inputRoles])).2.1
  have dh := (h.tensors .dh (by simp [IsInput,inputRoles])).2.1
  have ht : max (config p).T 1=(config p).T := Nat.max_eq_left he.2.2
  have hk : max (config p).K 1=(config p).K := Nat.max_eq_left h.key
  have hv : max (config p).V 1=(config p).V := Nat.max_eq_left h.value
  have nt : 0<(config p).NT := by
    have := he.2.2
    unfold Config.NT
    omega
  have hist : max ((config p).NT*(config p).K) 1=(config p).NT*(config p).K :=
    Nat.max_eq_left (Nat.mul_pos nt h.key)
  simp [sourceLayout,q,v,dh,shape,canonicalStrides,ht,hk,hv,hist,Config.layout,Nat.mul_assoc]

theorem formed_safe (p : Metadata) (h : InputContract p) (a : Allocation p)
    (he : Nonempty p) (outputs : OutputGuard p a) (r : Role) (off : Nat)
    (bound : off<envelope (config p)) : pointer p a r+4*off<2^64 := by
  apply formed_pointer64 (config p) _ _ _ bound
  by_cases ri : IsInput r
  · simpa [pointer,ri] using h.pointers he r ri
  · exact outputs he r ri

#axiomsClean denseAllocation
#axiomsClean sourceLayout_eq
#axiomsClean formed_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Direct
