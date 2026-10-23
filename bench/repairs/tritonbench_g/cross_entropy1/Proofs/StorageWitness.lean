import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Storage

namespace VeriTile.Bench.Repairs.CrossEntropy1.Physical.Witness
open VeriTile.Triton Source Host
noncomputable section

def old : RegionName := ⟨"retired-allocation"⟩
def fresh : RegionName := ⟨"new-allocation"⟩
def ownerRegion (generation : Nat) : RegionName := if generation=0 then old else fresh

/-- Eleven bytes deliberately leave a three-byte trailing partial word. -/
def buffer : Native.Buffer := ⟨16,11,0⟩

def locate (generation device address : Nat) : Option (RegionName × Nat × Fin 4) :=
  if device=0 ∧ 16≤address ∧ address<27 then
    some (ownerRegion generation,(address-16)/4,⟨(address-16)%4,by omega⟩)
  else none

def layout : Layout Nat where
  active := fun generation r => r=ownerRegion generation
  buffer := fun _ _ => buffer
  locate := locate
  other := fun _ _ _ => 0
  owner := by
    intro generation r i j active backed
    subst r
    have bound : 4*i+j.val<11 := backed
    have jbound := j.isLt
    have offset : (16+4*i+j.val)-16=4*i+j.val := by omega
    have quotient : (4*i+j.val)/4=i := by omega
    have remainder : (4*i+j.val)%4=j.val := by omega
    simp only [locate,buffer,show (0=0 ∧ 16≤16+4*i+j.val ∧ 16+4*i+j.val<27) from ⟨rfl,by omega,by omega⟩,
      ↓reduceIte,offset,quotient]
    apply congrArg some
    exact Prod.ext rfl (Prod.ext rfl (Fin.ext remainder))
  located := by
    intro generation device address r i j h
    by_cases inside : device=0 ∧ 16≤address ∧ address<27
    · simp only [locate,if_pos inside,Option.some.injEq,Prod.mk.injEq] at h
      obtain ⟨rfl,rfl,rfl⟩ := h
      refine ⟨rfl,inside.1,?_,?_⟩ <;> dsimp only [buffer]
      all_goals omega
    · simp only [locate,if_neg inside] at h
      cases h
  addressSpace := by intros; norm_num [buffer]
  aligned := by intros; norm_num [buffer]

def cell (i : Nat) : Source.Cell Word :=
  ⟨fun dtype => match dtype with
    | .real => if i=0 then 1 else if i=2 then 0x00ccbbaa else 0
    | .int => if i=0 then 1 else 0
    | .bool => Bool.false
    | other => FP.Structural.defaultValue other⟩

def state (generation : Nat) : World Word Nat :=
  ⟨fun _ i => cell i,fun r => r=old ∨ r=fresh,generation⟩

theorem simultaneous_int64_fp32_views :
    IntRead id (state 0) old 0 ∧ rawInt64 id (state 0) old 0=1 ∧
    rawWord id (state 0) old 0=1 ∧ rawWord id (state 0) old 1=0 ∧
    bytes layout id (state 0) 0 16=1 ∧ bytes layout id (state 0) 0 20=0 := by
  unfold IntRead
  decide +kernel

theorem byte_changing_store_and_partial_tail :
    bytes layout id {state 0 with memory:=Source.write (state 0).memory old 0 (.mk .real 0x01020304)} 0 16=4 ∧
    bytes layout id {state 0 with memory:=Source.write (state 0).memory old 0 (.mk .real 0x01020304)} 0 24=0xaa ∧
    bytes layout id {state 0 with memory:=Source.write (state 0).memory old 0 (.mk .real 0x01020304)} 0 25=0xbb ∧
    bytes layout id {state 0 with memory:=Source.write (state 0).memory old 0 (.mk .real 0x01020304)} 0 26=0xcc := by
  decide +kernel

theorem reuse_keeps_ghost_history :
    (state 1).live old ∧ (state 1).live fresh ∧
    ¬layout.active (state 1).native old ∧ layout.active (state 1).native fresh ∧
    layout.buffer (state 0).native old=layout.buffer (state 1).native fresh ∧
    bytes layout id {state 1 with memory:=Source.write (state 1).memory old 0 (.mk .real 123)} 0 16=1 ∧
    bytes layout id {state 1 with memory:=Source.write (state 1).memory fresh 0 (.mk .real 123)} 0 16=123 := by
  refine ⟨Or.inl rfl,Or.inr rfl,?_,rfl,rfl,?_,?_⟩
  · change old≠fresh
    decide +kernel
  · decide +kernel
  · decide +kernel

#axiomsClean layout
#axiomsClean simultaneous_int64_fp32_views
#axiomsClean byte_changing_store_and_partial_tail
#axiomsClean reuse_keeps_ghost_history
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Physical.Witness
