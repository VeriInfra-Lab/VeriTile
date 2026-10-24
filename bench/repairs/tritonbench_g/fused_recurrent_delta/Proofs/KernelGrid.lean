import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelGeometry
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelGrid
open VeriTile.Triton KernelProgram KernelRegisters KernelBackwardRegisters
noncomputable section
set_option maxHeartbeats 12000000

theorem value_unique (s : Shape) (BV bh bh' iv iv' t t' : Nat) (j j' : Fin BV)
    (ht : t<s.T) (ht' : t'<s.T) (hj : iv*BV+j.val<s.V) (hj' : iv'*BV+j'.val<s.V)
    (eq : valAddress s bh iv BV t j.val=valAddress s bh' iv' BV t' j'.val) : bh=bh' ∧ iv=iv' := by
  have flat : (bh*s.T+t)*s.V+(iv*BV+j.val)=(bh'*s.T+t')*s.V+(iv'*BV+j'.val) := by
    convert eq using 1 <;> simp only [valAddress] <;> ring
  obtain ⟨row,col⟩ := Geometry.mixed_injective s.V _ _ _ _ hj hj' flat
  exact ⟨(Geometry.mixed_injective s.T _ _ _ _ ht ht' row).1,
    (Geometry.mixed_injective BV _ _ _ _ j.isLt j'.isLt col).1⟩
theorem state_unique (s : Shape) (BV bh bh' iv iv' i i' : Nat) (j j' : Fin BV)
    (hi : i<s.K) (hi' : i'<s.K) (hj : iv*BV+j.val<s.V) (hj' : iv'*BV+j'.val<s.V)
    (eq : stateAddress s bh iv BV i j.val=stateAddress s bh' iv' BV i' j'.val) : bh=bh' ∧ iv=iv' := by
  have flat : (bh*s.K+i)*s.V+(iv*BV+j.val)=(bh'*s.K+i')*s.V+(iv'*BV+j'.val) := by
    convert eq using 1 <;> simp only [stateAddress] <;> ring
  obtain ⟨row,col⟩ := Geometry.mixed_injective s.V _ _ _ _ hj hj' flat
  exact ⟨(Geometry.mixed_injective s.K _ _ _ _ hi hi' row).1,
    (Geometry.mixed_injective BV _ _ _ _ j.isLt j'.isLt col).1⟩
theorem plane_unique (s : Shape) (p q : Program .backward s)
    (eq : plane s p.head p.value=plane s q.head q.value) : p.head=q.head ∧ p.value=q.value := by
  have flat : p.value*s.BH+p.head=q.value*s.BH+q.head := by
    simpa only [plane,Shape.BH,Nat.mul_assoc,Nat.add_comm] using eq
  have h := Geometry.mixed_injective s.BH _ _ _ _ p.headInside q.headInside flat
  exact ⟨h.2,h.1⟩
theorem key_unique (s : Shape) (p q : Program .backward s) (t t' i i' : Nat)
    (ht : t<s.T) (ht' : t'<s.T) (hi : i<s.K) (hi' : i'<s.K)
    (eq : keyAddress s (plane s p.head p.value) t i=keyAddress s (plane s q.head q.value) t' i') :
    p.head=q.head ∧ p.value=q.value := by
  have flat : (plane s p.head p.value*s.T+t)*s.K+i=(plane s q.head q.value*s.T+t')*s.K+i' := by
    convert eq using 1 <;> simp only [keyAddress] <;> ring
  have rows := (Geometry.mixed_injective s.K _ _ _ _ hi hi' flat).1
  exact plane_unique s p q (Geometry.mixed_injective s.T _ _ _ _ ht ht' rows).1

theorem forward_unique (s : Shape) (f : Flags) (p q : Program .forward s) (r : RegionName) (off : Nat)
    (a : KernelGeometry.Writes .forward s f p r off) (b : KernelGeometry.Writes .forward s f q r off) :
    p.head=q.head ∧ p.value=q.value := by
  rcases a with ⟨role,t,j,ht,hj,eq⟩|⟨enabled,role,i,j,hi,hj,eq⟩
  · rcases b with ⟨role',t',j',ht',hj',eq'⟩|⟨_,role',i',j',hi',hj',eq'⟩
    · exact value_unique s s.FBV p.head q.head p.value q.value t t' j j' ht ht' hj hj' (eq.symm.trans eq')
    · rcases role with role|role <;> simp_all
  · rcases b with ⟨role',t',j',ht',hj',eq'⟩|⟨_,role',i',j',hi',hj',eq'⟩
    · rcases role' with role'|role' <;> simp_all
    · exact state_unique s s.FBV p.head q.head p.value q.value i.val i'.val j j' hi hi' hj hj' (eq.symm.trans eq')

theorem backward_unique (s : Shape) (f : Flags) (p q : Program .backward s) (r : RegionName) (off : Nat)
    (a : KernelGeometry.Writes .backward s f p r off) (b : KernelGeometry.Writes .backward s f q r off) :
    p.head=q.head ∧ p.value=q.value := by
  rcases a with ka|va|ba|sa
  · obtain ⟨role,t,i,ht,hi,eq⟩ := ka
    rcases b with ⟨_,t',i',ht',hi',eq'⟩|⟨role',rest⟩|⟨role',rest⟩|⟨_,role',rest⟩
    · exact key_unique s p q t t' i.val i'.val ht ht' hi hi' (eq.symm.trans eq')
    all_goals rcases role with role|role <;> simp_all
  · obtain ⟨role,t,j,ht,hj,eq⟩ := va
    rcases b with ⟨role',rest⟩|⟨_,t',j',ht',hj',eq'⟩|⟨role',rest⟩|⟨_,role',rest⟩
    · rcases role' with role'|role' <;> simp_all
    · exact value_unique s s.BBV p.head q.head p.value q.value t t' j j' ht ht' hj hj' (eq.symm.trans eq')
    · simp_all
    · simp_all
  · obtain ⟨role,data⟩ := ba
    rcases b with ⟨role',rest⟩|⟨role',rest⟩|⟨_,data'⟩|⟨_,role',rest⟩
    · rcases role' with role'|role' <;> simp_all
    · simp_all
    · cases hb : f.vectorBeta <;> simp only [hb,Bool.false_eq_true,ite_false,ite_true] at data data'
      · obtain ⟨t,ht,eq⟩ := data
        obtain ⟨t',ht',eq'⟩ := data'
        exact plane_unique s p q (Geometry.mixed_injective s.T _ _ _ _ ht ht' (eq.symm.trans eq')).1
      · obtain ⟨t,j,ht,hj,eq⟩ := data
        obtain ⟨t',j',ht',hj',eq'⟩ := data'
        have h := value_unique s s.BBV (plane s p.head p.value) (plane s q.head q.value)
          p.value q.value t t' j j' ht ht' hj hj' (eq.symm.trans eq')
        exact plane_unique s p q h.1
    · simp_all
  · obtain ⟨enabled,role,i,j,hi,hj,eq⟩ := sa
    rcases b with ⟨role',rest⟩|⟨role',rest⟩|⟨role',rest⟩|⟨_,_,i',j',hi',hj',eq'⟩
    · rcases role' with role'|role' <;> simp_all
    · simp_all
    · simp_all
    · exact state_unique s s.BBV p.head q.head p.value q.value i.val i'.val j j' hi hi' hj hj' (eq.symm.trans eq')

theorem unique_program (kind : Kind) (s : Shape) (f : Flags) (p q : Program kind s) (r : RegionName) (off : Nat)
    (a : KernelGeometry.Writes kind s f p r off) (b : KernelGeometry.Writes kind s f q r off) : p=q := by
  have eq : p.head=q.head ∧ p.value=q.value := by
    cases kind
    · exact forward_unique s f p q r off a b
    · exact backward_unique s f p q r off a b
  cases p; cases q; cases eq.1; cases eq.2; rfl

#axiomsClean forward_unique
#axiomsClean backward_unique
#axiomsClean unique_program
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelGrid
