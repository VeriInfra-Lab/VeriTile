import bench.tritonbench_g.triton_mul2.Proofs.Execution

namespace VeriTile.Bench.Repairs.TritonMul2.Scheduling
noncomputable section
set_option maxHeartbeats 5000000
variable {α : Type}

def Touched (l : Layout) (pid cell : Nat) : Prop :=
  ∃ i,i < l.count ∧ pid*l.block ≤ i ∧ i < (pid+1)*l.block ∧ cell=l.output+i

theorem tile_unique (B p q i : Nat) (pl : p*B ≤ i) (pu : i < (p+1)*B)
    (ql : q*B ≤ i) (qu : i < (q+1)*B) : p=q := by
  rcases lt_trichotomy p q with pq | eq | qp
  · have h:=Nat.mul_le_mul_right B (by omega : p+1 ≤ q)
    omega
  · exact eq
  · have h:=Nat.mul_le_mul_right B (by omega : q+1 ≤ p)
    omega

theorem touched_disjoint (l : Layout) (p q cell : Nat) (different : p≠q)
    (hp : Touched l p cell) : ¬Touched l q cell := by
  obtain ⟨i,_,pl,pu,eq⟩:=hp
  rintro ⟨j,_,ql,qu,eq'⟩
  have same : i=j := by omega
  subst j
  exact different (tile_unique l.block p q i pl pu ql qu)

theorem other_input (ops : Operations α) (l : Layout) (allocation : Allocated l)
    (p q i : Nat) (different : p≠q) (hi : i < l.count)
    (lower : q*l.block ≤ i) (upper : i < (q+1)*l.block) (s : Memory α) :
    program ops l p s (l.input+i)=s (l.input+i) := by
  apply program_frame
  rintro ⟨j,hj,pl,pu,eq⟩
  rcases allocation.separation with same | separated
  · have ij : i=j := by omega
    subst j
    exact different (tile_unique l.block p q i pl pu lower upper)
  · exact separated j i hj hi eq.symm

/-- Distinct programs have disjoint read/write dependencies even in-place.
Thus their complete load/multiply/store phases commute without a numeric law. -/
theorem program_commute (ops : Operations α) (l : Layout) (allocation : Allocated l)
    (p q : Nat) (s : Memory α) :
    program ops l p (program ops l q s)=program ops l q (program ops l p s) := by
  by_cases same : p=q
  · subst q;rfl
  funext cell
  by_cases hp : Touched l p cell
  · have hq:=touched_disjoint l p q cell same hp
    obtain ⟨i,hi,lower,upper,rfl⟩:=hp
    rw [program_value ops l p _ i hi lower upper,program_frame ops l q _ _ hq,
      program_value ops l p _ i hi lower upper,
      other_input ops l allocation q p i (Ne.symm same) hi lower upper]
  · by_cases hq : Touched l q cell
    · obtain ⟨i,hi,lower,upper,rfl⟩:=hq
      rw [program_frame ops l p _ _ hp,program_value ops l q _ i hi lower upper,
        program_value ops l q _ i hi lower upper,other_input ops l allocation p q i same hi lower upper]
    · rw [program_frame ops l p _ _ hp,program_frame ops l q _ _ hq,
        program_frame ops l q _ _ hq,program_frame ops l p _ _ hp]

def execute (ops : Operations α) (l : Layout) (order : List Nat) (s : Memory α) :=
  order.foldl (fun memory pid=>program ops l pid memory) s

theorem range_prefix (ops : Operations α) (l : Layout) (s : Memory α) (count : Nat) :
    execute ops l (List.range count) s=runPrefix ops l s count := by
  induction count with
  | zero=>rfl
  | succ n ih=>simp only [execute,List.range_succ,List.foldl_append,List.foldl_cons,List.foldl_nil,runPrefix] at *;rw [ih]

/-- Every complete permutation of native programs gives the very same physical
memory. Cross-program interleavings are serializable because the access sets
proved above are disjoint; per-program vector loads precede that program's stores. -/
theorem order_equal (ops : Operations α) (l : Layout) (allocation : Allocated l)
    (order : List Nat) (permutation : order.Perm (List.range (grid l.count l.block))) (s : Memory α) :
    execute ops l order s=launch ops l s := by
  have commute : RightCommutative (fun (memory : Memory α) (pid : Nat)=>program ops l pid memory) :=
    ⟨fun memory p q=>program_commute ops l allocation q p memory⟩
  letI:=commute
  exact (permutation.foldl_eq s).trans (range_prefix ops l s _)

#axiomsClean touched_disjoint
#axiomsClean other_input
#axiomsClean program_commute
#axiomsClean order_equal
end
end VeriTile.Bench.Repairs.TritonMul2.Scheduling
