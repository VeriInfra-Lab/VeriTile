import bench.tritonbench_g.chunk_gated_attention.Sources.Grid

/-! Exact launch-loop plan: base starts at zero, advances by the native x-grid
limit, and stops at total. The structural fuel is proved sufficient for every
natural total and never truncates the source loop. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.LaunchPlan

structure Item where
  base : Nat
  size : Nat
  deriving DecidableEq

def build (total : Nat) : Nat → Nat → List Item
  | 0,_ => []
  | fuel+1,base => if base < total then
      ⟨base,min Grid.limit (total-base)⟩ :: build total fuel (base+Grid.limit)
    else []

def plans (total : Nat) := build total total 0

def Valid (total : Nat) (p : Item) : Prop :=
  0 < p.size ∧ p.size ≤ Grid.limit ∧ p.base+p.size ≤ total

theorem member_valid (total fuel base : Nat) (p : Item) (hp : p ∈ build total fuel base) :
    Valid total p ∧ base ≤ p.base := by
  induction fuel generalizing base with
  | zero => simp [build] at hp
  | succ fuel ih =>
    simp only [build] at hp
    split_ifs at hp with hb
    · rcases List.mem_cons.mp hp with he|ht
      · subst p
        refine ⟨⟨(Grid.actual_grid_legal hb).1,Nat.min_le_left _ _,?_⟩,le_rfl⟩
        have := Nat.min_le_right Grid.limit (total-base)
        dsimp only
        omega
      · obtain ⟨hv,hb'⟩ := ih _ ht
        exact ⟨hv,by omega⟩
    · simp at hp

theorem ordered (total fuel base : Nat) :
    (build total fuel base).Pairwise (fun p q => p.base+p.size ≤ q.base) := by
  induction fuel generalizing base with
  | zero => simp [build]
  | succ fuel ih =>
    simp only [build]
    split_ifs with hb
    · rw [List.pairwise_cons]
      constructor
      · intro q hq
        have hl := (member_valid total fuel (base+Grid.limit) q hq).2
        have hm := Nat.min_le_left Grid.limit (total-base)
        exact le_trans (Nat.add_le_add_left hm base) hl
      · exact ih _
    · exact List.Pairwise.nil

theorem covers (total fuel base n : Nat)
    (hf : total ≤ base+fuel*Grid.limit) (hb : base ≤ n) (hn : n < total) :
    ∃ p, p ∈ build total fuel base ∧ p.base ≤ n ∧ n < p.base+p.size := by
  induction fuel generalizing base with
  | zero => simp only [Nat.zero_mul,Nat.add_zero] at hf; omega
  | succ fuel ih =>
    have hbt : base < total := lt_of_le_of_lt hb hn
    simp only [build,if_pos hbt]
    by_cases hfirst : n < base+min Grid.limit (total-base)
    · exact ⟨⟨base,min Grid.limit (total-base)⟩,List.mem_cons_self,hb,hfirst⟩
    · have hnext : base+Grid.limit ≤ n := by
        have hd : base+(total-base)=total := Nat.add_sub_of_le hbt.le
        omega
      have htf : total ≤ (base+Grid.limit)+fuel*Grid.limit := by
        simpa only [Nat.succ_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using hf
      obtain ⟨p,hp,hpn,hnp⟩ := ih (base+Grid.limit) htf hnext
      exact ⟨p,List.mem_cons_of_mem _ hp,hpn,hnp⟩

theorem plans_valid (total : Nat) : ∀ p ∈ plans total, Valid total p :=
  fun p hp => (member_valid total total 0 p hp).1

theorem plans_ordered (total : Nat) :
    (plans total).Pairwise (fun p q => p.base+p.size ≤ q.base) := ordered total total 0

theorem plans_cover (total n : Nat) (hn : n < total) :
    ∃ p, p ∈ plans total ∧ p.base ≤ n ∧ n < p.base+p.size := by
  apply covers total total 0 n
  · have h : 1 ≤ Grid.limit := by norm_num [Grid.limit]
    simpa using Nat.mul_le_mul_left total h
  · exact Nat.zero_le _
  · exact hn

theorem member_size (total fuel base : Nat) (p : Item) (hp : p ∈ build total fuel base) :
    p.size=min Grid.limit (total-p.base) := by
  induction fuel generalizing base with
  | zero => simp [build] at hp
  | succ fuel ih =>
    simp only [build] at hp
    split_ifs at hp with hb
    · rcases List.mem_cons.mp hp with rfl|ht
      · rfl
      · exact ih _ ht
    · simp at hp

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkGatedAttention.LaunchPlan
