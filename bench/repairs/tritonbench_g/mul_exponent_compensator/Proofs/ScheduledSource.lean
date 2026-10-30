import bench.repairs.tritonbench_g.mul_exponent_compensator.IO
import bench.repairs.tritonbench_g.triton_mul2.Proofs.Scheduling

namespace VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.ScheduledSource
open VeriTile.Triton FP.Structural Interface
open VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs (programState gridCount)
noncomputable section
variable {α : Type} [Inhabited α]

/-- Exact memory effect of one already-proved source program. -/
def update (M : Algebra α) (p : Parameters) (pid : Nat)
    (mem : RegionName → Nat → Cell α) : RegionName → Nat → Cell α :=
  fun r i => if r=p.output ∧ pid*p.block ≤ i ∧ i<(pid+1)*p.block ∧ i<p.n
    then resultCell M p mem i else mem r i

theorem expected_update (M : Algebra α) (p : Parameters) (s : State α) :
    expectedMemory M p s=update M p (s.pids 0) s.mem := by
  funext r i
  rw [expected_cell]
  simp only [update,Nat.add_mul,Nat.one_mul]

theorem commute (M : Algebra α) (p : Parameters) (a b : Nat)
    (mem : RegionName → Nat → Cell α) :
    update M p a (update M p b mem)=update M p b (update M p a mem) := by
  by_cases same : a=b
  · subst b; rfl
  funext r i
  have disjoint : ¬(a*p.block ≤ i ∧ i<(a+1)*p.block ∧ b*p.block ≤ i ∧ i<(b+1)*p.block) := by
    rintro ⟨al,au,bl,bu⟩
    exact same (TritonMul2.Scheduling.tile_unique p.block a b i al au bl bu)
  by_cases ha : a*p.block ≤ i ∧ i<(a+1)*p.block
  · have hb : ¬(b*p.block ≤ i ∧ i<(b+1)*p.block) := fun hb => disjoint ⟨ha.1,ha.2,hb.1,hb.2⟩
    simp [update,resultCell,ha.1,ha.2,show ∀ r, ¬(r=p.output ∧ b*p.block ≤ i ∧ i<(b+1)*p.block ∧ i<p.n) from
      fun _ h=>hb ⟨h.2.1,h.2.2.1⟩]
  · by_cases hb : b*p.block ≤ i ∧ i<(b+1)*p.block
    · simp [update,resultCell,hb.1,hb.2,show ∀ r, ¬(r=p.output ∧ a*p.block ≤ i ∧ i<(a+1)*p.block ∧ i<p.n) from
        fun _ h=>ha ⟨h.2.1,h.2.2.1⟩]
    · simp [update,show ∀ r, ¬(r=p.output ∧ a*p.block ≤ i ∧ i<(a+1)*p.block ∧ i<p.n) from
        fun _ h=>ha ⟨h.2.1,h.2.2.1⟩,
        show ∀ r, ¬(r=p.output ∧ b*p.block ≤ i ∧ i<(b+1)*p.block ∧ i<p.n) from
        fun _ h=>hb ⟨h.2.1,h.2.2.1⟩]

theorem range_fold (M : Algebra α) (p : Parameters) (s : State α) (k : Nat) :
    (List.range k).foldl (fun mem pid=>update M p pid mem) s.mem=prefixMemory M p s k := by
  induction k with
  | zero => funext r i; simp [prefixMemory]
  | succ k ih =>
    simp only [List.range_succ,List.foldl_append,List.foldl_cons,List.foldl_nil,ih]
    have h := reference_step M p s k
    rw [expected_update] at h
    exact h

theorem ordered_memory (M : Algebra α) (p : Parameters) (s : State α)
    (positive : 0<p.block) (order : List Nat)
    (schedule : order.Perm (List.range (grid p.n p.block))) :
    order.foldl (fun mem pid=>update M p pid mem) s.mem=gridExpectedMemory M p s := by
  letI : RightCommutative (fun (mem : RegionName → Nat → Cell α) pid => update M p pid mem) :=
    ⟨fun mem a b=>commute M p b a mem⟩
  rw [schedule.foldl_eq s.mem]
  exact (range_fold M p s _).trans (grid_reference_complete M p s positive)

theorem executes (M : Algebra α) (src dst : RegionName) (n B : Nat)
    (order : List Nat) (s : State α) :
    ∃ final,runSchedule (order.map (invocation M src dst n B)) s=some final ∧
      final.mem=order.foldl (fun mem pid=>update M (parameters src dst n B) pid mem) s.mem := by
  induction order generalizing s with
  | nil => exact ⟨s,rfl,rfl⟩
  | cons pid rest ih =>
    let p := parameters src dst n B
    let start := (invocation M src dst n B pid).initial s
    obtain ⟨next,he,hm⟩ := complete_correct M p start
    obtain ⟨final,hf,hfm⟩ := ih next
    refine ⟨final,?_,?_⟩
    · change ((invocation M src dst n B pid).run s).bind _=some final
      change (exec M (candidateKernel src dst n B) start).bind _=some final
      change exec M (candidateKernel src dst n B) start=some next at he
      rw [he]
      exact hf
    · rw [hfm,hm,expected_update]
      rfl

theorem complete (M : Algebra α) (src dst : RegionName) (n B : Nat)
    (positive : 0<B) (order : List Nat)
    (schedule : order.Perm (List.range (grid n B))) (s : State α) :
    ∃ final,runSchedule (order.map (invocation M src dst n B)) s=some final ∧
      final.mem=gridExpectedMemory M (parameters src dst n B) s := by
  obtain ⟨final,he,hm⟩ := executes M src dst n B order s
  exact ⟨final,he,hm.trans (ordered_memory M _ s positive order schedule)⟩

#axiomsClean complete
end
end VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.ScheduledSource
