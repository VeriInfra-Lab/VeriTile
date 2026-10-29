import bench.tritonbench_g.sin_computation.Proofs.Real
import bench.repairs.tritonbench_g.sin_computation.Proofs.Contract
import VeriTile.Triton.Launch.Composition
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SinComputation.Contiguous.RealGrid
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SinComputation
noncomputable section
open Classical
set_option maxHeartbeats 4000000

/-- Mathematical repaired launch with the minimal complete BLOCK4 grid. -/
def grid (N : Nat) : Grid := ⟨[gridCount N]⟩
def kernel (X Y : RegionName) (N : Nat) : Kernel := (sin_kernel X Y N 4).toAlgKernel
/-- Local register/PID state for one program; memory is the current evolving memory. -/
def programState (s : BlockState) (N pid : Nat) : BlockState :=
  (s.withPids (fun ax => if ax=0 then pid else 0)).withNumPids
    (fun ax => if ax=0 then gridCount N else 1)

def runPrograms (X Y : RegionName) (N : Nat) : List Nat → BlockState → Option BlockState
  | [],s => some s
  | pid::rest,s => do
      let t ← exec (kernel X Y N) (programState s N pid)
      runPrograms X Y N rest t

/-- Raw exterior cells, including inactive output padding, are preserved. -/
def Frame (Y : RegionName) (N : Nat) (programs : List Nat) (s t : BlockState) : Prop :=
  ∀ r o, (r≠Y ∨ N≤o ∨ o/4∉programs) → t.mem r o=s.mem r o

def Values (X Y : RegionName) (N : Nat) (programs : List Nat) (s t : BlockState) : Prop :=
  ∀ o, o<N → o/4∈programs → t.readMem Y o=Real.sin (s.readMem X o)

theorem read_eq_of_mem {s t : BlockState} {r : RegionName} {o : Nat}
    (h : t.mem r o=s.mem r o) : t.readMem r o=s.readMem r o := by
  simp only [BlockState.readMem,h]

/-- The existing actual DSL execution theorem, instantiated at a mathematical
program id. X=Y is allowed: the input tile is loaded before the masked scatter. -/
theorem program_run (X Y : RegionName) (N pid : Nat) (s : BlockState) :
    ∃ t, exec (kernel X Y N) (programState s N pid)=some t ∧
      Values X Y N [pid] s t ∧ Frame Y N [pid] s t := by
  obtain ⟨t,he,hv,hf⟩ := sin_kernel_region_run X Y N 4 (programState s N pid)
    (fun j => s.readMem X (pid*4+j.val)) (by intro j _; rfl)
  refine ⟨t,he,?_,?_⟩
  · intro o ho hp
    have hp' : o/4=pid := by simpa using hp
    let j : Fin 4 := ⟨o%4,Nat.mod_lt _ (by decide)⟩
    have hj : pid*4+j.val=o := by dsimp [j]; omega
    have hh := hv j (by simpa [programState,hj] using ho)
    simpa [programState,hj] using hh
  · intro r o hout
    apply hf r o
    rcases hout with hr | hn | hp
    · exact Or.inl hr
    · right; intro j hj
      have hj' : pid*4+j.val<N := hj
      omega
    · right; intro j hj heq
      have hpid : o/4=pid := by have := j.isLt; change o=pid*4+j.val at heq; omega
      exact hp (by simp [hpid])

/-- Sequential execution uses the previous program's actual final memory.
No disjoint-input/output hypothesis is needed in this region model: distinct
programs access distinct index tiles even when the two regions are identical. -/
theorem runPrograms_correct (X Y : RegionName) (N : Nat) (programs : List Nat)
    (unique : programs.Nodup) (s : BlockState) :
    ∃ t, runPrograms X Y N programs s=some t ∧
      Values X Y N programs s t ∧ Frame Y N programs s t := by
  induction programs generalizing s with
  | nil =>
    exact ⟨s,rfl,by simp [Values],by intro r o _; rfl⟩
  | cons pid rest ih =>
    have hp : pid∉rest := (List.nodup_cons.mp unique).1
    have hr : rest.Nodup := (List.nodup_cons.mp unique).2
    obtain ⟨mid,hm,hvm,hfm⟩ := program_run X Y N pid s
    obtain ⟨t,ht,hvt,hft⟩ := ih hr mid
    refine ⟨t,by simp only [runPrograms,hm]; exact ht,?_,?_⟩
    · intro o ho hin
      rcases List.mem_cons.mp hin with hhead | htail
      · have hrest : o/4∉rest := by simpa [hhead] using hp
        have hkeep := read_eq_of_mem (hft Y o (Or.inr (Or.inr hrest)))
        exact hkeep.trans (hvm o ho (by simp [hhead]))
      · have hnot : o/4∉[pid] := by
          simp only [List.mem_singleton]
          intro heq; exact hp (heq ▸ htail)
        have hinput := read_eq_of_mem (hfm X o (Or.inr (Or.inr hnot)))
        rw [hvt o ho htail,hinput]
    · intro r o outside
      have a : r≠Y ∨ N≤o ∨ o/4∉rest := by
        rcases outside with h | h | h
        · exact Or.inl h
        · exact Or.inr (Or.inl h)
        · exact Or.inr (Or.inr (fun hx => h (List.mem_cons_of_mem pid hx)))
      have b : r≠Y ∨ N≤o ∨ o/4∉[pid] := by
        rcases outside with h | h | h
        · exact Or.inl h
        · exact Or.inr (Or.inl h)
        · right; right; intro hx
          have heq : o/4=pid := by simpa using hx
          exact h (by simp [heq])
      exact (hft r o a).trans (hfm r o b)

/-- Every logical index is covered, including tails and empty input. -/
theorem full_grid_correct (X Y : RegionName) (N : Nat) (s : BlockState) :
    ∃ t, runPrograms X Y N (List.range (gridCount N)) s=some t ∧
      (∀ o, o<N → t.readMem Y o=Real.sin (s.readMem X o)) ∧
      (∀ r o, r≠Y ∨ N≤o → t.mem r o=s.mem r o) := by
  obtain ⟨t,he,hv,hf⟩ := runPrograms_correct X Y N (List.range (gridCount N)) (List.nodup_range) s
  refine ⟨t,he,?_,?_⟩
  · intro o ho
    exact hv o ho (List.mem_range.mpr (by unfold gridCount; omega))
  · intro r o h
    apply hf r o
    rcases h with h | h
    · exact Or.inl h
    · exact Or.inr (Or.inl h)

/-- Any complete program permutation has the same mathematical values and
exterior raw-cell frame; exact in-place execution is included. -/
theorem complete_schedule_correct (X Y : RegionName) (N : Nat) (programs : List Nat)
    (schedule : programs.Perm (List.range (gridCount N))) (s : BlockState) :
    ∃ t, runPrograms X Y N programs s=some t ∧
      (∀ o, o<N → t.readMem Y o=Real.sin (s.readMem X o)) ∧
      (∀ r o, r≠Y ∨ N≤o → t.mem r o=s.mem r o) := by
  obtain ⟨t,he,hv,hf⟩ := runPrograms_correct X Y N programs
    (schedule.nodup_iff.mpr List.nodup_range) s
  refine ⟨t,he,?_,?_⟩
  · intro o ho
    exact hv o ho (schedule.mem_iff.mpr (List.mem_range.mpr (by unfold gridCount; omega)))
  · intro r o h
    apply hf r o
    rcases h with h | h
    · exact Or.inl h
    · exact Or.inr (Or.inl h)

#axiomsClean program_run
#axiomsClean runPrograms_correct
#axiomsClean full_grid_correct
#axiomsClean complete_schedule_correct
#specNonCircular Values avoiding [kernel, runPrograms]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SinComputation.Contiguous.RealGrid
