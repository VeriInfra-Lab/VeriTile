import bench.tritonbench_g.cosine_compute.Proofs.Real
import bench.tritonbench_g.vector_addition.Proofs.HostGrid
import VeriTile.Triton.Launch.Grid
import VeriTile.Meta.StatementAudit

/-! Generic positive-block whole launch of the original cosine projection.
Host math chooses the block later. The actual grid is (count,1,1). Programs
carry preceding memory and registers, replacing only launch metadata. -/
namespace VeriTile.Bench.TritonBenchG.CosineCompute.RealGrid
open VeriTile.Triton
noncomputable section
open Classical
set_option maxHeartbeats 4000000

def count (N B : Nat) : Nat := (N+B-1)/B
def grid (N B : Nat) : Grid := ⟨[count N B,1,1]⟩
def kernel (X O : RegionName) (N B : Nat) : Kernel :=
  (RealProof.realKernel Bool.false X O N B).toAlgKernel

theorem source_projection (storage : Storage) (X O : RegionName) (N B : Nat) :
    (CosineCompute.kernel storage Bool.false X O N B).eraseDType.toAlgKernel = kernel X O N B :=
  RealProof.projects storage Bool.false X O N B

def programState (s : BlockState) (N B pid : Nat) : BlockState :=
  (s.withPids (fun ax => if ax=0 then pid else 0)).withNumPids
    (fun ax => if ax=0 then count N B else 1)

def runPrograms (X O : RegionName) (N B : Nat) : List Nat → BlockState → Option BlockState
  | [],s => some s
  | pid::rest,s => do
      let t ← exec (kernel X O N B) (programState s N B pid)
      runPrograms X O N B rest t

def Frame (O : RegionName) (N B : Nat) (programs : List Nat) (s t : BlockState) : Prop :=
  ∀ r o, (r≠O ∨ N≤o ∨ o/B∉programs) → t.mem r o=s.mem r o

def Values (X O : RegionName) (N B : Nat) (programs : List Nat) (s t : BlockState) : Prop :=
  ∀ o, o<N → o/B∈programs → t.readMem O o=Real.cos (s.readMem X o)

theorem read_eq_of_mem {s t : BlockState} {r : RegionName} {o : Nat}
    (h : t.mem r o=s.mem r o) : t.readMem r o=s.readMem r o := by
  simp only [BlockState.readMem,h]

/-- Actual original DSL execution, with arbitrary positive block size. -/
theorem program_run (X O : RegionName) (N B pid : Nat) (positive : 0<B) (s : BlockState) :
    ∃ t, exec (kernel X O N B) (programState s N B pid)=some t ∧
      Values X O N B [pid] s t ∧ Frame O N B [pid] s t := by
  obtain ⟨t,he,hv,hf⟩ := RealProof.cos_func_region_run X O N B (programState s N B pid)
    (fun j => s.readMem X (pid*B+j.val)) (by intro j _; rfl)
  refine ⟨t,he,?_,?_⟩
  · intro o ho hp
    have hp' : o/B=pid := by simpa using hp
    let j : Fin B := ⟨o%B,Nat.mod_lt _ positive⟩
    have hj : pid*B+j.val=o := by
      dsimp [j]
      simpa only [hp'] using Nat.div_add_mod' o B
    have hh := hv j (by simpa [programState,hj] using ho)
    simpa [programState,hj] using hh
  · intro r o hout
    apply hf r o
    rcases hout with hr | hn | hp
    · exact Or.inl hr
    · right; intro j hj
      have hj' : pid*B+j.val<N := hj
      omega
    · right; intro j hj heq
      have heq' : o=pid*B+j.val := heq
      have hpid : o/B=pid := by
        rw [heq']
        calc (pid*B+j.val)/B = (B*pid+j.val)/B := by rw [Nat.mul_comm pid B]
             _ = pid+j.val/B := Nat.mul_add_div positive pid j.val
             _ = pid := by rw [Nat.div_eq_of_lt j.isLt]; omega
      exact hp (by simp [hpid])

/-- Distinct tiles preserve later input values, including logical X=O. -/
theorem runPrograms_correct (X O : RegionName) (N B : Nat) (positive : 0<B)
    (programs : List Nat) (unique : programs.Nodup) (s : BlockState) :
    ∃ t, runPrograms X O N B programs s=some t ∧
      Values X O N B programs s t ∧ Frame O N B programs s t := by
  induction programs generalizing s with
  | nil => exact ⟨s,rfl,by simp [Values],by intro r o _; rfl⟩
  | cons pid rest ih =>
    have hp : pid∉rest := (List.nodup_cons.mp unique).1
    have hr : rest.Nodup := (List.nodup_cons.mp unique).2
    obtain ⟨mid,hm,hvm,hfm⟩ := program_run X O N B pid positive s
    obtain ⟨t,ht,hvt,hft⟩ := ih hr mid
    refine ⟨t,by simp only [runPrograms,hm]; exact ht,?_,?_⟩
    · intro o ho hin
      rcases List.mem_cons.mp hin with hhead | htail
      · have hrest : o/B∉rest := by simpa [hhead] using hp
        have keep := read_eq_of_mem (hft O o (Or.inr (Or.inr hrest)))
        exact keep.trans (hvm o ho (by simp [hhead]))
      · have hnot : o/B∉[pid] := by
          simp only [List.mem_singleton]
          intro heq; exact hp (heq ▸ htail)
        have hx := read_eq_of_mem (hfm X o (Or.inr (Or.inr hnot)))
        rw [hvt o ho htail,hx]
    · intro r o outside
      have a : r≠O ∨ N≤o ∨ o/B∉rest := by
        rcases outside with h | h | h
        · exact Or.inl h
        · exact Or.inr (Or.inl h)
        · exact Or.inr (Or.inr (fun hx => h (List.mem_cons_of_mem pid hx)))
      have b : r≠O ∨ N≤o ∨ o/B∉[pid] := by
        rcases outside with h | h | h
        · exact Or.inl h
        · exact Or.inr (Or.inl h)
        · right; right; intro hx
          have heq : o/B=pid := by simpa using hx
          exact h (by simp [heq])
      exact (hft r o a).trans (hfm r o b)

/-- Positive ceil division covers every active natural index, regardless of
whether the block is a true power of two or natively admissible. -/
theorem grid_covers (N B : Nat) (positive : 0<B) : N≤count N B*B :=
  VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.grid_covers
    (⟨"x","x","out",N,B⟩ : VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.Parameters) positive

theorem complete_schedule_correct (X O : RegionName) (N B : Nat) (positive : 0<B)
    (programs : List Nat) (schedule : programs.Perm (List.range (count N B))) (s : BlockState) :
    ∃ t, runPrograms X O N B programs s=some t ∧
      (∀ o, o<N → t.readMem O o=Real.cos (s.readMem X o)) ∧
      (∀ r o, r≠O ∨ N≤o → t.mem r o=s.mem r o) := by
  obtain ⟨t,he,hv,hf⟩ := runPrograms_correct X O N B positive programs
    (schedule.nodup_iff.mpr List.nodup_range) s
  refine ⟨t,he,?_,?_⟩
  · intro o ho
    have owner : o/B<count N B := (Nat.div_lt_iff_lt_mul positive).mpr
      (lt_of_lt_of_le ho (grid_covers N B positive))
    exact hv o ho (schedule.mem_iff.mpr (List.mem_range.mpr owner))
  · intro r o h
    apply hf r o
    rcases h with h | h
    · exact Or.inl h
    · exact Or.inr (Or.inl h)

theorem full_grid_correct (X O : RegionName) (N B : Nat) (positive : 0<B) (s : BlockState) :
    ∃ t, runPrograms X O N B (List.range (count N B)) s=some t ∧
      (∀ o, o<N → t.readMem O o=Real.cos (s.readMem X o)) ∧
      (∀ r o, r≠O ∨ N≤o → t.mem r o=s.mem r o) :=
  complete_schedule_correct X O N B positive _ (List.Perm.refl _) s

/-- The source's three grid dimensions, with carried memory/registers. -/
theorem programState_metadata (s : BlockState) (N B pid : Nat) :
    (programState s N B pid).mem=s.mem ∧ (programState s N B pid).regs=s.regs ∧
      (∀ ax, (programState s N B pid).numPids ax=(grid N B).toNumPids ax) := by
  refine ⟨rfl,rfl,?_⟩
  intro ax
  by_cases h0 : ax=0
  · subst ax; simp [programState,BlockState.withNumPids,grid,Grid.toNumPids,Grid.rank,Grid.dim]
  by_cases h1 : ax=1
  · subst ax; simp [programState,BlockState.withNumPids,grid,Grid.toNumPids,Grid.rank,Grid.dim]
  by_cases h2 : ax=2
  · subst ax; simp [programState,BlockState.withNumPids,grid,Grid.toNumPids,Grid.rank,Grid.dim]
  have h3 : ¬ax<3 := by omega
  simp [programState,BlockState.withNumPids,grid,Grid.toNumPids,Grid.rank,h0,h3]

#axiomsClean source_projection
#axiomsClean program_run
#axiomsClean runPrograms_correct
#axiomsClean grid_covers
#axiomsClean complete_schedule_correct
#axiomsClean full_grid_correct
#axiomsClean programState_metadata
#specNonCircular Values avoiding [kernel,runPrograms]
#specNonCircular Frame avoiding [kernel,runPrograms]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.CosineCompute.RealGrid
