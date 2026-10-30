import bench.tritonbench_g.fp4_to_bf16.Proofs.Finite

/-! Finite-ABI binding of the complete reused source. Count and pid enter
through their actual widened bitvectors. The instruction-level arithmetic,
masks and addresses are certified in Finite; this executable binding uses their
nonnegative decoded indices. No task-output or successful-execution premise is
used to relate the resulting launch to the complete existing schedule. -/
namespace VeriTile.Bench.Repairs.Fp4ToBf16.Launch
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16 FPValue Host HostSchedule
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def parameters (h : Input) : Parameters :=
  ⟨h.input,h.output,(Finite.nativeCount h.shape.count).toNat,512⟩

def program (rest : Algebra Word) (h : Input) (pid : Nat) (initial : State Word) : Option (State Word) :=
  FP.Structural.exec (backend rest) (Kernels.source (parameters h))
    (programState h ((BitVec.ofNat 32 pid).signExtend 64).toNat initial)

def run (rest : Algebra Word) (h : Input) : List Nat → State Word → Option (State Word)
  | [],initial => some initial
  | pid::jobs,initial => (program rest h pid initial).bind (run rest h jobs)

theorem count_exact (h : Input) (s : Finite.Storage h.shape.count) :
    (Finite.nativeCount h.shape.count).toNat=h.shape.count := by
  rw [Finite.native_count_refinement]
  simp only [Kernels.count, BitVec.toNat_ofNat]
  have bound := Finite.storage_count_bound s
  apply Nat.mod_eq_of_lt
  norm_num
  omega

theorem pid_exact (h : Input) (pid : Nat) (gridSafe : grid h≤2147483647) (hp : pid<grid h) :
    ((BitVec.ofNat 32 pid).signExtend 64).toNat=pid := by
  rw [Finite.native_pid_refinement h.shape.count pid gridSafe hp]
  simp only [BitVec.toNat_ofNat]
  apply Nat.mod_eq_of_lt
  norm_num
  omega

theorem program_refinement (rest : Algebra Word) (h : Input) (s : Finite.Storage h.shape.count)
    (pid : Nat) (initial : State Word) (gridSafe : grid h≤2147483647) (hp : pid<grid h) :
    program rest h pid initial =
      FP.Structural.exec (backend rest) (source (Host.parameters h)) (programState h pid initial) := by
  unfold program parameters
  rw [count_exact h s, pid_exact h pid gridSafe hp]
  rfl

theorem schedule_refinement (rest : Algebra Word) (h : Input) (s : Finite.Storage h.shape.count)
    (jobs : List Nat) (initial : State Word) (gridSafe : grid h≤2147483647)
    (valid : ∀ pid∈jobs,pid<grid h) :
    run rest h jobs initial = runSchedule rest h jobs initial := by
  induction jobs generalizing initial with
  | nil => rfl
  | cons pid jobs ih =>
    have hp := valid pid (by simp)
    have hv : ∀ p∈jobs,p<grid h := fun p member => valid p (by simp [member])
    simp only [run, runSchedule, program_refinement rest h s pid initial gridSafe hp]
    congr 1
    funext previous
    exact ih previous hv

theorem complete_refinement (rest : Algebra Word) (h : Input) (s : Finite.Storage h.shape.count)
    (jobs : List Nat) (initial : State Word) (gridSafe : grid h≤2147483647)
    (order : jobs.Perm (List.range (grid h))) :
    run rest h jobs initial = runSchedule rest h jobs initial := by
  apply schedule_refinement rest h s jobs initial gridSafe
  intro pid member
  exact List.mem_range.mp (order.mem_iff.mp member)

#axiomsClean program_refinement
#axiomsClean complete_refinement
end
end VeriTile.Bench.Repairs.Fp4ToBf16.Launch
