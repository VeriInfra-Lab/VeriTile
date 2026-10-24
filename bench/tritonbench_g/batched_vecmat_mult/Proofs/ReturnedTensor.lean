import bench.tritonbench_g.batched_vecmat_mult.Proofs.HostContract
import bench.tritonbench_g.batched_vecmat_mult.Proofs.ScheduleContract

/-! Ordinary dense tensor coordinates connected to the unchanged source's
complete allocation/guard/grid execution. The allocator supplies actual fp32
words; no random-number distribution is assumed. The value theorem here keeps
the existing separate-operation software profile, including every rounded
product, reduction seed, carry addition and K=0 positive-zero initialization. -/
namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.ValueProofs
open VeriTile.Triton FP.Structural
open FP.Equational (ReductionPlan)
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 2000000

/-- Tensor-coordinate reference, independent of program IDs, output memory,
source execution and the initial output allocation. -/
def tensorBlock (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (mem : Memory) (m n c : Nat) : Word :=
  treeValue cfg (fun e : Fin p.BK => FP.mul cfg
    ((mem p.A (m*p.K + (c*p.BK+e.val))).read .real)
    ((mem p.B (m*p.N*p.K+n*p.K+(c*p.BK+e.val))).read .real)) (plans p.BK).tree

def tensorValue (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (mem : Memory) (m n : Nat) : Nat → Word
  | 0 => FP.Value.zero .fp32
  | c+1 => FP.add cfg (tensorValue cfg plans p mem m n c) (tensorBlock cfg plans p mem m n c)

theorem accumulated_tensor (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (s : State Word) (c : Nat) (i : TileIndex [p.BM,p.BN]) :
    accumulated cfg plans p s c i =
      tensorValue cfg plans p s.mem (row p s i.1.val) (col p s i.2.1.val) c := by
  induction c with
  | zero => rfl
  | succ c ih => rw [accumulated, tensorValue, ih]; rfl

/-- A complete schedule's reference actually stores each valid program lane.
The proof uses unique writers, not floating-point commutation. -/
theorem scheduled_readback (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (initial : State Word) (pid : Nat) (hp : pid < gridCount p)
    (i : TileIndex [p.BM,p.BN]) (jobs : List Nat) (mem : Memory)
    (valid : ∀ k ∈ jobs, k < gridCount p) (present : pid ∈ jobs) :
    scheduledInitialMemory cfg plans p initial jobs mem p.output
      (outputOffset p (programState p initial initial.mem pid) i) =
      Cell.mk .real (accumulated cfg plans p (programState p initial initial.mem pid)
        (p.K / p.BK) i) := by
  classical
  induction jobs generalizing mem with
  | nil => simp at present
  | cons k jobs ih =>
    change scheduledInitialMemory cfg plans p initial jobs
      (applyInitialProgram cfg plans p initial k mem) p.output _ = _
    by_cases tail : pid ∈ jobs
    · exact ih _ (fun q hq => valid q (List.mem_cons_of_mem _ hq)) tail
    · have head : pid = k := (List.mem_cons.mp present).resolve_right tail
      subst k
      rw [scheduled_memory_frame]
      · rw [applyInitialProgram, if_pos ⟨rfl,i,rfl⟩]
        exact actual_program_expected_readback cfg plans p initial initial.mem pid hp i
      · intro k hk write
        obtain ⟨_, j, address⟩ := write
        have same := (actual_program_output_unique p initial initial.mem initial.mem pid k hp
          (valid k (List.mem_cons_of_mem _ hk)) i j address).1
        exact tail (same ▸ hk)

/-- Every tensor coordinate is covered by the actual source grid and has the
independent initial-input value after any complete program permutation. -/
theorem complete_tensor_readback (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (initial : State Word) (hBM : 0 < p.BM) (hBN : 0 < p.BN)
    (hM : p.BM ∣ p.M) (hN : p.BN ∣ p.N) (m : Fin p.M) (n : Fin p.N) :
    completeInitialMemory cfg plans p initial p.output (m.val*p.N+n.val) =
      Cell.mk .real (tensorValue cfg plans p initial.mem m.val n.val (p.K/p.BK)) := by
  obtain ⟨pm,pn,i,j,hm,hn⟩ := complete_output_coverage p hBM hBN hM hN m n
  obtain ⟨hp,hpm,hpn⟩ := coordinates_cover p pm pn
  let lane : TileIndex [p.BM,p.BN] := (i,j,PUnit.unit)
  have hrow : row p (programState p initial initial.mem (pn.val*gridM p+pm.val)) i.val = m.val := by
    simpa [row, programState, hpm] using hm
  have hcol : col p (programState p initial initial.mem (pn.val*gridM p+pm.val)) j.val = n.val := by
    simpa [col, programState, hpn] using hn
  have h := scheduled_readback cfg plans p initial (pn.val*gridM p+pm.val) hp lane
    (List.range (gridCount p)) initial.mem (fun k hk => List.mem_range.mp hk) (List.mem_range.mpr hp)
  rw [accumulated_tensor] at h
  simp only [outputOffset, lane] at h
  rw [hrow, hcol] at h
  exact h

namespace Host

/-- Ordinary fresh output ABI. Input regions may alias each other; the output
allocation is distinct from each input. Tensor shapes and dense element strides
are precisely [M,K], [M,N,K], [M,N] as requested by `prepare`. -/
def TensorABI (l : Launch) : Prop := InputsSeparate l.parameters

structure ReturnedTensor where
  rows : Nat
  cols : Nat
  region : RegionName
  memory : Memory

def ReturnedTensor.at (t : ReturnedTensor) (m : Fin t.rows) (n : Fin t.cols) : Word :=
  (t.memory t.region (m.val*t.cols+n.val)).read .real

/-- Return the actual allocated output region after execution, as Python does. -/
def runTensor (allocate : Allocator) (rest : Algebra Word) (cfg : FP.Config)
    (plans : ∀ n, ReductionPlan n) (a : Arguments) (initial : State Word) :
    Except Error (Option ReturnedTensor) := do
  let launch ← prepare allocate a initial.mem
  pure ((runGrid rest cfg plans launch.parameters {initial with mem := launch.memory}).map
    fun final => ⟨launch.parameters.M, launch.parameters.N, launch.parameters.output, final.mem⟩)

/-- Full returned coordinates and frame, with original failure and guard order.
The sole fresh-output premise is an ordinary allocation ABI, not an execution
or final-output equation. Precision and all software rounding sites are fixed
by cfg/plans; this statement does not identify that graph with native FMA. -/
theorem returned_tensor_correct (allocate : Allocator) (rest : Algebra Word)
    (cfg : FP.Config) (plans : ∀ n, ReductionPlan n) (a : Arguments) (initial : State Word)
    (abi : ∀ l, prepare allocate a initial.mem = .ok l → TensorABI l) :
    match prepare allocate a initial.mem with
    | .error e => runTensor allocate rest cfg plans a initial = .error e
    | .ok l => ∃ result : ReturnedTensor,
        runTensor allocate rest cfg plans a initial = .ok (some result) ∧
        result.rows = a.M ∧ result.cols = a.N ∧ result.region = l.parameters.output ∧
        (∀ m : Fin l.parameters.M, ∀ n : Fin l.parameters.N,
          result.memory result.region (m.val*l.parameters.N+n.val) =
          Cell.mk .real (tensorValue cfg plans l.parameters l.memory m.val n.val
            (l.parameters.K/l.parameters.BK))) ∧
        (∀ r o, GridOutside l.parameters r o → result.memory r o = l.memory r o) := by
  cases hp : prepare allocate a initial.mem with
  | error e => simp [runTensor, hp, bind, Except.bind]
  | ok l =>
    have prep := prepare_fields allocate a initial.mem l hp
    obtain ⟨final,he,hm,hf⟩ := grid_correct rest cfg plans l.parameters {initial with mem := l.memory}
    refine ⟨⟨l.parameters.M,l.parameters.N,l.parameters.output,final.mem⟩,?_,
      prep.dimensions.1,prep.dimensions.2.1,rfl,?_,hf⟩
    · simp [runTensor, hp, he, bind, Except.bind, pure, Except.pure]
    · intro m n
      rw [hm, ← complete_initial_memory_eq_grid cfg plans l.parameters
        {initial with mem := l.memory} (abi l hp)]
      exact complete_tensor_readback cfg plans l.parameters {initial with mem := l.memory}
        (prep.blocks.1 ▸ prep.positive.1) (prep.blocks.2.1 ▸ prep.positive.2.1)
        (by rw [prep.blocks.1,prep.dimensions.1]; exact Nat.dvd_of_mod_eq_zero prep.divisible.2.1)
        (by rw [prep.blocks.2.1,prep.dimensions.2.1]; exact Nat.dvd_of_mod_eq_zero prep.divisible.2.2) m n

#axiomsClean returned_tensor_correct
end Host
#specNonCircular tensorValue avoiding [source, FP.Structural.exec, expectedMemory, completeInitialMemory]
#axiomsClean accumulated_tensor
#axiomsClean scheduled_readback
#axiomsClean complete_tensor_readback
end
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.ValueProofs
