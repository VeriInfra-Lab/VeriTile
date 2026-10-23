import bench.tritonbench_g.batched_vecmat_mult.Proofs.HostAccess
import bench.tritonbench_g.batched_vecmat_mult.Proofs.HostPrepare

/-! The original natural-integer logical wrapper: three ordered fp32 CUDA
allocation requests, ordered modulo guards, actual derived grid, and every
program's real interpreter execution with previous memory. The allocation
primitive is explicit, including failures. Its native tensor/RNG/ABI/freshness
implementation and Python values outside natural metadata remain structural
boundaries, alongside native compiler legality and device concurrency. -/
namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.ValueProofs.Host
open VeriTile.Triton FP.Structural
open FP.Equational (ReductionPlan)
noncomputable section
local instance hostContractWord : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def runWrapper (allocate : Allocator) (rest : Algebra Word) (cfg : FP.Config)
    (plans : ∀ n, ReductionPlan n) (a : Arguments) (initial : State Word) :
    Except Error (Option (State Word)) := do
  let launch ← prepare allocate a initial.mem
  pure (runGrid rest cfg plans launch.parameters { initial with mem := launch.memory })

specification batched_vecmat_logical_wrapper_correctness
    (allocate : Allocator) (rest : Algebra Word) (cfg : FP.Config)
    (plans : ∀ n, ReductionPlan n) (a : Arguments) (initial : State Word) :
    match prepare allocate a initial.mem with
    | .error e => runWrapper allocate rest cfg plans a initial = .error e
    | .ok launch =>
        Prepared allocate a initial.mem launch ∧
        ∃ final, runWrapper allocate rest cfg plans a initial = .ok (some final) ∧
          final.mem = gridExpectedMemory cfg plans launch.parameters
            { initial with mem := launch.memory } ∧
          (∀ r o, GridOutside launch.parameters r o → final.mem r o = launch.memory r o) := by
  cases hp : prepare allocate a initial.mem with
  | error e => simp [runWrapper, hp, bind, Except.bind]
  | ok launch =>
    obtain ⟨final, he, hm, hf⟩ := grid_correct rest cfg plans launch.parameters
      { initial with mem := launch.memory }
    refine ⟨prepare_fields allocate a initial.mem launch hp, final, ?_, hm, hf⟩
    simp [runWrapper, hp, bind, Except.bind, pure, Except.pure, he]

#axiomsClean batched_vecmat_logical_wrapper_correctness
end
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.ValueProofs.Host
