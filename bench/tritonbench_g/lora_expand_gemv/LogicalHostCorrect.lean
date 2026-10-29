import bench.tritonbench_g.lora_expand_gemv.Proofs.HostGrid
import bench.tritonbench_g.lora_expand_gemv.Proofs.HostCases

/-! The original wrapper's complete ordered logical host computation followed
by execution of every program in its (N,batches) grid. This is one explicit
sequential grid schedule over shared total typed memory. Output dtype is an
explicit fp16/bf16/fp32 binding; indices use the existing signed mathematical
cell model. The Python wrapper's unchecked output/index dtype domain, native
tensor metadata/allocation/ABI, compiler admissibility and device concurrency
remain separate structural obligations, not deferred numerical assumptions. -/
namespace VeriTile.Bench.TritonBenchG.LoraExpandGemv.Host
open VeriTile.Triton FP.Structural
open FPValue (Word OutputFormat)
open FP.Equational (ReductionPlan)
noncomputable section
local instance logicalHostWord : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Host failures terminate before any launch. A successful setup invokes
the actual admitted source successively over the entire original grid. The
output-format parameter is bound to the descriptor by the public contract. -/
def runWrapper (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (a : Arguments) (out : OutputFormat) (initial : State Word) :
    Except HostError (Option (State Word)) := do
  let launch ← prepare a
  pure (runLaunch rest cfg plans launch out initial)

/-- All host assertion/shape branches, including both 3D and singleton-4D
weights and the original ADD_INPUTS default/choice, feed the complete grid
execution. The reference observes prior program writes and signed sentinel
indices; it is not a merge of programs executed against isolated memories. -/
specification lora_expand_logical_wrapper_correctness
    (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (a : Arguments) (out : OutputFormat) (initial : State Word)
    (hout : a.output.dtype = outputDType out) :
    match prepare a with
    | .error e => runWrapper rest cfg plans a out initial = .error e
    | .ok launch =>
        Prepared a launch ∧ launch.outputType = outputDType out ∧
        ∃ final, runWrapper rest cfg plans a out initial = .ok (some final) ∧
          final.mem = expectedMemory cfg plans launch out initial ∧
          (∀ r o, Outside launch r o → final.mem r o = initial.mem r o) := by
  cases hp : prepare a with
  | error e => simp [runWrapper, hp, bind, Except.bind]
  | ok launch =>
    have properties := prepare_fields a launch hp
    obtain ⟨final, he, hm, hf⟩ := launch_correct rest cfg plans launch out initial
      (prepare_positive_block a launch hp)
    refine ⟨properties, properties.outputType.trans hout, final, ?_, hm, hf⟩
    simp [runWrapper, hp, bind, Except.bind, pure, Except.pure, he]

/-- Metadata rejection has no memory effect because no operational launch is
invoked. This also preserves the original order of early errors. -/
theorem rejected_before_launch (rest : Algebra Word) (cfg : FP.Config)
    (plans : ∀ n, ReductionPlan n) (a : Arguments) (out : OutputFormat)
    (initial : State Word) (e : HostError) (h : prepare a = .error e) :
    runWrapper rest cfg plans a out initial = .error e := by
  simp [runWrapper, h, bind, Except.bind]

#axiomsClean lora_expand_logical_wrapper_correctness
#axiomsClean rejected_before_launch
end
end VeriTile.Bench.TritonBenchG.LoraExpandGemv.Host
