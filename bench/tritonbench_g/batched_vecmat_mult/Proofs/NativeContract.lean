import bench.tritonbench_g.batched_vecmat_mult.Proofs.SourceGraphTensor
import bench.tritonbench_g.batched_vecmat_mult.Proofs.NativeChoices
import bench.tritonbench_g.batched_vecmat_mult.Proofs.SourceGraphReal
import bench.tritonbench_g.batched_vecmat_mult.Proofs.ReturnedTensor

/-! End-to-end explicit contracted-graph interpretation of the original
wrapper and unchanged typed source. Ordinary fp32 allocation results are
embedded as address-tagged word leaves; source execution builds the operation graph; only
then `contract` introduces ternary FMA, and `evaluate` rounds each operation. A PlanSelector records a separate valid
tree for each tagged reduction footprint, preserving shared-writer choices.
This is a defined lowering model, not a claim that arbitrary native launches
match it. Native binding is the separately pinned, bounded SM90 text evidence.
The historical separate-operation contracts retain their original meanings. -/
namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.SourceGraph.Native
open VeriTile.Triton FP.Structural
open FP.Equational (ReductionPlan)
open ValueProofs.Host (Allocator Arguments Error Launch prepare TensorABI ReturnedTensor)
noncomputable section
local instance : Inhabited ValueProofs.Word := ⟨FP.Value.zero .fp32⟩

/-- Map only floating carriers; pointer, integer and Boolean cells stay typed. -/
def mapValue (f : α → β) : (d : TileDType) → Value α d → Value β d
  | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 => f
  | .nat | .int | .bool | .ptr | .blockPtr => id

def mapCell (f : α → β) : Cell α → Cell β
  | .mk d v => .mk d (mapValue f d v)

def liftMemory (mem : ValueProofs.Memory) : Memory := fun r o => mapCell (Expr.input r o) (mem r o)
def decodeMemory (cfg : FP.Config) (mem : Memory) : ValueProofs.Memory :=
  fun r o => mapCell (fun e => evaluate cfg (contract e)) (mem r o)

theorem decode_lift (cfg : FP.Config) (mem : ValueProofs.Memory) :
    decodeMemory cfg (liftMemory mem) = mem := by
  funext r o
  unfold decodeMemory liftMemory
  cases h : mem r o with
  | mk d v => cases d <;> rfl

def liftState (initial : State ValueProofs.Word) (mem : ValueProofs.Memory) : State Expr where
  mem := liftMemory mem
  regs := fun _ _ _ => none
  pids := initial.pids
  numPids := initial.numPids
  undef := fun d _ _ => defaultValue d

/-- The expected value contains no execution or output-memory lookup. All
rounding is in the distinct FP.add, FP.mul and FP.fma scalar primitives. -/
def expected (cfg : FP.Config) (plans : PlanSelector)
    (p : ValueProofs.Parameters) (mem : ValueProofs.Memory) (m n : Nat) : ValueProofs.Word :=
  evaluate cfg (contract (tensorValue plans p (liftMemory mem) m n (p.K/p.BK)))

/-- Exact original allocation/guard order, followed by complete source graph
execution over the actual floor grid and return of the allocated output view. -/
def runTensor (allocate : Allocator) (rest : Algebra Expr) (cfg : FP.Config)
    (plans : PlanSelector) (a : Arguments) (initial : State ValueProofs.Word) :
    Except Error (Option ReturnedTensor) := do
  let launch ← prepare allocate a initial.mem
  pure ((runGrid rest plans launch.parameters (liftState initial launch.memory)).map
    fun final => ⟨launch.parameters.M,launch.parameters.N,launch.parameters.output,
      decodeMemory cfg final.mem⟩)

/-- All coordinates of the returned tensor, including empty K; successful
execution and frame follow from source execution, not an output premise.
Fresh output and natural metadata are the ordinary public host ABI. -/
theorem returned_tensor_correct (allocate : Allocator) (rest : Algebra Expr)
    (cfg : FP.Config) (plans : PlanSelector) (a : Arguments)
    (initial : State ValueProofs.Word)
    (abi : ∀ l, prepare allocate a initial.mem = .ok l → TensorABI l) :
    match prepare allocate a initial.mem with
    | .error e => runTensor allocate rest cfg plans a initial = .error e
    | .ok l => ∃ result : ReturnedTensor,
        runTensor allocate rest cfg plans a initial = .ok (some result) ∧
        result.rows = a.M ∧ result.cols = a.N ∧ result.region = l.parameters.output ∧
        (∀ m : Fin l.parameters.M, ∀ n : Fin l.parameters.N,
          result.memory result.region (m.val*l.parameters.N+n.val) =
            Cell.mk .real (expected cfg plans l.parameters l.memory m.val n.val)) ∧
        (∀ r o, ValueProofs.GridOutside l.parameters r o → result.memory r o = l.memory r o) := by
  cases hp : prepare allocate a initial.mem with
  | error e => simp [runTensor, hp, bind, Except.bind]
  | ok l =>
    have prep := ValueProofs.Host.prepare_fields allocate a initial.mem l hp
    obtain ⟨final,he,hm,hf⟩ := grid_correct rest plans l.parameters (liftState initial l.memory)
    refine ⟨⟨l.parameters.M,l.parameters.N,l.parameters.output,decodeMemory cfg final.mem⟩,?_,
      prep.dimensions.1,prep.dimensions.2.1,rfl,?_,?_⟩
    · simp [runTensor, hp, he, bind, Except.bind, pure, Except.pure]
    · intro m n
      have separate : InputsSeparate l.parameters := abi l hp
      have hc := complete_tensor_readback plans l.parameters (liftState initial l.memory)
        (prep.blocks.1 ▸ prep.positive.1) (prep.blocks.2.1 ▸ prep.positive.2.1)
        (by rw [prep.blocks.1,prep.dimensions.1]; exact Nat.dvd_of_mod_eq_zero prep.divisible.2.1)
        (by rw [prep.blocks.2.1,prep.dimensions.2.1]; exact Nat.dvd_of_mod_eq_zero prep.divisible.2.2) m n
      change mapCell _ (final.mem _ _) = _
      rw [hm, ← complete_initial_memory_eq_grid plans l.parameters (liftState initial l.memory) separate, hc]
      rfl
    · intro r o outside
      have hg : GridOutside l.parameters r o := outside
      change decodeMemory cfg final.mem r o = _
      unfold decodeMemory
      rw [hf r o hg]
      exact congrFun (congrFun (decode_lift cfg l.memory) r) o

/-- Source graph execution for any complete program permutation has the same
returned words. This uses disjoint source writes; no FP arithmetic law occurs. -/
theorem arbitrary_order (rest : Algebra Expr) (cfg : FP.Config) (plans : PlanSelector)
    (p : ValueProofs.Parameters) (initial : State ValueProofs.Word) (jobs : List Nat)
    (separate : ValueProofs.InputsSeparate p) (order : jobs.Perm (List.range (gridCount p))) :
    ∃ final, runSchedule rest plans p (liftState initial initial.mem) jobs = some final ∧
      decodeMemory cfg final.mem =
        decodeMemory cfg (gridExpectedMemory plans p (liftState initial initial.mem)) := by
  obtain ⟨final,he,hm,_⟩ := complete_schedule_correct rest plans p
    (liftState initial initial.mem) separate jobs order
  refine ⟨final,he,?_⟩
  rw [hm,complete_initial_memory_eq_grid plans p (liftState initial initial.mem) separate]

#specNonCircular expected avoiding [runTensor, source, FP.Structural.exec, runGrid]
#axiomsClean decode_lift
#axiomsClean returned_tensor_correct
#axiomsClean arbitrary_order
end
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.SourceGraph.Native
