import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.FiniteExecution
import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.HostReal
import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.HostFP

/-! Public host semantics executes the finite64 model, then uses its proved
whole-grid refinement to compose the existing Real/FP and caller-frame proofs.
The source compiler and runtime implementation remain ordinary API boundaries. -/
namespace VeriTile.Bench.Repairs.BatchedVecmatMult.FiniteHost
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
open ValueProofs.Host Whole
noncomputable section
local instance : Inhabited ValueProofs.Word := ⟨FP.Value.zero .fp32⟩

/-- Exactly the same allocation/guard/setup order; successful launch executes
finite ABI casts, finite signed addresses and the signed dynamic loop. -/
def runGraph (allocate : Allocator) (setup : Setup) (_rest : Algebra SourceGraph.Expr)
    (plans : SourceGraph.PlanSelector) (a : Arguments) (initial : State ValueProofs.Word) :
    Except Whole.Error (Option GraphTensor) :=
  match prepare allocate a initial.mem with
  | .error e => .error (.preparation e)
  | .ok l => match setup l with
    | .error e => .error (.setup e)
    | .ok () => .ok ((FiniteExecution.grid plans l.parameters
        (SourceGraph.Native.liftState initial l.memory)).map fun mem =>
          ⟨l.parameters.M,l.parameters.N,l.parameters.output,mem⟩)

def runTensor (allocate : Allocator) (setup : Setup) (rest : Algebra SourceGraph.Expr)
    (cfg : FP.Config) (plans : SourceGraph.PlanSelector) (a : Arguments)
    (initial : State ValueProofs.Word) : Except Whole.Error (Option ReturnedTensor) :=
  match runGraph allocate setup rest plans a initial with
  | .error e => .error e
  | .ok result => .ok (result.map fun t =>
      ⟨t.rows,t.cols,t.region,SourceGraph.Native.decodeMemory cfg t.memory⟩)

/-- Every successful setup derives the integer premises and consumes the
finite execution refinement. This is equality of complete host outcomes,
not a separate assertion of index bounds alongside a Nat executor. -/
theorem run_graph_eq (allocate : Allocator) (setup : Setup) (rest : Algebra SourceGraph.Expr)
    (plans : SourceGraph.PlanSelector) (a : Arguments) (initial : State ValueProofs.Word)
    (native : SetupLaw setup)
    (storage : ∀ l, prepare allocate a initial.mem = .ok l → Index64.StorageLimits l.parameters) :
    runGraph allocate setup rest plans a initial = Whole.runGraph allocate setup rest plans a initial := by
  cases hp : prepare allocate a initial.mem with
  | error e => simp [runGraph,Whole.runGraph,hp]
  | ok l =>
    cases hs : setup l with
    | error e => simp [runGraph,Whole.runGraph,hp,hs]
    | ok u =>
      cases u
      have h := launch_integer_agreement l (storage l hp) (native l hs)
      simp [runGraph,Whole.runGraph,hp,hs,
        FiniteExecution.grid_source_execution rest plans l.parameters
          (SourceGraph.Native.liftState initial l.memory) h,
        Option.map_map,Function.comp_def]

theorem run_tensor_eq (allocate : Allocator) (setup : Setup) (rest : Algebra SourceGraph.Expr)
    (cfg : FP.Config) (plans : SourceGraph.PlanSelector) (a : Arguments)
    (initial : State ValueProofs.Word) (native : SetupLaw setup)
    (storage : ∀ l, prepare allocate a initial.mem = .ok l → Index64.StorageLimits l.parameters) :
    runTensor allocate setup rest cfg plans a initial =
      Whole.runTensor allocate setup rest cfg plans a initial := by
  unfold runTensor Whole.runTensor
  rw [run_graph_eq allocate setup rest plans a initial native storage]
  rfl

theorem returned_real_correct
    (decode : ValueProofs.Word → ℝ) (zero : decode (FP.Value.zero .fp32) = 0)
    (allocate : Allocator) (setup : Setup) (rest : Algebra SourceGraph.Expr)
    (plans : SourceGraph.PlanSelector) (a : Arguments) (initial : State ValueProofs.Word)
    (liveCells : RegionName → Nat → Prop)
    (frame : AllocationFrame allocate) (fresh : AllocationFresh liveCells allocate)
    (native : SetupLaw setup)
    (abi : ∀ l, prepare allocate a initial.mem = .ok l → TensorABI l)
    (storage : ∀ l, prepare allocate a initial.mem = .ok l → Index64.StorageLimits l.parameters) :
    match prepare allocate a initial.mem with
    | .error e => runGraph allocate setup rest plans a initial = .error (.preparation e)
    | .ok l => match setup l with
      | .error e => runGraph allocate setup rest plans a initial = .error (.setup e)
      | .ok () => IntegerAgreement l.parameters ∧ ∃ result : GraphTensor,
          runGraph allocate setup rest plans a initial = .ok (some result) ∧
          result.rows = a.M ∧ result.cols = a.N ∧ result.region = l.parameters.output ∧
          (∀ m : Fin l.parameters.M, ∀ n : Fin l.parameters.N,
            SourceGraph.realValue decode (SourceGraph.contract
              ((result.memory result.region (m.val*l.parameters.N+n.val)).read .real)) =
                realExpected decode l.parameters l.memory m.val n.val) ∧
          (∀ r o, ValueProofs.GridOutside l.parameters r o →
            result.memory r o = SourceGraph.Native.liftMemory l.memory r o) ∧
          (∀ r o, OutsidePrepared l.parameters r o →
            result.memory r o = SourceGraph.Native.liftMemory initial.mem r o) ∧
          (∀ r o, liveCells r o →
            result.memory r o = SourceGraph.Native.liftMemory initial.mem r o) := by
  simpa only [run_graph_eq allocate setup rest plans a initial native storage] using
    Whole.returned_real_correct decode zero allocate setup rest plans a initial
      liveCells frame fresh native abi storage

theorem returned_fp32_correct
    (allocate : Allocator) (setup : Setup) (rest : Algebra SourceGraph.Expr)
    (cfg : FP.Config) (D : SourceGraph.Local.DotAcc)
    (law : SourceGraph.Local.PrimitiveLaw cfg D)
    (plans : SourceGraph.PlanSelector) (a : Arguments) (initial : State ValueProofs.Word)
    (liveCells : RegionName → Nat → Prop)
    (frame : AllocationFrame allocate) (fresh : AllocationFresh liveCells allocate)
    (native : SetupLaw setup)
    (abi : ∀ l, prepare allocate a initial.mem = .ok l → TensorABI l)
    (storage : ∀ l, prepare allocate a initial.mem = .ok l → Index64.StorageLimits l.parameters) :
    match prepare allocate a initial.mem with
    | .error e => runTensor allocate setup rest cfg plans a initial = .error (.preparation e)
    | .ok l => match setup l with
      | .error e => runTensor allocate setup rest cfg plans a initial = .error (.setup e)
      | .ok () => IntegerAgreement l.parameters ∧ ∃ result : ReturnedTensor,
          runTensor allocate setup rest cfg plans a initial = .ok (some result) ∧
          result.rows = a.M ∧ result.cols = a.N ∧ result.region = l.parameters.output ∧
          (∀ m : Fin l.parameters.M, ∀ n : Fin l.parameters.N,
            result.memory result.region (m.val*l.parameters.N+n.val) = Cell.mk .real
              (SourceGraph.Local.fold D plans l.parameters l.memory m.val n.val
                (l.parameters.K/l.parameters.BK))) ∧
          (∀ r o, ValueProofs.GridOutside l.parameters r o → result.memory r o = l.memory r o) ∧
          (∀ r o, OutsidePrepared l.parameters r o → result.memory r o = initial.mem r o) ∧
          (∀ r o, liveCells r o → result.memory r o = initial.mem r o) := by
  simpa only [run_tensor_eq allocate setup rest cfg plans a initial native storage] using
    Whole.returned_fp32_correct allocate setup rest cfg D law plans a initial
      liveCells frame fresh native abi storage

#axiomsClean run_graph_eq
#axiomsClean run_tensor_eq
#axiomsClean returned_real_correct
#axiomsClean returned_fp32_correct
end
end VeriTile.Bench.Repairs.BatchedVecmatMult.FiniteHost
