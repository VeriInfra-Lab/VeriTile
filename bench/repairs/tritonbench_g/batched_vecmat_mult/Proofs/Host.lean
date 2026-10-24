import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.AdmissionCore
import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.FiniteIndex
import bench.tritonbench_g.batched_vecmat_mult.Proofs.NativeContract

/-! Ordinary host API boundaries for the separate repair. Allocation and native
setup are explicit calls, in the Python order. Setup can fail after the three
allocations and K/M/N guards; its successful ABI is not a kernel correctness
assumption. The allocator's local frame is composed to the pre-call memory.
No allocator, compiler, launcher or RNG implementation is reproved here. -/
namespace VeriTile.Bench.Repairs.BatchedVecmatMult.Whole
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
open ValueProofs.Host
noncomputable section
local instance : Inhabited ValueProofs.Word := ⟨FP.Value.zero .fp32⟩

abbrev Setup := Launch → Except String Unit

def SetupLaw (setup : Setup) : Prop :=
  ∀ l, setup l = .ok () → LaunchABI l

inductive Error where
  | preparation (error : ValueProofs.Host.Error)
  | setup (message : String)
  deriving DecidableEq, Repr

structure GraphTensor where
  rows : Nat
  cols : Nat
  region : RegionName
  memory : SourceGraph.Memory

/-- The three allocation requests and guards execute exactly once. Native
admission is next; only its successful branch executes the complete grid. -/
def runGraph (allocate : Allocator) (setup : Setup) (rest : Algebra SourceGraph.Expr)
    (plans : SourceGraph.PlanSelector) (a : Arguments) (initial : State ValueProofs.Word) :
    Except Error (Option GraphTensor) :=
  match prepare allocate a initial.mem with
  | .error e => .error (.preparation e)
  | .ok l => match setup l with
    | .error e => .error (.setup e)
    | .ok () => .ok ((SourceGraph.runGrid rest plans l.parameters
        (SourceGraph.Native.liftState initial l.memory)).map fun final =>
          ⟨l.parameters.M,l.parameters.N,l.parameters.output,final.mem⟩)

/-- Same host program and grid, with actual FP32-word output interpretation. -/
def runTensor (allocate : Allocator) (setup : Setup) (rest : Algebra SourceGraph.Expr)
    (cfg : FP.Config) (plans : SourceGraph.PlanSelector) (a : Arguments)
    (initial : State ValueProofs.Word) : Except Error (Option ReturnedTensor) :=
  match runGraph allocate setup rest plans a initial with
  | .error e => .error e
  | .ok result => .ok (result.map fun t =>
      ⟨t.rows,t.cols,t.region,SourceGraph.Native.decodeMemory cfg t.memory⟩)

theorem prepare_error_precedes_setup (allocate : Allocator) (setup : Setup)
    (rest : Algebra SourceGraph.Expr) (cfg : FP.Config) (plans : SourceGraph.PlanSelector)
    (a : Arguments) (initial : State ValueProofs.Word) (e : ValueProofs.Host.Error)
    (h : prepare allocate a initial.mem = .error e) :
    runTensor allocate setup rest cfg plans a initial = .error (.preparation e) := by
  simp [runTensor,runGraph,h]

theorem setup_failure_after_prepare (allocate : Allocator) (setup : Setup)
    (rest : Algebra SourceGraph.Expr) (cfg : FP.Config) (plans : SourceGraph.PlanSelector)
    (a : Arguments) (initial : State ValueProofs.Word) (l : Launch) (e : String)
    (hp : prepare allocate a initial.mem = .ok l) (hs : setup l = .error e) :
    runTensor allocate setup rest cfg plans a initial = .error (.setup e) := by
  simp [runTensor,runGraph,hp,hs]

/-- Reuse the complete existing execution only after successful setup. -/
theorem admitted_run_eq (allocate : Allocator) (setup : Setup)
    (rest : Algebra SourceGraph.Expr) (cfg : FP.Config) (plans : SourceGraph.PlanSelector)
    (a : Arguments) (initial : State ValueProofs.Word) (l : Launch)
    (hp : prepare allocate a initial.mem = .ok l) (hs : setup l = .ok ()) :
    runTensor allocate setup rest cfg plans a initial =
      (SourceGraph.Native.runTensor allocate rest cfg plans a initial).mapError Error.preparation := by
  simp [runTensor,runGraph,SourceGraph.Native.runTensor,hp,hs,bind,Except.bind,
    pure,Except.pure,Except.mapError,Option.map_map,Function.comp_def]

def OutsideAllocation (request : AllocationRequest) (region r : RegionName) (o : Nat) : Prop :=
  r ≠ region ∨ request.shape.prod ≤ o

/-- A normal allocation initializes only its returned storage window. This is
a local non-numerical library contract, not the whole wrapper's frame. -/
def AllocationFrame (allocate : Allocator) : Prop :=
  ∀ request mem result, allocate request mem = .ok result →
    ∀ r o, OutsideAllocation request result.region r o → result.memory r o = mem r o

/-- Existing live caller cells are disjoint from each newly allocated window.
The predicate is arbitrary, so all caller-owned cells can be protected. -/
def AllocationFresh (liveCells : RegionName → Nat → Prop) (allocate : Allocator) : Prop :=
  ∀ request mem result, allocate request mem = .ok result →
    ∀ r o, liveCells r o → OutsideAllocation request result.region r o

def OutsidePrepared (p : Parameters) (r : RegionName) (o : Nat) : Prop :=
  (r ≠ p.A ∨ p.M*p.K ≤ o) ∧
  (r ≠ p.B ∨ p.M*p.N*p.K ≤ o) ∧
  (r ≠ p.output ∨ p.M*p.N ≤ o)

theorem prepare_frame (allocate : Allocator) (a : Arguments) (mem : ValueProofs.Memory)
    (l : Launch) (hp : prepare allocate a mem = .ok l) (frame : AllocationFrame allocate)
    (r : RegionName) (o : Nat) (outside : OutsidePrepared l.parameters r o) :
    l.memory r o = mem r o := by
  have prep := prepare_fields allocate a mem l hp
  obtain ⟨va,vb,vo,ha,hb,ho,hA,hB,hO,hm⟩ := prep.allocations
  rcases outside with ⟨oa,ob,oo⟩
  rw [hA,prep.dimensions.1,prep.dimensions.2.2] at oa
  rw [hB,prep.dimensions.1,prep.dimensions.2.1,prep.dimensions.2.2] at ob
  rw [hO,prep.dimensions.1,prep.dimensions.2.1] at oo
  have fa := frame ⟨.randn,[a.M,a.K]⟩ mem va ha r o
    (by simpa [OutsideAllocation] using oa)
  have fb := frame ⟨.randn,[a.M,a.N,a.K]⟩ va.memory vb hb r o
    (by simpa [OutsideAllocation,Nat.mul_assoc] using ob)
  have fo := frame ⟨.zeros,[a.M,a.N]⟩ vb.memory vo ho r o
    (by simpa [OutsideAllocation] using oo)
  exact (congrFun (congrFun hm r) o).trans (fo.trans (fb.trans fa))

theorem protected_outside (allocate : Allocator) (a : Arguments) (mem : ValueProofs.Memory)
    (l : Launch) (hp : prepare allocate a mem = .ok l)
    (liveCells : RegionName → Nat → Prop) (fresh : AllocationFresh liveCells allocate)
    (r : RegionName) (o : Nat) (h : liveCells r o) : OutsidePrepared l.parameters r o := by
  have prep := prepare_fields allocate a mem l hp
  obtain ⟨va,vb,vo,ha,hb,ho,hA,hB,hO,_⟩ := prep.allocations
  refine ⟨?_,?_,?_⟩
  · have x := fresh ⟨.randn,[a.M,a.K]⟩ mem va ha r o h
    simpa [OutsideAllocation,hA,prep.dimensions.1,prep.dimensions.2.2] using x
  · have x := fresh ⟨.randn,[a.M,a.N,a.K]⟩ va.memory vb hb r o h
    simpa [OutsideAllocation,hB,prep.dimensions.1,prep.dimensions.2.1,
      prep.dimensions.2.2,Nat.mul_assoc] using x
  · have x := fresh ⟨.zeros,[a.M,a.N]⟩ vb.memory vo ho r o h
    simpa [OutsideAllocation,hO,prep.dimensions.1,prep.dimensions.2.1] using x

theorem outside_grid (p : Parameters) (r : RegionName) (o : Nat)
    (h : OutsidePrepared p r o) : ValueProofs.GridOutside p r o := by
  rcases h.2.2 with h | h
  · exact Or.inl h
  · right
    intro pm pn lane eqn
    have bound := ValueProofs.output_address_in_declared_extent p pm pn lane.1 lane.2.1
    omega

#axiomsClean prepare_error_precedes_setup
#axiomsClean setup_failure_after_prepare
#axiomsClean admitted_run_eq
#axiomsClean prepare_frame
#axiomsClean protected_outside
#axiomsClean outside_grid
#axiomsClean runtime_dimension_word
#axiomsClean runtime_dimension_exact
#axiomsClean signed_loop_bound_exact
#axiomsClean launch_integer_agreement
end
end VeriTile.Bench.Repairs.BatchedVecmatMult.Whole
