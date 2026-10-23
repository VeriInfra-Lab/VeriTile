import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.CheckedHost
import bench.repairs.tritonbench_g.batched_vecmat_mult.Kernels
import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.FiniteIndex
import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.FiniteHost

/-! The repair changes only exact integer addressing. This public FP contract
retains the full original wrapper, actual input words, complete source grid,
allocated return identity, every output coordinate and complete memory frame.
FP32 precision and all-word domain are explicit. The only numerical obligation
is one local reduction on raw A/B vectors, incoming carry and a selected valid
tree; no execution, memory, index or final-output law is deferred. Contraction
is interpreted by the existing explicit graph, never by equating mul/add/FMA.
The selector keeps every configuration/tree choice separate. -/
namespace VeriTile.Bench.Repairs.BatchedVecmatMult.HistoricalFPEquivalence
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
open ValueProofs.Host
noncomputable section
local instance : Inhabited ValueProofs.Word := ⟨FP.Value.zero .fp32⟩

/-- The earlier cast-erased logical wrapper, retained with its precise scope:
native setup outcomes and pre-allocation frame are supplied by the new theorem. -/
abbrev runLogicalTensor := SourceGraph.Native.runTensor

specification repaired_logical_returned_fp32_correct
    (allocate : Allocator) (rest : Algebra SourceGraph.Expr)
    (D : SourceGraph.Local.DotAcc)
    (law : SourceGraph.Local.PrimitiveLaw FP.Config.ieeeValue D)
    (plans : SourceGraph.PlanSelector) (a : Arguments) (initial : State ValueProofs.Word)
    (abi : ∀ l, prepare allocate a initial.mem = .ok l → TensorABI l)
    (storage : ∀ l, prepare allocate a initial.mem = .ok l → Index64.StorageLimits l.parameters) :
    match prepare allocate a initial.mem with
    | .error e => runLogicalTensor allocate rest FP.Config.ieeeValue plans a initial = .error e
    | .ok l => Index64.AddressAgreement l.parameters ∧ ∃ result : ReturnedTensor,
        runLogicalTensor allocate rest FP.Config.ieeeValue plans a initial = .ok (some result) ∧
        result.rows = a.M ∧ result.cols = a.N ∧ result.region = l.parameters.output ∧
        (∀ m : Fin l.parameters.M, ∀ n : Fin l.parameters.N,
          result.memory result.region (m.val*l.parameters.N+n.val) = Cell.mk .real
            (SourceGraph.Local.fold D plans l.parameters l.memory m.val n.val
              (l.parameters.K/l.parameters.BK))) ∧
        (∀ r o, ValueProofs.GridOutside l.parameters r o → result.memory r o = l.memory r o) := by
  have h := SourceGraph.Local.returned_tensor_local_correct
    allocate rest FP.Config.ieeeValue D law plans a initial abi
  cases hp : prepare allocate a initial.mem with
  | error e => simpa only [hp] using h
  | ok l =>
    refine ⟨Index64.complete_address_agreement l.parameters (storage l hp), ?_⟩
    simpa only [hp] using h

/-- Full host program: allocation/guard errors, native setup errors, then the
same complete typed source and returned output allocation. -/
abbrev runTensor := FiniteHost.runTensor

/-- All allocation and launch obligations are local ordinary API contracts.
They do not assume kernel success, a final output equation or a whole-program
frame. Successful setup yields the complete integer correspondence; setup
failure is retained even for positive divisible shapes. The conclusion keeps
both the launch frame and the frame relative to the caller's initial memory. -/
specification repaired_returned_fp32_correct
    (allocate : Allocator) (setup : Whole.Setup) (rest : Algebra SourceGraph.Expr)
    (D : SourceGraph.Local.DotAcc)
    (law : SourceGraph.Local.PrimitiveLaw FP.Config.ieeeValue D)
    (plans : SourceGraph.PlanSelector) (a : Arguments) (initial : State ValueProofs.Word)
    (liveCells : RegionName → Nat → Prop)
    (frame : Whole.AllocationFrame allocate) (fresh : Whole.AllocationFresh liveCells allocate)
    (native : Whole.SetupLaw setup)
    (abi : ∀ l, prepare allocate a initial.mem = .ok l → TensorABI l)
    (storage : ∀ l, prepare allocate a initial.mem = .ok l → Index64.StorageLimits l.parameters) :
    match prepare allocate a initial.mem with
    | .error e => runTensor allocate setup rest FP.Config.ieeeValue plans a initial =
        .error (.preparation e)
    | .ok l => match setup l with
      | .error e => runTensor allocate setup rest FP.Config.ieeeValue plans a initial =
          .error (.setup e)
      | .ok () => Whole.IntegerAgreement l.parameters ∧ ∃ result : ReturnedTensor,
          runTensor allocate setup rest FP.Config.ieeeValue plans a initial = .ok (some result) ∧
          result.rows = a.M ∧ result.cols = a.N ∧ result.region = l.parameters.output ∧
          (∀ m : Fin l.parameters.M, ∀ n : Fin l.parameters.N,
            result.memory result.region (m.val*l.parameters.N+n.val) = Cell.mk .real
              (SourceGraph.Local.fold D plans l.parameters l.memory m.val n.val
                (l.parameters.K/l.parameters.BK))) ∧
          (∀ r o, ValueProofs.GridOutside l.parameters r o → result.memory r o = l.memory r o) ∧
          (∀ r o, Whole.OutsidePrepared l.parameters r o → result.memory r o = initial.mem r o) ∧
          (∀ r o, liveCells r o → result.memory r o = initial.mem r o) := by
  exact FiniteHost.returned_fp32_correct allocate setup rest FP.Config.ieeeValue D law plans a initial
    liveCells frame fresh native abi storage

/-- Exact scalar admission composed with the complete finite source result. -/
specification repaired_checked_returned_fp32_correct
    (allocate : Allocator) (setup : Whole.Setup) (rest : Algebra SourceGraph.Expr)
    (D : SourceGraph.Local.DotAcc)
    (law : SourceGraph.Local.PrimitiveLaw FP.Config.ieeeValue D)
    (plans : SourceGraph.PlanSelector) (raw : Contract.Arguments) (admitted : Contract.InputContract raw) (initial : State ValueProofs.Word)
    (liveCells : RegionName → Nat → Prop)
    (frame : Whole.AllocationFrame allocate) (fresh : Whole.AllocationFresh liveCells allocate)
    (native : Whole.SetupLaw setup)
    (abi : ∀ l, prepare allocate (Contract.toArguments raw) initial.mem = .ok l → TensorABI l)
    (storage : ∀ l, prepare allocate (Contract.toArguments raw) initial.mem = .ok l → Index64.StorageLimits l.parameters) :
    match prepare allocate (Contract.toArguments raw) initial.mem with
    | .error e => CheckedHost.runTensor allocate setup rest FP.Config.ieeeValue plans raw initial =
        some (.error (.preparation e))
    | .ok l => match setup l with
      | .error e => CheckedHost.runTensor allocate setup rest FP.Config.ieeeValue plans raw initial =
          some (.error (.setup e))
      | .ok () => Whole.IntegerAgreement l.parameters ∧ ∃ result : ReturnedTensor,
          CheckedHost.runTensor allocate setup rest FP.Config.ieeeValue plans raw initial = some (.ok (some result)) ∧
          result.rows = (Contract.toArguments raw).M ∧ result.cols = (Contract.toArguments raw).N ∧ result.region = l.parameters.output ∧
          (∀ m : Fin l.parameters.M, ∀ n : Fin l.parameters.N,
            result.memory result.region (m.val*l.parameters.N+n.val) = Cell.mk .real
              (SourceGraph.Local.fold D plans l.parameters l.memory m.val n.val
                (l.parameters.K/l.parameters.BK))) ∧
          (∀ r o, ValueProofs.GridOutside l.parameters r o → result.memory r o = l.memory r o) ∧
          (∀ r o, Whole.OutsidePrepared l.parameters r o → result.memory r o = initial.mem r o) ∧
          (∀ r o, liveCells r o → result.memory r o = initial.mem r o) := by
  simpa only [CheckedHost.tensor_admitted _ _ _ _ _ _ _ admitted, Option.some.injEq] using
    repaired_returned_fp32_correct allocate setup rest D law plans
      (Contract.toArguments raw) initial liveCells frame fresh native abi storage

#axiomsClean repaired_checked_returned_fp32_correct


/-- Index correspondence is an unconditional theorem of the stated ordinary
storage ABI. It is never passed as a numerical assumption. K=0 output stores
are covered separately by `Index64.all_output_addresses`. -/
theorem repaired_all_native_input_addresses (p : Parameters) (s : Index64.StorageLimits p)
    (pm : Fin (ValueProofs.gridM p)) (pn : Fin (ValueProofs.gridN p)) (c : Fin (p.K/p.BK))
    (i : Fin p.BM) (j : Fin p.BN) (e : Fin p.BK) :
    (Index64.aOffset p pm.val c.val i.val e.val).toInt =
      (((pm.val*p.BM+i.val)*p.K+(c.val*p.BK+e.val) : Nat) : Int) ∧
    (Index64.bOffset p pm.val pn.val c.val i.val j.val e.val).toInt =
      (((pm.val*p.BM+i.val)*p.N*p.K+(pn.val*p.BN+j.val)*p.K+(c.val*p.BK+e.val) : Nat) : Int) ∧
    ((pm.val*p.BM+i.val)*p.K+(c.val*p.BK+e.val))*4+3 < p.M*p.K*4 ∧
    ((pm.val*p.BM+i.val)*p.N*p.K+(pn.val*p.BN+j.val)*p.K+(c.val*p.BK+e.val))*4+3 < p.M*p.N*p.K*4 := by
  exact Index64.all_input_addresses p s pm pn c i j e

#axiomsClean repaired_returned_fp32_correct
#axiomsClean repaired_logical_returned_fp32_correct
#axiomsClean repaired_all_native_input_addresses
end
end VeriTile.Bench.Repairs.BatchedVecmatMult.HistoricalFPEquivalence
