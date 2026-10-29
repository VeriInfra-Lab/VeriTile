import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.Host
import bench.tritonbench_g.batched_vecmat_mult.Proofs.LocalReduction

namespace VeriTile.Bench.Repairs.BatchedVecmatMult.Whole
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
open ValueProofs.Host
noncomputable section
local instance hostFPWordInhabited : Inhabited ValueProofs.Word := ⟨FP.Value.zero .fp32⟩

/-- Complete host outcome, including native setup errors, exact finite indices,
actual return identity, all output coordinates and pre-call memory preservation.
Only the raw-vector/carry/tree primitive law is numerical. Allocation frame and
freshness and successful setup ABI are ordinary local API contracts. -/
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
  cases hp : prepare allocate a initial.mem with
  | error e => exact prepare_error_precedes_setup allocate setup rest cfg plans a initial e hp
  | ok l =>
    simp only []
    cases hs : setup l with
    | error e => exact setup_failure_after_prepare allocate setup rest cfg plans a initial l e hp hs
    | ok u =>
      cases u
      have h := SourceGraph.Local.returned_tensor_local_correct
        allocate rest cfg D law plans a initial abi
      simp only [hp] at h
      obtain ⟨result,he,hr,hc,hregion,hvalue,hframe⟩ := h
      have executed := admitted_run_eq allocate setup rest cfg plans a initial l hp hs
      rw [he] at executed
      refine ⟨launch_integer_agreement l (storage l hp) (native l hs),
        result,executed,hr,hc,hregion,hvalue,hframe,?_,?_⟩
      · intro r o outside
        exact (hframe r o (outside_grid l.parameters r o outside)).trans
          (prepare_frame allocate a initial.mem l hp frame r o outside)
      · intro r o live
        have outside := protected_outside allocate a initial.mem l hp liveCells fresh r o live
        exact (hframe r o (outside_grid l.parameters r o outside)).trans
          (prepare_frame allocate a initial.mem l hp frame r o outside)

#axiomsClean returned_fp32_correct
end
end VeriTile.Bench.Repairs.BatchedVecmatMult.Whole
