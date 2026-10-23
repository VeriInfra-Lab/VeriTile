import bench.repairs.tritonbench_g.l2_norm_triton1.Kernels
import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.RealSource
import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.Evidence
import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.Binding

namespace VeriTile.Bench.Repairs.L2NormTriton1.Correct
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open Host RealExecution
noncomputable section
open Classical

/-- Every supported logical view and alias uses its original entry values.
The independent formula is x_i/sqrt(sum_j x_j^2+eps). Real sqrt/division are
total and epsilon is unrestricted; this theorem is not an FP error estimate. -/
def target {M β : Type} (ops : Ops M ℝ β) (call : Call ℝ)
    (entry : World M) (row col : Nat) : ℝ := reference ops call entry row col

structure ResultSpec {M β : Type} (ops : Ops M ℝ β) (rt : Runtime M)
    (call : Call ℝ) (entry : World M) (out : View) (after : World M) : Prop where
  whole : PublicFacts ops negation adapter rt call entry out after
  values : ∀ row col,row*(call.input.shape.getLastD 0)+col<call.input.numel →
    col<call.input.shape.getLastD 0 →
    logical ops negation after.memory out (row*(call.input.shape.getLastD 0)+col)=target ops call entry row col
  execution : ExecutionEvidence ops negation adapter rt call entry out after

theorem source_result {M β : Type} (ops : Ops M ℝ β) (rt : Runtime M)
    (lib : Library ops negation rt) (call : Call ℝ) (entry : World M)
    (domain : Domain call entry) (out : View) (after : World M)
    (success : execute ops adapter rt call entry=.ok out after) :
    ResultSpec ops rt call entry out after := by
  have facts := execute_success ops negation adapter rt lib call entry domain out after success
  refine ⟨facts,?_,execution_evidence ops negation adapter rt lib call entry domain out after facts.trace⟩
  intro row col active hc
  rw [facts.values row col active hc,expected_real ops call entry row col facts.accepted hc]
  rfl

/-- Full caller-byte frame for success and every failure prefix, including
partial private kernel stores and failed context restoration/return reshape. -/
theorem source_frame {M β : Type} (ops : Ops M ℝ β) (rt : Runtime M)
    (lib : Library ops negation rt) (call : Call ℝ) (entry : World M)
    (domain : Domain call entry) : Keep ops entry (execute ops adapter rt call entry).state :=
  execute_frame ops negation adapter rt lib call entry domain

/-- The concrete Python checks and public input premise describe the same domain. -/
theorem python_checks_iff (argument : ArgumentMetadata) :
    checkInputs argument=.ok () ↔ AdmittedArguments argument := checkInputs_iff argument

/-- Whole checked wrapper. TensorBinding, live storage and Library are ordinary
API semantics; the only input-domain premise is the checked InputContract. -/
theorem l2_host_real {M β : Type} (ops : Ops M ℝ β) (rt : Runtime M)
    (lib : Library ops negation rt) (metadata : TensorMetadata) (input : View) (eps : ℝ)
    (entry : World M) (h : InputContract metadata) (binding : TensorBinding metadata input)
    (live : input.storage∈entry.allocations) (out : View) (after : World M)
    (success : checkedExecute ops adapter rt metadata input eps entry=.ok (.ok out after)) :
    ResultSpec ops rt (checkedCall metadata input eps) entry out after := by
  rw [checked_execute ops adapter rt metadata input eps entry h] at success
  exact source_result ops rt lib (checkedCall metadata input eps) entry
    (checked_domain metadata input eps entry h binding live) out after (Except.ok.inj success)

theorem l2_real_all_outcomes {M β : Type} (ops : Ops M ℝ β) (rt : Runtime M)
    (lib : Library ops negation rt) (metadata : TensorMetadata) (input : View) (eps : ℝ)
    (entry : World M) (h : InputContract metadata) (binding : TensorBinding metadata input)
    (live : input.storage∈entry.allocations) :
    match checkedExecute ops adapter rt metadata input eps entry with
    | .error _ => True
    | .ok result => Keep ops entry result.state := by
  rw [checked_execute ops adapter rt metadata input eps entry h]
  exact source_frame ops rt lib (checkedCall metadata input eps) entry
    (checked_domain metadata input eps entry h binding live)

#axiomsClean python_checks_iff
#axiomsClean l2_host_real
#axiomsClean l2_real_all_outcomes
#specNonCircular target avoiding [Tile.scalar,Tile.sourceRun,Host.execute,Physical.run,Host.expected]
#axiomsClean source_result
#axiomsClean source_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton1.Correct
