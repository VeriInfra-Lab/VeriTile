import bench.repairs.tritonbench_g.adam_update_triton.Proofs.PublicFacts

namespace VeriTile.Bench.Repairs.AdamUpdateTriton.HostCorrect
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open Host
noncomputable section

/-- Independent Lion formula on original logical entry values. The source is
named Adam but has one momentum and a sign update. Coefficients are arbitrary;
this is a Real interpretation of normalized scalars, not a Real/FP encoding. -/
def target {M β : Type} (ops : Ops M ℝ β) (params : Scalars ℝ)
    (call : Call) (before : World M) (output : Output) (i : Nat) : ℝ :=
  Math.target params output (logical ops Observation.realNegation before.memory call.request.p i)
    (logical ops Observation.realNegation before.memory call.request.grad i)
    (logical ops Observation.realNegation before.memory call.request.momentum i)

structure ResultSpec {M β : Type} (ops : Ops M ℝ β) (params : Scalars ℝ)
    (rt : Runtime M) (driver : ChunkRuntime M ℝ ℝ ops Machine.realModel rt)
    (convert : Scalar.Conversions ℝ) (call : Call) (args : Scalars (BitVec 64))
    (before after : World M) : Prop where
  whole : WholeSpec ops Observation.realNegation (Machine.realModel params) call.request before after
  values : ∀ output i,i<call.request.count →
    logical ops Observation.realNegation after.memory (outputView call.request output) i=target ops params call before output i
  gradient : ∀ i,i<call.request.grad.numel →
    logical ops Observation.realNegation after.memory call.request.grad i=
      logical ops Observation.realNegation before.memory call.request.grad i
  surplus : ∀ i,call.request.count≤ i → i<call.request.momentum.numel →
    logical ops Observation.realNegation after.memory call.request.momentum i=
      logical ops Observation.realNegation before.memory call.request.momentum i
  execution : Execution ops Machine.realModel rt driver convert call args params before after (fun _ _ _ _ _=>True)

theorem python_checks_iff (p g m : ArgumentMetadata) :
    checkInputs p g m=.ok () ↔ AdmittedArguments p g m := checkInputs_iff p g m

/-- Complete checked direct in-place call. Unit is Python None; original
objects and metadata remain the same. Every tuner trial/restore, both configs,
all chunks and masked tails are retained. Errors carry their actual states. -/
theorem lion_host_real {M β : Type} (ops : Ops M ℝ β) (readBytes : Observation.ReadByBytes ops)
    (rt : Runtime M) (lib : Library rt) (driver : ChunkRuntime M ℝ ℝ ops Machine.realModel rt)
    (convert : Scalar.Conversions ℝ) (call : Call) (h : Admitted call) (tensor : Bindings call)
    (args : Scalars (BitVec 64)) (params : Scalars ℝ)
    (normalized : call.request.count≠0 → Scalar.normalize convert call.arguments=.ok args)
    (binding : call.request.count≠0 → Scalar.bind convert args=.ok params) (before : World M) :
    Ensures (fun _ after=>ResultSpec ops params rt driver convert call args before after)
      (run rt driver convert call before) := by
  cases actual : run rt driver convert call before with
  | error => trivial
  | ok unit after =>
    cases unit
    have whole:=run_spec ops Observation.realNegation Machine.realModel rt lib driver convert call h tensor
      args params normalized binding before after actual
    refine ⟨whole,?_,?_,?_,execution ops Machine.realModel rt lib driver convert call h tensor
      args params normalized binding before after actual _ (fun _ _ _ _ _ _=>trivial)⟩
    · intro output i active
      simpa only [Machine.real_scalar,target] using whole.values output i active
    · intro i active
      exact read_only ops Observation.realNegation readBytes Machine.realModel rt lib driver convert call h tensor
        args params normalized binding before after actual whole .third i active (Or.inl rfl)
    · intro i lowerBound upperBound
      exact read_only ops Observation.realNegation readBytes Machine.realModel rt lib driver convert call h tensor
        args params normalized binding before after actual whole .b i upperBound (Or.inr lowerBound)

#specNonCircular target avoiding [Tile.scalar,Tile.sourceRun,Machine.scalar,Machine.run,run,pipeline]
#axiomsClean python_checks_iff
#axiomsClean lion_host_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.HostCorrect
