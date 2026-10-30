import bench.tritonbench_g.fused_recurrent_retention.APISource
import bench.tritonbench_g.fused_recurrent_retention.Proofs.HostPlans
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section
set_option linter.unusedVariables false

/- Rank-four tensor handles expose actual storage offsets/strides; wrapper
identity is separate from storage identity, so detach can preserve storage. -/
theorem detach_storage (x : TensorObject) (id : Nat) :
    (detach x id).view=x.view ∧ (detach x id).requiresGrad=Bool.false := ⟨rfl,rfl⟩

/- This handle is returned by new_empty/sum: offset zero, fresh allocation,
original FP32 dtype/device and canonical contiguous strides even for T=0. -/
def executeForward (a : Arithmetic α) (s : Shape) (plans : Plans) (args : ForwardArguments)
    (alloc : Allocation 3 caller) (m : Memory α) : ForwardReturn α where
  memory := forwardRun a s plans args.buffers alloc args.seed.isSome args.outputFinal m
  output := resultTensor (alloc.storage 2) args.q.view.device s.B s.H s.T s.DV
  finalState := if args.outputFinal then some (resultTensor (alloc.storage 1) args.q.view.device s.B s.H s.DK s.DV) else none
  context := args.saved

/- Backward is a separate call with a new memory snapshot. do and F are actual
post-forward autograd arguments; default materialized zero tensors and expanded
or strided cotangents are all ordinary tensor inputs. -/
def executeBackward (a : Arithmetic α) (s : Shape) (plans : Plans) (ctx : SavedContext)
    (do_ : Tensor4) (final : Option Tensor4) (alloc : Allocation 6 caller) (m : Memory α) : BackwardReturn α where
  memory := backwardRun a s plans (ctx.buffers do_ final) alloc ctx.initial.isSome final.isSome m
  dq := resultTensor (alloc.storage 3) ctx.q.view.device s.B s.H s.T s.DK
  dk := resultTensor (alloc.storage 4) ctx.q.view.device s.B s.H s.T s.DK
  dv := resultTensor (alloc.storage 5) ctx.q.view.device s.B s.H s.T s.DV
  initialGradient := none
  outputFlagGradient := none

theorem saved_identity (a : Arithmetic α) (s : Shape) (plans : Plans) (args : ForwardArguments)
    (alloc : Allocation 3 caller) (m : Memory α) :
    let ctx := (executeForward a s plans args alloc m).context
    ctx.q=args.q ∧ ctx.k=args.k ∧ ctx.v=args.v ∧
      ctx.initial=args.seed.map (fun x => detach x args.detachedIdentity) ∧ ctx.outputFinal=args.outputFinal :=
  ⟨rfl,rfl,rfl,rfl,rfl⟩

theorem no_initial_gradient (a : Arithmetic α) (s : Shape) (plans : Plans) (ctx : SavedContext)
    (do_ : Tensor4) (final : Option Tensor4) (alloc : Allocation 6 caller) (m : Memory α) :
    (executeBackward a s plans ctx do_ final alloc m).initialGradient=none ∧
    (executeBackward a s plans ctx do_ final alloc m).outputFlagGradient=none := ⟨rfl,rfl⟩

theorem saved_values (ctx : SavedContext) (s : Shape) (do_ : Tensor4) (final : Option Tensor4)
    (forwardSnapshot backwardSnapshot : Memory α) (same : ctx.Unchanged forwardSnapshot backwardSnapshot) :
    let before := (ctx.buffers do_ final).observe s forwardSnapshot
    let after := (ctx.buffers do_ final).observe s backwardSnapshot
    after.q=before.q ∧ after.k=before.k ∧ after.v=before.v ∧ after.seed=before.seed := by
  refine ⟨?_,?_,?_,?_⟩ <;> funext h t d <;> apply same <;>
    simp [SavedContext.savedStorage,SavedContext.buffers,InputBuffers.qAddress,InputBuffers.kAddress,
      InputBuffers.vAddress,InputBuffers.seedAddress]

theorem forward_return_fresh (a : Arithmetic α) (s : Shape) (plans : Plans) (args : ForwardArguments)
    (alloc : Allocation 3 caller) (m : Memory α) :
    (executeForward a s plans args alloc m).output.view.storage∉caller := alloc.fresh 2

theorem backward_returns_fresh (a : Arithmetic α) (s : Shape) (plans : Plans) (ctx : SavedContext)
    (do_ : Tensor4) (final : Option Tensor4) (alloc : Allocation 6 caller) (m : Memory α) :
    let ret := executeBackward a s plans ctx do_ final alloc m
    ret.dq.view.storage∉caller ∧ ret.dk.view.storage∉caller ∧ ret.dv.view.storage∉caller ∧
      ret.dq.view.storage≠ret.dk.view.storage ∧ ret.dk.view.storage≠ret.dv.view.storage ∧ ret.dq.view.storage≠ret.dv.view.storage :=
  ⟨alloc.fresh 3,alloc.fresh 4,alloc.fresh 5,alloc.ne 3 4 (by decide),alloc.ne 4 5 (by decide),alloc.ne 3 5 (by decide)⟩

#axiomsClean detach_storage
#axiomsClean saved_identity
#axiomsClean no_initial_gradient
#axiomsClean saved_values
#axiomsClean forward_return_fresh
#axiomsClean backward_returns_fresh
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
