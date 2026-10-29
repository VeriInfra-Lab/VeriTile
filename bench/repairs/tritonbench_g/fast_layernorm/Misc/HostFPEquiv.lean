import bench.repairs.tritonbench_g.fast_layernorm.Kernels
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.PublicDomain
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.FullForwardNative
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.BackwardPost
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.Numerics
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.Readiness

/-! Whole default-FP32 source equivalence to the selected ordered scalar graph.
All UInt32 payloads are admitted, including exceptional values. The selected
per-lane trees, operand swaps, FMA decisions, N=1 variants, source warp counts,
epsilon ABI and actual saved statistics are retained. No reassociation or
finite-only hypothesis is used. Explicit cast/store precision is in Kernels.

Target-bit conformance remains Compiler.LocalConformance plus
Numerics.EpsilonConformance and TensorNegation. These are local scalar
obligations; they cannot discharge a memory, shape, index, launch, protocol,
saved-value or kernel-result property. No native/GPU conformance is claimed.
The independent Real reader is not imported by this entry. -/
namespace VeriTile.Bench.Repairs.FastLayernorm.HostEquivalence
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host Preparation Wrapper BackwardPreparation
noncomputable section
open scoped Classical
variable {Store Metadata : Type} {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32}
  {caller : List Storage.Tensor} {platform : Compiler.Platform}
  {lib : APIHost api neg caller Metadata} {metadata : MetadataAPI api neg caller Metadata}
  {service : Runtime.Service platform api neg caller Metadata} {identity : Nat}
  {module : Module} {x : Storage.Tensor} {initial : State Store Metadata}

theorem python_checks_iff_preconditions (x w : VeriTile.Torch.ArgumentMetadata)
    (bias : Option VeriTile.Torch.ArgumentMetadata) (eps : Contract.Epsilon) :
    Contract.checkArguments x w bias eps=.ok () ↔ Contract.AdmittedArguments x w bias eps :=
  Contract.checkArguments_iff x w bias eps

theorem python_backward_checks_iff_preconditions (dy x : VeriTile.Torch.TensorMetadata) :
    Contract.checkGradient dy x=.ok () ↔ Contract.GradientContract dy x :=
  Contract.checkGradient_iff dy x

specification fast_layernorm_tensor_binding (observations : Observations lib)
    (domain : Domain observations module x caller initial) (out : Dense)
    (ok : ((sourceForward lib metadata service identity module x).run initial).result=.ok out) :
    ∃ f : ForwardCall.SuccessfulCall lib metadata service identity x module.weight module.bias module.selectedEpsilon initial,
      f.body.output=out ∧ out.tensor.view.shape=x.view.shape ∧
      out.tensor.view.dtype=0 ∧ out.tensor.view.device=x.view.device ∧
      SameStorage f.body.preparation.data.x.tensor x ∧
      SameStorage f.body.preparation.data.w.tensor module.weight ∧
      (∀ original prepared,module.bias=some original → f.body.preparation.data.bias=some prepared →
        SameStorage prepared.tensor original) ∧
      (∀ t,t=x ∨ t=module.weight ∨ t∈module.bias.toList → ∀ offset,offset<t.backing.nbytes →
        Storage.observe api ((sourceForward lib metadata service identity module x).run initial).state.memory
          (t.backing.owner,t.backing.base+offset)=
          Storage.observe api initial.memory (t.backing.owner,t.backing.base+offset)) := by
  obtain ⟨_,f,he,_⟩ := ForwardCall.public_route out ok
  subst out
  obtain ⟨hx,hw,hb⟩ := forward_input_views observations domain f
  exact ⟨f,rfl,(ForwardCall.output_storage f).2.1,f.body.output.tensor.fp32,
    forward_output_device f,hx,hw,hb,forward_original_frame observations domain⟩

specification fast_layernorm_repaired_fp32_returns (observations : Observations lib)
    (_domain : Domain observations module x caller initial) (out : Dense)
    (ok : ((sourceForward lib metadata service identity module x).run initial).result=.ok out) :
    ∃ f : ForwardCall.SuccessfulCall lib metadata service identity x module.weight module.bias module.selectedEpsilon initial,
      f.body.output=out ∧ ((sourceForward lib metadata service identity module x).run initial).state=f.final ∧
      out.tensor.backing=f.body.preparation.y.tensor.backing ∧ out.tensor.view.shape=x.view.shape ∧
      f.final.owner=initial.owner ∧ Readiness.TensorReady f.final out.tensor ∧
      (∀ i : Fin f.body.preparation.data.y.count,
        Storage.logical api neg f.final.memory out.tensor i.val=
          forwardReference (platform.scalar f.body.dispatch.code.object.primitives) f.body.dispatch.code.plan
            (Execution.rows f.body.preparation.data.x f.shape.columns f.body.dispatch.s2.memory api (i.val/f.shape.columns))
            (Execution.vector f.body.preparation.data.w f.body.dispatch.s2.memory api)
            (Runtime.biasValues (platform.scalar f.body.dispatch.code.object.primitives) api f.body.preparation.data f.body.dispatch.s2.memory)
            (platform.epsilon f.body.dispatch.code.object module.selectedEpsilon) (i.val%f.shape.columns)) := by
  obtain ⟨_,f,he,hs⟩ := ForwardCall.public_route out ok
  subst out
  exact ⟨f,rfl,hs,(ForwardCall.output_storage f).1,(ForwardCall.output_storage f).2.1,
    ForwardCall.restored_owner f,Readiness.forward_return f,ForwardCall.forward_native f⟩

variable {c : SavedContext} {saved : SavedLayout c} {dy : Storage.Tensor}
  {backLib : APIHost api neg caller Metadata} {backService : BackwardRuntime.Service platform api neg caller Metadata}

/-- Readback is from the actual returned gradient handles, after both launches,
all views, and device-context exit. -/
def NativeGradients (b : BackwardCall.SuccessfulCall backLib backService identity c saved dy initial) : Prop :=
  (∀ i : Fin b.body.preparation.data.dx.count,
    Storage.logical api neg b.final.memory b.body.output.1 i.val=
      BackwardRuntime.sourceValue c (platform.scalar b.body.dx.code.object.primitives) b.body.dx.code.plan api
        b.body.preparation.data .dx b.body.dx.s2.memory 0 i.val) ∧
  (∀ i : Fin b.body.preparation.data.dw.count,
    Storage.logical api neg b.final.memory b.body.output.2.1 i.val=
      BackwardRuntime.sourceValue c (platform.scalar b.body.affine.code.object.primitives) b.body.affine.code.plan api
        b.body.preparation.data .affine b.body.affine.s2.memory 0 i.val) ∧
  (∀ db out,b.body.preparation.data.db=some db → b.body.output.2.2.1=some out → ∀ i : Fin db.count,
    Storage.logical api neg b.final.memory out i.val=
      BackwardRuntime.sourceValue c (platform.scalar b.body.affine.code.object.primitives) b.body.affine.code.plan api
        b.body.preparation.data .affine b.body.affine.s2.memory 1 i.val)

specification fast_layernorm_repaired_backward_fp32
    (backObservations : Observations backLib)
    (_gradient : GradientDomain backObservations c dy)
    (lookup : backLib.context initial.metadata identity=some c) (out : GradientReturn.Gradients)
    (ok : (sourceBackward backLib backService identity c saved dy initial lookup).outcome.result=.ok out) :
    ∃ b : BackwardCall.SuccessfulCall backLib backService identity c saved dy initial,
      b.body.output=out ∧ (sourceBackward backLib backService identity c saved dy initial lookup).outcome.state=b.final ∧
      out.1.backing=b.body.preparation.data.dx.tensor.backing ∧ out.1.view.shape=c.inputShape ∧
      out.2.1.view.shape=c.weightShape ∧ out.2.1.view.device=c.weightDevice ∧ out.2.2.2=none ∧
      b.final.owner=initial.owner ∧
      (∀ t,t∈GradientReturn.returned out → Readiness.TensorReady b.final t) ∧
      out.2.2.1.isSome=c.bias.isSome ∧
      (∀ db tensor,b.body.preparation.data.db=some db → out.2.2.1=some tensor →
        tensor.view.shape=c.biasShape.getD [] ∧ tensor.view.device=c.biasDevice.getD 0 ∧ tensor.view.shape.prod=db.count) ∧
      NativeGradients b := by
  obtain ⟨b,he,hs⟩ := BackwardCall.invoke_route lookup out ok
  subst out
  exact ⟨b,rfl,hs,(BackwardCall.dx_return_storage b.body).1,(BackwardCall.dx_return_storage b.body).2,
    (BackwardCall.dw_return_metadata b.body).1,(BackwardCall.dw_return_metadata b.body).2,rfl,
    BackwardCall.restored_owner b,Readiness.backward_return b,
    (GradientReturn.optional_presence b.body.returns.br).trans (BackwardLayout.route_layout b.body.preparation saved).dbPresence,
    BackwardCall.db_return_metadata b.body,
    BackwardCall.dx_native b,BackwardCall.dw_native b,fun db out hdb hout i => BackwardCall.db_native b db out hdb hout i⟩

specification fast_layernorm_repaired_all_outcome_frame (key : Key) (hk : Storage.caller caller key) :
    Storage.observe api ((sourceForward lib metadata service identity module x).run initial).state.memory key=
      Storage.observe api initial.memory key := ForwardCall.public_frame _ _ _ _ _ _ _ key hk

#axiomsClean fast_layernorm_repaired_fp32_returns
#axiomsClean fast_layernorm_tensor_binding
#axiomsClean fast_layernorm_repaired_backward_fp32
#axiomsClean fast_layernorm_repaired_all_outcome_frame
-- #print_fp_assumptions is restricted to registered FP-equivalence relations;
-- these existential whole-call specifications use the supported declaration
-- and module audits, with their exact numerical deferrals listed above.
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.HostEquivalence
