import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.PublicSource
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.ForwardFP
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.BackwardFP
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.Readiness

/-! Full default-FP32 return/readback specifications. The domain includes every
UInt32 payload, including NaN, infinities, subnormals and signed zero. Each
selected object retains its reduction trees, swaps, FMA choices, count/epsilon
ABI, source branch, load/store casts and N=1 operations. No reassociation,
commutation, FMA=mul+add or finite-only condition is assumed.

Numerics.Conformance is the remaining target-bit boundary: operationwise
zero/one/count/add/sub/mul/div/fma/sqrt, epsilon conversion and tensor negation.
It contains no kernel result, memory, successful-execution or protocol law.
The backward target expands actual scratch readback to the grouped first-launch
graph, then applies the independently selected final-launch graph. No whole
scratch/output equality is assumed. No GPU or target-bit conformance is claimed.
Global grid-product/PID/segmentation compatibility is separately deferred.
-/
namespace VeriTile.Bench.Repairs.LayerNormTriton.Equivalence
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm hiding Ops Plan
open VeriTile.Bench.Repairs.FastLayernorm.Host
open Storage
noncomputable section
variable {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32} {caller : List Storage.Tensor}
  {Metadata : Type} {platform : Compiler.Platform} {lib : APIHost api neg caller Metadata}
  {service : Host.ForwardService platform api neg caller Metadata} {identity : Nat}
  {x w bias : Storage.Tensor} {normalized : List Nat} {eps : Compiler.Shared.HostScalar}
  {initial : State Store Metadata}

specification layer_norm_repaired_forward_fp32 (out : Dense)
    (ok : ((sourceForward lib service identity x w bias normalized eps).run initial).result=.ok out) :
    ∃ f : Host.Call lib service identity x w bias normalized eps initial,
      f.body.output=out ∧ ((sourceForward lib service identity x w bias normalized eps).run initial).state=f.final ∧
      out.tensor.backing=f.body.preparation.data.y.tensor.backing ∧ out.tensor.view.shape=x.view.shape ∧
      f.final.owner=initial.owner ∧ Readiness.TensorReady f.final out.tensor ∧
      (∀ i : Fin f.body.preparation.data.y.count,
        Storage.logical api neg f.final.memory out.tensor i.val=f.body.launch.nativeValue i.val) ∧
      (∀ target : Numerics.Target,f.body.launch.conformance target → ∀ i : Fin f.body.preparation.data.y.count,
        Storage.logical api neg f.final.memory out.tensor i.val=f.body.launch.targetValue target i.val) := by
  obtain ⟨f,he,hs⟩ := Host.public_route out ok
  subst out
  exact ⟨f,rfl,hs,(Host.returned_identity f).1,(Host.returned_identity f).2.1,(Host.returned_identity f).2.2,
    Readiness.forward_return f,Host.returned_native f,Host.returned_fp f⟩

variable {c : SavedContext} {b : Dense} {saved : BackwardHost.Saved c b} {dy : Storage.Tensor}
  {extra : BackwardHost.APIExtra api neg caller Metadata}
  {backService : BackwardHost.Service platform api neg caller Metadata}

specification layer_norm_repaired_backward_fp32 (lookup : lib.context initial.metadata identity=some c)
    (out : List (Option Storage.Tensor))
    (ok : (sourceBackward lib extra backService identity c b saved dy initial lookup).outcome.result=.ok out) :
    ∃ back : BackwardHost.Call lib extra backService identity c b saved dy initial,
      back.body.output=out ∧ (sourceBackward lib extra backService identity c b saved dy initial lookup).outcome.state=back.final ∧
      out=[some back.body.returns.dx.tensor,none,some back.body.returns.wr.output,some back.body.returns.br.output,none] ∧
      back.body.returns.dx.tensor.backing=back.body.preparation.data.base.dx.tensor.backing ∧
      back.body.returns.dx.tensor.view.shape=c.inputShape ∧
      back.body.returns.wr.output.view.shape=c.weightShape ∧ back.body.returns.wr.output.view.device=c.x.tensor.view.device ∧
      back.body.returns.br.output.view.shape=c.biasShape.getD [] ∧ back.body.returns.br.output.view.device=c.x.tensor.view.device ∧
      back.final.owner=initial.owner ∧
      (∀ k : Fin 3,Readiness.TensorReady back.final (back.body.returns.outputs k)) ∧
      (∀ k : Fin 3,∀ i : Fin (BackwardHost.outputs back.body.preparation.data.base k).count,
        Storage.logical api neg back.final.memory (back.body.returns.outputs k) i.val=back.body.launch.nativeValue k i.val) ∧
      (∀ target : Numerics.Target,back.body.launch.conformance target → ∀ k : Fin 3,
        ∀ i : Fin (BackwardHost.outputs back.body.preparation.data.base k).count,
          Storage.logical api neg back.final.memory (back.body.returns.outputs k) i.val=back.body.launch.targetValue target k i.val) := by
  obtain ⟨back,he,hs⟩ := BackwardHost.public_route lookup out ok
  have metadataProof := BackwardHost.returned_metadata back.body.returns
  have device (k : Fin 3) : (BackwardHost.outputs back.body.preparation.data.base k).tensor.view.device=c.x.tensor.view.device := by
    exact (BackwardHost.outputs back.body.preparation.data.base k).tensor.owner.trans
      ((back.body.ready.base.owners k).trans c.x.tensor.owner.symm)
  exact ⟨back,he,hs,he.symm,metadataProof.1,metadataProof.2.1,metadataProof.2.2.1,metadataProof.2.2.2.1.trans (device 1),metadataProof.2.2.2.2.1,metadataProof.2.2.2.2.2.trans (device 2),
    BackwardHost.restored_owner back,Readiness.backward_return back,BackwardHost.returned_native back,BackwardHost.returned_fp back⟩

specification layer_norm_repaired_forward_frame (key : Key) (hk : Storage.caller caller key) :
    Storage.observe api ((sourceForward lib service identity x w bias normalized eps).run initial).state.memory key=
      Storage.observe api initial.memory key := (sourceForward lib service identity x w bias normalized eps).frame initial key hk

specification layer_norm_repaired_backward_frame (lookup : lib.context initial.metadata identity=some c)
    (key : Key) (hk : Storage.caller caller key) :
    Storage.observe api (sourceBackward lib extra backService identity c b saved dy initial lookup).outcome.state.memory key=
      Storage.observe api initial.memory key := BackwardHost.all_outcome_frame lookup key hk

#axiomsClean layer_norm_repaired_forward_fp32
#axiomsClean layer_norm_repaired_backward_fp32
#axiomsClean layer_norm_repaired_forward_frame
#axiomsClean layer_norm_repaired_backward_frame
-- #print_fp_assumptions accepts registered FP relations, not these existential
-- whole-call specifications. The exact local premises are visible above and
-- the declaration/module audits below reject nonstandard axioms and sorryAx.
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.Equivalence
