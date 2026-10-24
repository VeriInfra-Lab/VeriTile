import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.AdmissionBinding
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.NativeResult

namespace VeriTile.Bench.Repairs.LayerNormTriton.Equivalence
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm hiding Ops Plan
open VeriTile.Bench.Repairs.FastLayernorm.Host
open Storage
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
variable {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32} {caller : List Storage.Tensor}
  {Metadata : Type} {platform : Compiler.Platform} {lib : APIHost api neg caller Metadata}
  {service : Host.ForwardService platform api neg caller Metadata} {identity : Nat}
  {x w bias : Storage.Tensor} {normalized : List Nat} {eps : Compiler.Shared.HostScalar}
  {initial : State Store Metadata}
specification layer_norm_repaired_forward_fp32_checked (out : Dense)
    (observations : Checked.Observations lib) (inputNormalized : Option (List Nat)) (inputEps : Option Float)
    (actual : Outcome (State Store Metadata) Dense)
    (admitted : Checked.forward lib observations service identity x w bias inputNormalized inputEps initial=.ok actual)
    (ok : actual.result=.ok out) :
    ∃ f : Host.Call lib service identity x w bias (inputNormalized.getD []) (Checked.epsilon inputEps) initial,
      f.body.output=out ∧
      (f.body.preparation.data.x.tensor.backing=x.backing ∧
       f.body.preparation.data.w.tensor.backing=w.backing ∧
       f.body.context.weightDevice=f.body.context.x.tensor.view.device ∧
       f.body.context.biasDevice=some f.body.context.x.tensor.view.device) ∧ actual.state=f.final ∧
      out.tensor.backing=f.body.preparation.data.y.tensor.backing ∧ out.tensor.view.shape=x.view.shape ∧
      f.final.owner=initial.owner ∧ Readiness.TensorReady f.final out.tensor ∧
      (∀ i : Fin f.body.preparation.data.y.count,
        Storage.logical api neg f.final.memory out.tensor i.val=f.body.launch.nativeValue i.val) ∧
      (∀ target : Numerics.Target,f.body.launch.conformance target → ∀ i : Fin f.body.preparation.data.y.count,
        Storage.logical api neg f.final.memory out.tensor i.val=f.body.launch.targetValue target i.val) := by
  obtain ⟨contract,eq⟩ := Checked.forward_admitted lib observations service identity x w bias inputNormalized inputEps initial actual admitted
  subst actual
  obtain ⟨f,h⟩ := layer_norm_repaired_forward_fp32 out ok
  exact ⟨f,h.1,Checked.forward_storage observations service inputNormalized inputEps contract f,h.2⟩

variable {c : SavedContext} {b : Dense} {saved : BackwardHost.Saved c b} {dy : Storage.Tensor}
  {extra : BackwardHost.APIExtra api neg caller Metadata}
  {backService : BackwardHost.Service platform api neg caller Metadata}
specification layer_norm_repaired_backward_fp32_checked (lookup : lib.context initial.metadata identity=some c)
    (out : List (Option Storage.Tensor))
    (observations : Checked.Observations lib)
    (actual : Outcome (State Store Metadata) (List (Option Storage.Tensor)))
    (admitted : Checked.backward lib observations extra backService identity c b saved dy initial lookup=.ok actual)
    (ok : actual.result=.ok out) :
    ∃ back : BackwardHost.Call lib extra backService identity c b saved dy initial,
      back.body.output=out ∧ actual.state=back.final ∧
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
  obtain ⟨contract,eq⟩ := Checked.backward_admitted lib observations extra backService identity c b saved dy initial lookup actual out admitted ok
  subst actual
  exact layer_norm_repaired_backward_fp32 lookup out ok

theorem layer_norm_repaired_forward_checked_frame
    (observations : Checked.Observations lib) (inputNormalized : Option (List Nat)) (inputEps : Option Float)
    (actual : Outcome (State Store Metadata) Dense)
    (admitted : Checked.forward lib observations service identity x w bias inputNormalized inputEps initial=.ok actual)
    (key : Key) (hk : Storage.caller caller key) :
    Storage.observe api actual.state.memory key=Storage.observe api initial.memory key := by
  obtain ⟨_,eq⟩ := Checked.forward_admitted lib observations service identity x w bias inputNormalized inputEps initial actual admitted
  subst actual
  exact layer_norm_repaired_forward_frame key hk

theorem layer_norm_repaired_backward_checked_frame
    (lookup : lib.context initial.metadata identity=some c)
    (observations : Checked.Observations lib)
    (actual : Outcome (State Store Metadata) (List (Option Storage.Tensor))) (out : List (Option Storage.Tensor))
    (admitted : Checked.backward lib observations extra backService identity c b saved dy initial lookup=.ok actual)
    (ok : actual.result=.ok out) (key : Key) (hk : Storage.caller caller key) :
    Storage.observe api actual.state.memory key=Storage.observe api initial.memory key := by
  obtain ⟨_,eq⟩ := Checked.backward_admitted lib observations extra backService identity c b saved dy initial lookup actual out admitted ok
  subst actual
  exact layer_norm_repaired_backward_frame lookup key hk

#axiomsClean layer_norm_repaired_forward_checked_frame
#axiomsClean layer_norm_repaired_backward_checked_frame
#axiomsClean layer_norm_repaired_forward_fp32_checked
#axiomsClean layer_norm_repaired_backward_fp32_checked
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.Equivalence
