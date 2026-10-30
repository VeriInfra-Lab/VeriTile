import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.AdmissionBinding
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.RealResult

namespace VeriTile.Bench.Repairs.LayerNormTriton.Correctness
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
specification layer_norm_repaired_forward_real_checked (domain : Domain x w bias caller initial) (out : Dense)
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
      f.final.owner=initial.owner ∧
      (∀ m : Storage.World (Nat → ℝ),∀ epsilon : ℝ,∀ i : Fin f.body.preparation.data.y.count,
        Storage.realLogical (f.realFinal m epsilon) out.tensor i.val=
          RealProof.Spec.output f.shape.columns (Host.logicalRow m x f.shape.columns (i.val/f.shape.columns))
            (Storage.realLogical m w) (Storage.realLogical m bias) epsilon (i.val%f.shape.columns)) := by
  obtain ⟨contract,eq⟩ := Checked.forward_admitted lib observations service identity x w bias inputNormalized inputEps initial actual admitted
  subst actual
  obtain ⟨f,h⟩ := layer_norm_repaired_forward_real domain out ok
  exact ⟨f,h.1,Checked.forward_storage observations service inputNormalized inputEps contract f,h.2⟩

variable (f : Host.Call lib service identity x w bias normalized eps initial)
  {dy : Storage.Tensor} {current : State Store Metadata}
  {backLib : APIHost api neg (backwardInputs caller f.body.context dy) Metadata}
  {extra : BackwardHost.APIExtra api neg (backwardInputs caller f.body.context dy) Metadata}
  {backService : BackwardHost.Service platform api neg (backwardInputs caller f.body.context dy) Metadata}
specification layer_norm_repaired_backward_real_checked
    (version : Whole.SavedVersion api neg f.body.context f.final current)
    (lookup : backLib.context current.metadata identity=some f.body.context)
    (xp : initial.live x) (wp : initial.live w) (dyp : current.live dy) (out : List (Option Storage.Tensor))
    (observations : Checked.Observations backLib)
    (actual : Outcome (State Store Metadata) (List (Option Storage.Tensor)))
    (admitted : Checked.backward backLib observations extra backService identity f.body.context (savedBias f) (savedStructure f) dy current lookup=.ok actual)
    (ok : actual.result=.ok out) :
    ∃ back : BackCall f (current:=current) (dy:=dy) (backLib:=backLib) (extra:=extra) (backService:=backService),
      back.body.output=out ∧
      actual.state=back.final ∧
      out=[some back.body.returns.dx.tensor,none,some back.body.returns.wr.output,some back.body.returns.br.output,none] ∧
      RealGradients f version back := by
  obtain ⟨contract,eq⟩ := Checked.backward_admitted backLib observations extra backService identity f.body.context (savedBias f) (savedStructure f) dy current lookup actual out admitted ok
  subst actual
  exact layer_norm_repaired_backward_real f version lookup xp wp dyp out ok

#axiomsClean layer_norm_repaired_forward_real_checked
#axiomsClean layer_norm_repaired_backward_real_checked
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.Correctness
