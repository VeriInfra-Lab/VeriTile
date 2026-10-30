import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.PublicSource
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.Whole

/-! Independent total Real formulas on the actual native call route, with an
arbitrary Real caller memory and epsilon. No decoding of FP words into exact
Reals or second successful invocation is assumed. For each row:
μ=Σx/N, v=Σ(x-μ)^2/N, r=1/sqrt(v+ε), z=(x-μ)r, Y=zW+B;
DX=(DY W-z mean(DY W z)-mean(DY W))r, DW=Σrows DY z, DB=Σrows DY.
Real sqrt/division are total; this is not a differentiability assertion at
singular inputs. Saved X/W/μ/r are derived from the actual forward route.
The saved-version receipt only preserves ordinary saved handle coordinates.
-/
namespace VeriTile.Bench.Repairs.LayerNormTriton.Correctness
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

specification layer_norm_repaired_forward_real (domain : Domain x w bias caller initial) (out : Dense)
    (ok : ((sourceForward lib service identity x w bias normalized eps).run initial).result=.ok out) :
    ∃ f : Host.Call lib service identity x w bias normalized eps initial,
      f.body.output=out ∧ ((sourceForward lib service identity x w bias normalized eps).run initial).state=f.final ∧
      out.tensor.backing=f.body.preparation.data.y.tensor.backing ∧ out.tensor.view.shape=x.view.shape ∧
      f.final.owner=initial.owner ∧
      (∀ m : Storage.World (Nat → ℝ),∀ epsilon : ℝ,∀ i : Fin f.body.preparation.data.y.count,
        Storage.realLogical (f.realFinal m epsilon) out.tensor i.val=
          RealProof.Spec.output f.shape.columns (Host.logicalRow m x f.shape.columns (i.val/f.shape.columns))
            (Storage.realLogical m w) (Storage.realLogical m bias) epsilon (i.val%f.shape.columns)) := by
  obtain ⟨f,he,hs⟩ := Host.public_route out ok
  subst out
  exact ⟨f,rfl,hs,(Host.returned_identity f).1,(Host.returned_identity f).2.1,(Host.returned_identity f).2.2,
    Host.returned_real f domain.xLive domain.wLive domain.bLive⟩

variable (f : Host.Call lib service identity x w bias normalized eps initial)
  {dy : Storage.Tensor} {current : State Store Metadata}

def savedBias : Dense := (Host.saved_layout f).choose
theorem savedStructure : BackwardHost.Saved f.body.context (savedBias f) := (Host.saved_layout f).choose_spec

abbrev backwardInputs := VeriTile.Bench.Repairs.FastLayernorm.Whole.backwardInputs
variable {backLib : APIHost api neg (backwardInputs caller f.body.context dy) Metadata}
  {extra : BackwardHost.APIExtra api neg (backwardInputs caller f.body.context dy) Metadata}
  {backService : BackwardHost.Service platform api neg (backwardInputs caller f.body.context dy) Metadata}

abbrev BackCall := Whole.BackCall f (savedStructure f) (current:=current) (dy:=dy)
  (backLib:=backLib) (extra:=extra) (backService:=backService)

def RealGradients (version : Whole.SavedVersion api neg f.body.context f.final current)
    (back : BackCall f (current:=current) (dy:=dy) (backLib:=backLib) (extra:=extra) (backService:=backService)) : Prop :=
  ∀ m : Storage.World (Nat → ℝ),∀ epsilon : ℝ,
    (∀ i : Fin back.body.preparation.data.base.dx.count,
      Storage.realLogical (Whole.realFinal f (savedStructure f) version back m epsilon) back.body.returns.dx.tensor i.val=
        RealProof.Spec.gradient f.shape.columns (Host.logicalRow m x f.shape.columns (i.val/f.shape.columns))
          (Whole.incoming f (dy:=dy) version m epsilon (i.val/f.shape.columns)) (Storage.realLogical m w)
          (RealProof.Spec.mean f.shape.columns (Host.logicalRow m x f.shape.columns (i.val/f.shape.columns)))
          (RealProof.Spec.inverse f.shape.columns (Host.logicalRow m x f.shape.columns (i.val/f.shape.columns)) epsilon)
          (i.val%f.shape.columns)) ∧
    (∀ i : Fin back.body.preparation.data.base.dw.count,
      Storage.realLogical (Whole.realFinal f (savedStructure f) version back m epsilon) back.body.returns.wr.output i.val=
        RealProof.Spec.weightGradient f.shape.columns f.shape.rows (Host.logicalRow m x f.shape.columns)
          (Whole.incoming f (dy:=dy) version m epsilon)
          (fun row => ⟨RealProof.Spec.mean f.shape.columns (Host.logicalRow m x f.shape.columns row),
            RealProof.Spec.inverse f.shape.columns (Host.logicalRow m x f.shape.columns row) epsilon⟩) i.val) ∧
    (∀ i : Fin back.body.preparation.data.base.db.count,
      Storage.realLogical (Whole.realFinal f (savedStructure f) version back m epsilon) back.body.returns.br.output i.val=
        RealProof.Spec.biasGradient f.shape.columns f.shape.rows (Whole.incoming f (dy:=dy) version m epsilon) i.val)

specification layer_norm_repaired_backward_real
    (version : Whole.SavedVersion api neg f.body.context f.final current)
    (lookup : backLib.context current.metadata identity=some f.body.context)
    (xp : initial.live x) (wp : initial.live w) (dyp : current.live dy) (out : List (Option Storage.Tensor))
    (ok : (sourceBackward backLib extra backService identity f.body.context (savedBias f) (savedStructure f) dy current lookup).outcome.result=.ok out) :
    ∃ back : BackCall f (current:=current) (dy:=dy) (backLib:=backLib) (extra:=extra) (backService:=backService),
      back.body.output=out ∧
      (sourceBackward backLib extra backService identity f.body.context (savedBias f) (savedStructure f) dy current lookup).outcome.state=back.final ∧
      out=[some back.body.returns.dx.tensor,none,some back.body.returns.wr.output,some back.body.returns.br.output,none] ∧
      RealGradients f version back := by
  obtain ⟨back,he,hs⟩ := BackwardHost.public_route lookup out ok
  refine ⟨back,he,hs,he.symm,?_⟩
  intro m epsilon
  exact ⟨Whole.dx_real f (savedStructure f) version back xp wp dyp m epsilon,
    Whole.dw_real f (savedStructure f) version back xp dyp m epsilon,
    Whole.db_real f (savedStructure f) version back dyp m epsilon⟩

specification layer_norm_repaired_backward_frame
    (lookup : backLib.context current.metadata identity=some f.body.context) (key : Key)
    (hk : Storage.caller (backwardInputs caller f.body.context dy) key) :
    Storage.observe api (sourceBackward backLib extra backService identity f.body.context (savedBias f) (savedStructure f) dy current lookup).outcome.state.memory key=
      Storage.observe api current.memory key := BackwardHost.all_outcome_frame lookup key hk

#axiomsClean layer_norm_repaired_forward_real
#axiomsClean layer_norm_repaired_backward_real
#axiomsClean layer_norm_repaired_backward_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.Correctness
