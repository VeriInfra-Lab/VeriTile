import bench.repairs.tritonbench_g.fast_layernorm.Kernels
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.PublicDomain
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.Whole

/-! Independent total Real LayerNorm and first-gradient formulas, on the same
native call routes, allocations, compiler selections and schedules. The Real
caller memory and epsilon are arbitrary; no FP decoder or second successful
Real invocation appears. Real sqrt/division are total. These formulas do not
claim differentiability at singular or negative-variance-plus-epsilon points.

For each row: μ=Σx/N, v=Σ(x-μ)^2/N, r=1/sqrt(v+ε), z=(x-μ)r,
y=z·w+b, dx=r·(dy·w-mean(dy·w)-z·mean(dy·w·z)). Affine gradients
sum dy·z and dy over rows; every extra affine cell is zero. Saved μ/r and
saved X/W are proved from the actual preceding forward stores and contiguous/view calls.
Only the ordinary saved-tensor version receipt preserves them between calls. -/
namespace VeriTile.Bench.Repairs.FastLayernorm.HostCorrectness
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host Preparation Wrapper
noncomputable section
open scoped Classical
variable {Store Metadata : Type} {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32}
  {caller : List Storage.Tensor} {platform : Compiler.Platform}
  {lib : APIHost api neg caller Metadata} {metadata : MetadataAPI api neg caller Metadata}
  {service : Runtime.Service platform api neg caller Metadata} {identity : Nat}
  {module : Module} {x : Storage.Tensor} {initial : State Store Metadata}
  {observations : Observations lib}

theorem python_checks_iff_preconditions (x w : VeriTile.Torch.ArgumentMetadata)
    (bias : Option VeriTile.Torch.ArgumentMetadata) (eps : Contract.Epsilon) :
    Contract.checkArguments x w bias eps=.ok () ↔ Contract.AdmittedArguments x w bias eps :=
  Contract.checkArguments_iff x w bias eps

theorem python_backward_checks_iff_preconditions (dy x : VeriTile.Torch.TensorMetadata) :
    Contract.checkGradient dy x=.ok () ↔ Contract.GradientContract dy x :=
  Contract.checkGradient_iff dy x

specification fast_layernorm_repaired_real_returns (domain : Domain observations module x caller initial) (out : Dense)
    (ok : ((sourceForward lib metadata service identity module x).run initial).result=.ok out) :
    ∃ f : ForwardCall.SuccessfulCall lib metadata service identity x module.weight module.bias module.selectedEpsilon initial,
      f.body.output=out ∧ ((sourceForward lib metadata service identity module x).run initial).state=f.final ∧
      out.tensor.backing=f.body.preparation.y.tensor.backing ∧ out.tensor.view.shape=x.view.shape ∧
      (∀ m : Storage.World (Nat → ℝ),∀ epsilon : ℝ,∀ i : Fin x.view.shape.prod,
        Storage.realLogical (f.realFinal m epsilon) out.tensor i.val=
          RealProof.outputSpec f.shape.columns (ForwardCall.logicalX m x f.shape.columns (i.val/f.shape.columns))
            (ForwardCall.logicalW m module.weight) (ForwardCall.logicalBias m module.bias) epsilon (i.val%f.shape.columns)) := by
  obtain ⟨_,f,he,hs⟩ := ForwardCall.public_route out ok
  subst out
  exact ⟨f,rfl,hs,(ForwardCall.output_storage f).1,(ForwardCall.output_storage f).2.1,
    fun m epsilon i => ForwardCall.forward_logical_real f domain.xLive domain.weightLive domain.biasLive m epsilon i⟩

variable {w : Storage.Tensor} {bias : Option Storage.Tensor} {eps : Compiler.Shared.HostScalar}
  (f : ForwardCall.SuccessfulCall lib metadata service identity x w bias eps initial)
  {dy : Storage.Tensor} {current : State Store Metadata}
  {backLib : APIHost api neg (Whole.backwardInputs caller f.body.context dy) Metadata}
  {backService : BackwardRuntime.Service platform api neg (Whole.backwardInputs caller f.body.context dy) Metadata}

specification fast_layernorm_repaired_saved_state
    (xPresent : initial.live x) (wPresent : initial.live w) :
    lib.context f.final.metadata identity=some f.body.context ∧
    (∀ m : Storage.World (Nat → ℝ),∀ epsilon : ℝ,
      (∀ i : Fin x.view.shape.prod,Storage.realLogical (f.realFinal m epsilon) f.body.context.x.tensor i.val=
        Storage.realLogical m x i.val) ∧
      (∀ i : Fin w.view.shape.prod,Storage.realLogical (f.realFinal m epsilon) f.body.context.w.tensor i.val=
        Storage.realLogical m w i.val) ∧
      (∀ row,row<f.shape.rows →
        Storage.realLogical (f.realFinal m epsilon) f.body.context.mu.tensor row=
          RealProof.meanSpec f.shape.columns (ForwardCall.logicalX m x f.shape.columns row) ∧
        Storage.realLogical (f.realFinal m epsilon) f.body.context.r.tensor row=
          RealProof.invSpec f.shape.columns (ForwardCall.logicalX m x f.shape.columns row) epsilon)) := by
  exact ⟨ForwardCall.saved_context f,fun m epsilon =>
    ⟨ForwardCall.saved_x_real f xPresent m epsilon,ForwardCall.saved_w_real f wPresent m epsilon,
      fun row hr => ⟨ForwardCall.saved_mean_real f xPresent m epsilon row hr,
        ForwardCall.saved_inverse_real f xPresent m epsilon row hr⟩⟩⟩

/-- Independent formulas over original forward inputs and incoming logical dY.
The original physical view/alias relations are retained by owner-tagged memory. -/
def RealGradients (version : Whole.SavedVersion api neg f.body.context f.final current)
    (b : Whole.BackCall f (current:=current) (backLib:=backLib) (backService:=backService)) : Prop :=
  ∀ m : Storage.World (Nat → ℝ),∀ epsilon : ℝ,
    (∀ i : Fin b.body.preparation.data.dx.count,
      Storage.realLogical (Whole.realFinal f version b m epsilon) b.body.output.1 i.val=
        RealProof.gradientSpec f.shape.columns (ForwardCall.logicalX m x f.shape.columns (i.val/f.shape.columns))
          (Whole.incoming f (dy:=dy) version m epsilon (i.val/f.shape.columns)) (ForwardCall.logicalW m w)
          (RealProof.meanSpec f.shape.columns (ForwardCall.logicalX m x f.shape.columns (i.val/f.shape.columns)))
          (RealProof.invSpec f.shape.columns (ForwardCall.logicalX m x f.shape.columns (i.val/f.shape.columns)) epsilon)
          (i.val%f.shape.columns)) ∧
    (∀ i : Fin b.body.preparation.data.dw.count,
      Storage.realLogical (Whole.realFinal f version b m epsilon) b.body.output.2.1 i.val=
        RealProof.weightGradientSpec f.shape.columns f.shape.rows (ForwardCall.logicalX m x f.shape.columns)
          (Whole.incoming f (dy:=dy) version m epsilon)
          (fun row => ⟨RealProof.meanSpec f.shape.columns (ForwardCall.logicalX m x f.shape.columns row),
            RealProof.invSpec f.shape.columns (ForwardCall.logicalX m x f.shape.columns row) epsilon⟩) i.val) ∧
    (∀ db out,b.body.preparation.data.db=some db → b.body.output.2.2.1=some out → ∀ i : Fin db.count,
      Storage.realLogical (Whole.realFinal f version b m epsilon) out i.val=
        RealProof.biasGradientSpec f.shape.columns f.shape.rows (Whole.incoming f (dy:=dy) version m epsilon) i.val)

specification fast_layernorm_repaired_backward_real
    (backObservations : Observations backLib)
    (_gradient : GradientDomain backObservations f.body.context dy)
    (version : Whole.SavedVersion api neg f.body.context f.final current)
    (lookup : backLib.context current.metadata identity=some f.body.context)
    (xPresent : initial.live x) (wPresent : initial.live w) (dyPresent : current.live dy)
    (out : GradientReturn.Gradients)
    (ok : (sourceBackward backLib backService identity f.body.context (ForwardCall.body_saved_layout f.body) dy current lookup).outcome.result=.ok out) :
    ∃ b : Whole.BackCall f (current:=current) (backLib:=backLib) (backService:=backService),
      b.body.output=out ∧
      (sourceBackward backLib backService identity f.body.context (ForwardCall.body_saved_layout f.body) dy current lookup).outcome.state=b.final ∧
      out.2.2.2=none ∧ RealGradients f version b := by
  obtain ⟨b,he,hs⟩ := BackwardCall.invoke_route lookup out ok
  refine ⟨b,he,hs,?_,?_⟩
  · rw [←he]; rfl
  · intro m epsilon
    exact ⟨Whole.dx_real f version b xPresent wPresent dyPresent m epsilon,
      Whole.dw_real f version b xPresent dyPresent m epsilon,
      fun db out hdb hout i => Whole.db_real f version b db out hdb hout dyPresent m epsilon i⟩

specification fast_layernorm_repaired_backward_frame
    (lookup : backLib.context current.metadata identity=some f.body.context) (key : Key)
    (hk : Storage.caller (Whole.backwardInputs caller f.body.context dy) key) :
    Storage.observe api (sourceBackward backLib backService identity f.body.context
      (ForwardCall.body_saved_layout f.body) dy current lookup).outcome.state.memory key=
      Storage.observe api current.memory key := BackwardCall.all_outcome_frame lookup key hk

#axiomsClean fast_layernorm_repaired_real_returns
#axiomsClean fast_layernorm_repaired_saved_state
#axiomsClean fast_layernorm_repaired_backward_real
#axiomsClean fast_layernorm_repaired_backward_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.HostCorrectness
