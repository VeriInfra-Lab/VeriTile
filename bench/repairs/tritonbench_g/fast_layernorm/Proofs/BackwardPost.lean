import bench.repairs.tritonbench_g.fast_layernorm.Proofs.SuccessfulBackward
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.BackwardFormula
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.ReturnShape

namespace VeriTile.Bench.Repairs.FastLayernorm.BackwardCall
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host BackwardPreparation BackwardLayout
noncomputable section
open scoped Classical
variable {Store Metadata : Type} {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32}
  {inputs : List Storage.Tensor} {platform : Compiler.Platform}
  {lib : APIHost api neg inputs Metadata} {service : BackwardRuntime.Service platform api neg inputs Metadata}
  {identity : Nat} {c : SavedContext} {saved : SavedLayout c} {dy : Storage.Tensor} {initial : State Store Metadata}

theorem db_return_metadata (r : BodyRoute lib service c saved dy initial)
    (db : Dense) (out : Storage.Tensor) (hdb : r.preparation.data.db=some db)
    (hout : r.output.2.2.1=some out) :
    out.view.shape=c.biasShape.getD [] ∧ out.view.device=c.biasDevice.getD 0 ∧ out.view.shape.prod=db.count := by
  have info := GradientReturn.db_metadata r.returns db out hdb hout
  obtain ⟨original,hbias,allocated⟩ := BackwardLayout.optional_some r.preparation.db db hdb
  have owner := (lib.emptyLikeResult original r.preparation.sW db allocated).2.2
  exact ⟨info.1,info.2.1.trans (db.tensor.owner.trans (owner.trans (saved.biasOwner original hbias).symm)),info.2.2⟩

theorem post_values (r : SuccessfulCall lib service identity c saved dy initial) (t : Storage.Tensor)
    (member : t∈GradientReturn.returned r.body.output) (i : Nat) (hi : i<t.view.shape.prod) :
    Storage.logical api neg r.final.memory t i=Storage.logical api neg r.body.final.memory t i :=
  (lib.leave r.accessState.owner).transition r.body.final |>.values _ (GradientReturn.return_live r.body.returns t member) i hi

theorem post_real (r : SuccessfulCall lib service identity c saved dy initial) (t : Storage.Tensor)
    (member : t∈GradientReturn.returned r.body.output) (m : Storage.World (Nat → ℝ))
    (i : Nat) (hi : i<t.view.shape.prod) :
    Storage.realLogical (r.realFinal m) t i=Storage.realLogical (r.body.realFinal (r.preTransition.shadow m)) t i :=
  (lib.leave r.accessState.owner).transition r.body.final |>.realValues _ _
    (GradientReturn.return_live r.body.returns t member) i hi

theorem dx_native (r : SuccessfulCall lib service identity c saved dy initial)
    (i : Fin r.body.preparation.data.dx.count) :
    Storage.logical api neg r.final.memory r.body.output.1 i.val=
      BackwardRuntime.sourceValue c (platform.scalar r.body.dx.code.object.primitives) r.body.dx.code.plan api
        r.body.preparation.data .dx r.body.dx.s2.memory 0 i.val := by
  have hs := (dx_return_storage r.body).2
  have hi : i.val<r.body.output.1.view.shape.prod := by
    rw [hs,saved.inputShape,←(route_layout r.body.preparation saved).dxCount]
    exact i.isLt
  exact (post_values r _ (by simp [GradientReturn.returned]) i.val hi).trans (body_native_dx r.body i)

theorem dw_native (r : SuccessfulCall lib service identity c saved dy initial)
    (i : Fin r.body.preparation.data.dw.count) :
    Storage.logical api neg r.final.memory r.body.output.2.1 i.val=
      BackwardRuntime.sourceValue c (platform.scalar r.body.affine.code.object.primitives) r.body.affine.code.plan api
        r.body.preparation.data .affine r.body.affine.s2.memory 0 i.val := by
  have hs := (dw_return_metadata r.body).1
  have hi : i.val<r.body.output.2.1.view.shape.prod := by
    rw [hs,saved.weightShape,←(route_layout r.body.preparation saved).dwCount]
    exact i.isLt
  exact (post_values r _ (by simp [GradientReturn.returned]) i.val hi).trans (body_native_dw r.body i)

theorem db_native (r : SuccessfulCall lib service identity c saved dy initial) (db : Dense) (out : Storage.Tensor)
    (hdb : r.body.preparation.data.db=some db) (hout : r.body.output.2.2.1=some out)
    (i : Fin db.count) :
    Storage.logical api neg r.final.memory out i.val=
      BackwardRuntime.sourceValue c (platform.scalar r.body.affine.code.object.primitives) r.body.affine.code.plan api
        r.body.preparation.data .affine r.body.affine.s2.memory 1 i.val :=
  (post_values r out (by simp [GradientReturn.returned,hout]) i.val
    (by rw [(GradientReturn.db_metadata r.body.returns db out hdb hout).2.2]; exact i.isLt)).trans
      (body_native_db r.body db out hdb hout i)

#axiomsClean dx_native
#axiomsClean dw_native
#axiomsClean db_native
#axiomsClean post_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.BackwardCall
