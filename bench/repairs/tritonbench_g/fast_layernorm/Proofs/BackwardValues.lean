import bench.repairs.tritonbench_g.fast_layernorm.Proofs.BackwardCall
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.GradientValues

namespace VeriTile.Bench.Repairs.FastLayernorm.BackwardCall
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host Execution BackwardPreparation BackwardLayout
noncomputable section
open scoped Classical
variable {Store Metadata : Type} {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32}
  {inputs : List Storage.Tensor} {platform : Compiler.Platform}
  {lib : APIHost api neg inputs Metadata} {service : BackwardRuntime.Service platform api neg inputs Metadata}
  {c : SavedContext} {saved : SavedLayout c} {dy : Storage.Tensor} {initial : State Store Metadata}

theorem optional_live {b : Option Dense} (r : OptionalAllocation lib initial b) (out : Dense)
    (h : r.output=some out) : r.final.live out.tensor := by
  cases r with
  | none => simp [OptionalAllocation.output] at h
  | @some source value ok =>
    have he : value=out := by simpa [OptionalAllocation.output] using h
    subst out
    exact (lib.emptyLikeResult source initial value ok).1.present

theorem preparation_present (r : BackwardPreparation.Route lib c dy initial) :
    r.final.live r.data.dy.tensor ∧ ∀ out,out∈outputs r.data → r.final.live out.tensor := by
  have hx := (lib.emptyLikeResult c.x r.dyr.final r.dx r.dxOK).1.present
  have hw := (lib.emptyLikeResult c.w r.sX r.dw r.dwOK).1.present
  constructor
  · exact r.tail.live _ (Preparation.materialize_metadata r.dyr).2.2.2
  · intro out ho
    cases hb : r.db.output with
    | none =>
      simp [outputs,BackwardPreparation.Route.data,hb] at ho
      rcases ho with rfl|rfl
      · exact r.db.transition.live _ ((lib.emptyLike c.w).transition r.sX |>.live _ hx)
      · exact r.db.transition.live _ hw
    | some db =>
      obtain ⟨b,hborig,hok⟩ := optional_some r.db db hb
      have present := (lib.emptyLikeResult b r.sW db hok).1.present
      simp [outputs,BackwardPreparation.Route.data,hb] at ho
      rcases ho with rfl|rfl|rfl
      · exact r.db.transition.live _ ((lib.emptyLike c.w).transition r.sX |>.live _ hx)
      · exact r.db.transition.live _ hw
      · exact optional_live r.db out hb

theorem dx_outputs_live (r : BodyRoute lib service c saved dy initial) (out : Dense)
    (member : out∈outputs r.preparation.data) : r.dx.final.live out.tensor :=
  r.dx.preTransition.live _ ((preparation_present r.preparation).2 out member)

theorem affine_outputs_live (r : BodyRoute lib service c saved dy initial) (out : Dense)
    (member : out∈outputs r.preparation.data) : r.affine.final.live out.tensor :=
  r.affine.preTransition.live _ (dx_outputs_live r out member)

theorem body_native_dx (r : BodyRoute lib service c saved dy initial) (i : Fin r.preparation.data.dx.count) :
    Storage.logical api neg r.final.memory r.output.1 i.val=
      BackwardRuntime.sourceValue c (platform.scalar r.dx.code.object.primitives) r.dx.code.plan api
        r.preparation.data .dx r.dx.s2.memory 0 i.val := by
  let layout := route_layout r.preparation saved
  have ho : r.preparation.data.dx.owner=c.x.owner := layout.owners _ (by simp [outputs])
  have hv := GradientReturn.dx_values r.returns i.val i.isLt
  change Storage.logical api neg r.final.memory r.output.1 i.val=_ at hv
  rw [hv,Storage.dense_logical api neg _ _ i.val i.isLt]
  have hp := BackwardRuntime.affine_preserves_dx c (platform.scalar r.affine.code.object.primitives)
    r.affine.code.plan api r.preparation.data layout r.affine.schedule r.affine.s2.memory i
  change api.read (r.affine.final.memory r.preparation.data.dx.owner) (r.preparation.data.dx.address i.val)=_ at hp
  rw [hp]
  have ht := r.affine.preTransition.values r.preparation.data.dx.tensor (dx_outputs_live r _ (by simp [outputs]))
    i.val (by rw [←r.preparation.data.dx.countEq]; exact i.isLt)
  rw [Storage.dense_logical api neg _ _ i.val i.isLt,Storage.dense_logical api neg _ _ i.val i.isLt] at ht
  rw [ht,ho]
  exact BackwardRuntime.route_readback r.dx (0 : Fin 1) i

theorem body_native_dw (r : BodyRoute lib service c saved dy initial) (i : Fin r.preparation.data.dw.count) :
    Storage.logical api neg r.final.memory r.output.2.1 i.val=
      BackwardRuntime.sourceValue c (platform.scalar r.affine.code.object.primitives) r.affine.code.plan api
        r.preparation.data .affine r.affine.s2.memory 0 i.val := by
  let layout := route_layout r.preparation saved
  have ho : r.preparation.data.dw.owner=c.x.owner := layout.owners _ (by simp [outputs])
  have hv := GradientReturn.dw_values r.returns (affine_outputs_live r _ (by simp [outputs])) i.val i.isLt
  change Storage.logical api neg r.final.memory r.output.2.1 i.val=_ at hv
  rw [hv,Storage.dense_logical api neg _ _ i.val i.isLt,ho]
  let k : Fin (affineCount r.preparation.data) := ⟨0,by simp [affineCount]⟩
  exact BackwardRuntime.route_readback r.affine k i

theorem body_native_db (r : BodyRoute lib service c saved dy initial) (db : Dense) (out : Storage.Tensor)
    (hdb : r.preparation.data.db=some db) (hout : r.output.2.2.1=some out) (i : Fin db.count) :
    Storage.logical api neg r.final.memory out i.val=
      BackwardRuntime.sourceValue c (platform.scalar r.affine.code.object.primitives) r.affine.code.plan api
        r.preparation.data .affine r.affine.s2.memory 1 i.val := by
  let layout := route_layout r.preparation saved
  have hm : db∈outputs r.preparation.data := by simp [outputs,hdb]
  have ho : db.owner=c.x.owner := layout.owners _ hm
  have hv := GradientReturn.db_values r.returns db out hdb hout (affine_outputs_live r db hm) i.val i.isLt
  change Storage.logical api neg r.final.memory out i.val=_ at hv
  rw [hv,Storage.dense_logical api neg _ _ i.val i.isLt,ho]
  let k : Fin (affineCount r.preparation.data) := ⟨1,by simp [affineCount,hdb]⟩
  have hk : BackwardRuntime.Kind.affine.output r.preparation.data k=db := by simp [BackwardRuntime.Kind.output,affineOutput,k,hdb]
  have hi : i.val<(BackwardRuntime.Kind.affine.output r.preparation.data k).count := by rw [hk]; exact i.isLt
  have read := BackwardRuntime.route_readback r.affine k ⟨i.val,hi⟩
  simp only at read
  rw [hk] at read
  exact read

#axiomsClean preparation_present
#axiomsClean body_native_dx
#axiomsClean body_native_dw
#axiomsClean body_native_db
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.BackwardCall
