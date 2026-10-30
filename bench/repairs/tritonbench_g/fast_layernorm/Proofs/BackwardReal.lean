import bench.repairs.tritonbench_g.fast_layernorm.Proofs.BackwardValues

namespace VeriTile.Bench.Repairs.FastLayernorm.BackwardCall
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host Execution BackwardPreparation BackwardLayout
noncomputable section
open scoped Classical
variable {Store Metadata : Type} {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32}
  {inputs : List Storage.Tensor} {platform : Compiler.Platform}
  {lib : APIHost api neg inputs Metadata} {service : BackwardRuntime.Service platform api neg inputs Metadata}
  {c : SavedContext} {saved : SavedLayout c} {dy : Storage.Tensor} {initial : State Store Metadata}

def BodyRoute.dxInput (r : BodyRoute lib service c saved dy initial) (m : Storage.World (Nat → ℝ)) :=
  r.dx.preTransition.shadow (r.preparation.transition.shadow m)

def BodyRoute.affineInput (r : BodyRoute lib service c saved dy initial) (m : Storage.World (Nat → ℝ)) :=
  r.affine.preTransition.shadow (r.dx.realFinal RealProof.ops (r.preparation.transition.shadow m))

theorem body_real_dx (r : BodyRoute lib service c saved dy initial) (m : Storage.World (Nat → ℝ))
    (i : Fin r.preparation.data.dx.count) :
    Storage.realLogical (r.realFinal m) r.output.1 i.val=
      RealProof.gradientSpec c.columns
        (rows c.x c.columns (r.dxInput m) Memory.realAPI (i.val/c.columns))
        (rows r.preparation.data.dy c.columns (r.dxInput m) Memory.realAPI (i.val/c.columns))
        (vector c.w (r.dxInput m) Memory.realAPI)
        ((Execution.saved Memory.realAPI c.r c.mu (r.dxInput m) (i.val/c.columns)).mean)
        ((Execution.saved Memory.realAPI c.r c.mu (r.dxInput m) (i.val/c.columns)).inverse) (i.val%c.columns) := by
  let layout := route_layout r.preparation saved
  have ho : r.preparation.data.dx.owner=c.x.owner := layout.owners _ (by simp [outputs])
  let mid := r.dx.realFinal RealProof.ops (r.preparation.transition.shadow m)
  have hv := GradientReturn.dx_real r.returns (r.affine.realFinal RealProof.ops mid) i.val i.isLt
  change Storage.realLogical (r.realFinal m) r.output.1 i.val=_ at hv
  rw [hv,Storage.dense_real _ _ _ i.isLt]
  have hp := BackwardRuntime.affine_preserves_dx c RealProof.ops r.affine.code.plan Memory.realAPI
    r.preparation.data layout r.affine.schedule (r.affineInput m) i
  change r.affine.realFinal RealProof.ops mid r.preparation.data.dx.owner (r.preparation.data.dx.address i.val)=_ at hp
  rw [hp]
  have ht := r.affine.preTransition.realValues mid r.preparation.data.dx.tensor (dx_outputs_live r _ (by simp [outputs]))
    i.val (by rw [←r.preparation.data.dx.countEq]; exact i.isLt)
  rw [Storage.dense_real _ _ _ i.isLt,Storage.dense_real _ _ _ i.isLt] at ht
  change r.affineInput m r.preparation.data.dx.owner (r.preparation.data.dx.address i.val)=_
  change r.affineInput m r.preparation.data.dx.owner (r.preparation.data.dx.address i.val)=
    r.dx.realFinal RealProof.ops (r.preparation.transition.shadow m) r.preparation.data.dx.owner
      (r.preparation.data.dx.address i.val) at ht
  rw [ht,ho]
  have read := BackwardRuntime.route_real_readback r.dx (r.preparation.transition.shadow m) (0 : Fin 1) i
  change r.dx.realFinal RealProof.ops (r.preparation.transition.shadow m) c.x.owner
    (r.preparation.data.dx.address i.val)=_ at read
  exact read.trans (RealProof.backward_real _ _ _ _ _ _)

theorem body_real_dw (r : BodyRoute lib service c saved dy initial) (m : Storage.World (Nat → ℝ))
    (i : Fin r.preparation.data.dw.count) :
    Storage.realLogical (r.realFinal m) r.output.2.1 i.val=
      RealProof.weightGradientSpec c.columns c.rows (rows c.x c.columns (r.affineInput m) Memory.realAPI)
        (rows r.preparation.data.dy c.columns (r.affineInput m) Memory.realAPI)
        (Execution.saved Memory.realAPI c.r c.mu (r.affineInput m)) i.val := by
  let layout := route_layout r.preparation saved
  have ho : r.preparation.data.dw.owner=c.x.owner := layout.owners _ (by simp [outputs])
  let mid := r.dx.realFinal RealProof.ops (r.preparation.transition.shadow m)
  have hv := GradientReturn.dw_real r.returns (affine_outputs_live r _ (by simp [outputs]))
    (r.affine.realFinal RealProof.ops mid) i.val i.isLt
  change Storage.realLogical (r.realFinal m) r.output.2.1 i.val=_ at hv
  rw [hv,Storage.dense_real _ _ _ i.isLt,ho]
  let k : Fin (affineCount r.preparation.data) := ⟨0,by simp [affineCount]⟩
  have read := BackwardRuntime.route_real_readback r.affine mid k i
  change r.affine.realFinal RealProof.ops mid c.x.owner (r.preparation.data.dw.address i.val)=_ at read
  rw [read]
  change (affineReference RealProof.ops r.affine.code.plan c.rows _ _ _ i.val).1=_
  rw [RealProof.affine_real]
  rfl

theorem body_real_db (r : BodyRoute lib service c saved dy initial) (db : Dense) (out : Storage.Tensor)
    (hdb : r.preparation.data.db=some db) (hout : r.output.2.2.1=some out)
    (m : Storage.World (Nat → ℝ)) (i : Fin db.count) :
    Storage.realLogical (r.realFinal m) out i.val=
      RealProof.biasGradientSpec c.columns c.rows
        (rows r.preparation.data.dy c.columns (r.affineInput m) Memory.realAPI) i.val := by
  let layout := route_layout r.preparation saved
  have hm : db∈outputs r.preparation.data := by simp [outputs,hdb]
  have ho : db.owner=c.x.owner := layout.owners _ hm
  let mid := r.dx.realFinal RealProof.ops (r.preparation.transition.shadow m)
  have hv := GradientReturn.db_real r.returns db out hdb hout (affine_outputs_live r db hm)
    (r.affine.realFinal RealProof.ops mid) i.val i.isLt
  change Storage.realLogical (r.realFinal m) out i.val=_ at hv
  rw [hv,Storage.dense_real _ _ _ i.isLt,ho]
  let k : Fin (affineCount r.preparation.data) := ⟨1,by simp [affineCount,hdb]⟩
  have hk : BackwardRuntime.Kind.affine.output r.preparation.data k=db := by simp [BackwardRuntime.Kind.output,affineOutput,k,hdb]
  have hi : i.val<(BackwardRuntime.Kind.affine.output r.preparation.data k).count := by rw [hk]; exact i.isLt
  have read := BackwardRuntime.route_real_readback r.affine mid k ⟨i.val,hi⟩
  simp only at read
  rw [hk] at read
  rw [read]
  change (affineReference RealProof.ops r.affine.code.plan c.rows _ _ _ i.val).2=_
  rw [RealProof.affine_real]
  rfl

#axiomsClean body_real_dx
#axiomsClean body_real_dw
#axiomsClean body_real_db
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.BackwardCall
