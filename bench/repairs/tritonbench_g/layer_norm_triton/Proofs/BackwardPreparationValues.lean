import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.BackwardNative

namespace VeriTile.Bench.Repairs.LayerNormTriton.BackwardHost
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm hiding Ops Plan
open VeriTile.Bench.Repairs.FastLayernorm.Host
open Storage Preparation
noncomputable section
open scoped Classical
variable {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32} {inputs : List Storage.Tensor}
  {Metadata : Type} {lib : APIHost api neg inputs Metadata} {extra : APIExtra api neg inputs Metadata}
  {c : SavedContext} {b : Dense} {dy : Storage.Tensor} {initial : State Store Metadata}

theorem preparation_present (r : PreparationRoute lib extra c b dy initial) :
    r.final.live r.data.base.dy.tensor ∧ ∀ k,r.final.live (outputs r.data.base k).tensor :=
  ⟨r.scratch.transition.live _ (base_present r.base).1,
   fun k => r.scratch.transition.live _ ((base_present r.base).2 k)⟩

theorem preparation_inactive (r : PreparationRoute lib extra c b dy initial) (hn : r.data.scratch=none) :
    c.rows=0 ∨ c.columns=0 := by
  cases h : r.scratch with
  | skipped inactive => simpa only [not_and_or,ne_eq,not_not] using inactive
  | allocated active sr => simp [PreparationRoute.data,h,ScratchBranch.output] at hn

theorem preparation_scratch_present (r : PreparationRoute lib extra c b dy initial)
    (sc : Scratch) (hs : r.data.scratch=some sc) : r.final.live sc.dw.tensor ∧ r.final.live sc.db.tensor := by
  cases h : r.scratch with
  | skipped inactive => simp [PreparationRoute.data,h,ScratchBranch.output] at hs
  | allocated active sr =>
    have he : sc=sr.data := by simpa [PreparationRoute.data,h,ScratchBranch.output] using hs.symm
    subst sc
    simp only [PreparationRoute.final,h,ScratchBranch.final]
    constructor
    · exact (extra.empty c.x.owner (scratchShape c)).transition sr.middle |>.live _
        (extra.emptyResult c.x.owner (scratchShape c) r.base.final sr.dw sr.dwOK).1.present
    · exact (extra.emptyResult c.x.owner (scratchShape c) sr.middle sr.db sr.dbOK).1.present

theorem base_zero_native (r : BaseRoute lib extra c b dy initial) :
    (∀ i : Fin r.dw.count,Storage.logical api neg r.final.memory r.dw.tensor i.val=0) ∧
    (∀ i : Fin r.db.count,Storage.logical api neg r.final.memory r.db.tensor i.val=0) := by
  constructor
  · intro i
    have h := (extra.zerosLike b).transition r.sW |>.values r.dw.tensor
      (extra.zeroResult c.w r.sX r.dw r.dwOK).1.present i.val (by rw [←r.dw.countEq]; exact i.isLt)
    exact h.trans (extra.zeroNative c.w r.sX r.dw r.dwOK i)
  · exact extra.zeroNative b r.sW r.db r.dbOK

theorem preparation_zero_native (r : PreparationRoute lib extra c b dy initial) :
    (∀ i : Fin r.data.base.dw.count,Storage.logical api neg r.final.memory r.data.base.dw.tensor i.val=0) ∧
    (∀ i : Fin r.data.base.db.count,Storage.logical api neg r.final.memory r.data.base.db.tensor i.val=0) := by
  constructor
  · intro i
    exact (r.scratch.transition.values r.data.base.dw.tensor ((base_present r.base).2 1) i.val
      (by rw [←r.data.base.dw.countEq]; exact i.isLt)).trans ((base_zero_native r.base).1 i)
  · intro i
    exact (r.scratch.transition.values r.data.base.db.tensor ((base_present r.base).2 2) i.val
      (by rw [←r.data.base.db.countEq]; exact i.isLt)).trans ((base_zero_native r.base).2 i)

theorem preparation_zero_real (r : PreparationRoute lib extra c b dy initial) (m : Storage.World (Nat → ℝ)) :
    (∀ i : Fin r.data.base.dw.count,Storage.realLogical (r.transition.shadow m) r.data.base.dw.tensor i.val=0) ∧
    (∀ i : Fin r.data.base.db.count,Storage.realLogical (r.transition.shadow m) r.data.base.db.tensor i.val=0) := by
  constructor
  · intro i
    exact (r.scratch.transition.realValues (r.base.transition.shadow m) r.data.base.dw.tensor ((base_present r.base).2 1) i.val
      (by rw [←r.data.base.dw.countEq]; exact i.isLt)).trans ((base_zero_real r.base m).1 _ i.isLt)
  · intro i
    exact (r.scratch.transition.realValues (r.base.transition.shadow m) r.data.base.db.tensor ((base_present r.base).2 2) i.val
      (by rw [←r.data.base.db.countEq]; exact i.isLt)).trans ((base_zero_real r.base m).2 _ i.isLt)

theorem preparation_dy_real (r : PreparationRoute lib extra c b dy initial) (m : Storage.World (Nat → ℝ))
    (i : Fin r.data.base.dy.count) :
    Storage.realLogical (r.transition.shadow m) r.data.base.dy.tensor i.val=Storage.realLogical m dy i.val := by
  have hc := (materialize_metadata r.base.dyr).2.1
  have hi : i.val<dy.view.shape.prod := by simpa only [←hc] using i.isLt
  let tr := r.base.tail.then r.scratch.transition
  simpa only [tr,PreparationRoute.transition,BaseRoute.transition,Transition.then,Function.comp_apply,
    PreparationRoute.data,BaseRoute.data] using (tr.realValues (r.base.dyr.transition.shadow m) _ (materialize_metadata r.base.dyr).2.2.2 i.val
    (by rw [←r.base.dyr.output.countEq]; exact i.isLt)).trans (materialize_real r.base.dyr m _ hi)

#axiomsClean preparation_present
#axiomsClean preparation_inactive
#axiomsClean preparation_zero_native
#axiomsClean preparation_zero_real
#axiomsClean preparation_dy_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.BackwardHost
