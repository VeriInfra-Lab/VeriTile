import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.BackwardResult
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.GradientValues

namespace VeriTile.Bench.Repairs.LayerNormTriton.BackwardHost
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm hiding Ops Plan
open VeriTile.Bench.Repairs.FastLayernorm.Host
open Storage
noncomputable section
variable {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32} {inputs : List Storage.Tensor}
  {Metadata : Type} {lib : APIHost api neg inputs Metadata} {c : SavedContext} {p : Base} {initial : State Store Metadata}

def Base.returnData (p : Base) : BackwardPreparation.Data := ⟨p.dy,p.dx,p.dw,some p.db⟩

def ReturnRoute.reuse (r : ReturnRoute lib c p initial) : GradientReturn.Route lib c p.returnData initial :=
  ⟨r.dx,r.dxOK,r.wr,.some r.br⟩

def ReturnRoute.transition (r : ReturnRoute lib c p initial) : Transition api neg inputs initial r.final := r.reuse.transition

theorem return_dx_native (r : ReturnRoute lib c p initial) (i : Fin p.dx.count) :
    Storage.logical api neg r.final.memory r.dx.tensor i.val=Storage.logical api neg initial.memory p.dx.tensor i.val :=
  GradientReturn.dx_values r.reuse i.val i.isLt

theorem return_dw_native (r : ReturnRoute lib c p initial) (present : initial.live p.dw.tensor) (i : Fin p.dw.count) :
    Storage.logical api neg r.final.memory r.wr.output i.val=Storage.logical api neg initial.memory p.dw.tensor i.val :=
  GradientReturn.dw_values r.reuse present i.val i.isLt

theorem return_db_native (r : ReturnRoute lib c p initial) (present : initial.live p.db.tensor) (i : Fin p.db.count) :
    Storage.logical api neg r.final.memory r.br.output i.val=Storage.logical api neg initial.memory p.db.tensor i.val :=
  GradientReturn.db_values r.reuse p.db r.br.output rfl rfl present i.val i.isLt

theorem return_dx_real (r : ReturnRoute lib c p initial) (m : Storage.World (Nat → ℝ)) (i : Fin p.dx.count) :
    Storage.realLogical (r.transition.shadow m) r.dx.tensor i.val=Storage.realLogical m p.dx.tensor i.val :=
  GradientReturn.dx_real r.reuse m i.val i.isLt

theorem return_dw_real (r : ReturnRoute lib c p initial) (present : initial.live p.dw.tensor)
    (m : Storage.World (Nat → ℝ)) (i : Fin p.dw.count) :
    Storage.realLogical (r.transition.shadow m) r.wr.output i.val=Storage.realLogical m p.dw.tensor i.val :=
  GradientReturn.dw_real r.reuse present m i.val i.isLt

theorem return_db_real (r : ReturnRoute lib c p initial) (present : initial.live p.db.tensor)
    (m : Storage.World (Nat → ℝ)) (i : Fin p.db.count) :
    Storage.realLogical (r.transition.shadow m) r.br.output i.val=Storage.realLogical m p.db.tensor i.val :=
  GradientReturn.db_real r.reuse p.db r.br.output rfl rfl present m i.val i.isLt

#axiomsClean return_dx_native
#axiomsClean return_dw_real
#axiomsClean return_db_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.BackwardHost
