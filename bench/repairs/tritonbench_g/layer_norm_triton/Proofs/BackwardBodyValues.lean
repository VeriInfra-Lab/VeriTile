import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.BackwardReal
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.RealCongruence

namespace VeriTile.Bench.Repairs.LayerNormTriton.BackwardHost
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm hiding Ops Plan
open VeriTile.Bench.Repairs.FastLayernorm.Host
open Storage
noncomputable section
variable {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32} {inputs : List Storage.Tensor}
  {Metadata : Type} {platform : Compiler.Platform} {service : Service platform api neg inputs Metadata}
  {c : SavedContext} {data : All} {ready : Ready data c inputs} {initial : State Store Metadata}

def LaunchBranch.preReal (r : LaunchBranch service c data ready initial) (m : Storage.World (Nat → ℝ)) :=
  match r with | .skipped _ => m | .ran _ _ r => r.first.preTransition.shadow m

def LaunchBranch.realFinal (r : LaunchBranch service c data ready initial) (m : Storage.World (Nat → ℝ)) :=
  match r with | .skipped _ => m | .ran _ _ r => r.realFinal m

def LaunchBranch.nativeValue (r : LaunchBranch service c data ready initial) (k : Fin 3) (i : Nat) : UInt32 :=
  match r with
  | .skipped _ => 0
  | .ran _ _ r => match k with | ⟨0,_⟩ => r.dxValue i | ⟨1,_⟩ => r.dwValue i | ⟨2,_⟩ => r.dbValue i

theorem launch_live (r : LaunchBranch service c data ready initial) (t : Storage.Tensor) (present : initial.live t) :
    r.final.live t := by
  cases r with | skipped _ => exact present | ran _ _ r => exact launches_live r t present

theorem launch_pre_real (r : LaunchBranch service c data ready initial) (m : Storage.World (Nat → ℝ))
    (t : Storage.Tensor) (present : initial.live t) (i : Nat) (hi : i<t.view.shape.prod) :
    Storage.realLogical (r.preReal m) t i=Storage.realLogical m t i := by
  cases r with | skipped _ => rfl | ran _ _ r => exact r.first.preTransition.realValues m t present i hi

theorem launch_native (r : LaunchBranch service c data ready initial)
    (inactive : data.scratch=none → c.rows=0 ∨ c.columns=0)
    (present : initial.live data.base.dx.tensor)
    (dw : ∀ i : Fin data.base.dw.count,Storage.logical api neg initial.memory data.base.dw.tensor i.val=0)
    (db : ∀ i : Fin data.base.db.count,Storage.logical api neg initial.memory data.base.db.tensor i.val=0)
    (k : Fin 3) (i : Fin (outputs data.base k).count) :
    Storage.logical api neg r.final.memory (outputs data.base k).tensor i.val=r.nativeValue k i.val := by
  cases r with
  | skipped hn =>
    fin_cases k
    · have hz : data.base.dx.count=0 := by rcases inactive hn with h|h <;> simp [ready.base.dxCount,h]
      exact False.elim (by have hi := i.isLt; change i.val<data.base.dx.count at hi; omega)
    · exact dw i
    · exact db i
  | ran sc hs lr =>
    fin_cases k
    · exact launches_dx_native lr present i
    · exact launches_dw_native lr i
    · exact launches_db_native lr i

variable {lib : APIHost api neg inputs Metadata} {extra : APIExtra api neg inputs Metadata}
  {b : Dense} {saved : Saved c b} {dy : Storage.Tensor}

def BodyRoute.realFinal (r : BodyRoute lib extra service c b saved dy initial) (m : Storage.World (Nat → ℝ)) :=
  r.returns.transition.shadow (r.launch.realFinal (r.preparation.transition.shadow m))

def BodyRoute.preReal (r : BodyRoute lib extra service c b saved dy initial) (m : Storage.World (Nat → ℝ)) :=
  r.launch.preReal (r.preparation.transition.shadow m)

theorem body_native (r : BodyRoute lib extra service c b saved dy initial) :
    (∀ i : Fin r.preparation.data.base.dx.count,
      Storage.logical api neg r.final.memory r.returns.dx.tensor i.val=r.launch.nativeValue 0 i.val) ∧
    (∀ i : Fin r.preparation.data.base.dw.count,
      Storage.logical api neg r.final.memory r.returns.wr.output i.val=r.launch.nativeValue 1 i.val) ∧
    (∀ i : Fin r.preparation.data.base.db.count,
      Storage.logical api neg r.final.memory r.returns.br.output i.val=r.launch.nativeValue 2 i.val) := by
  have hp := preparation_present r.preparation
  have hz := preparation_zero_native r.preparation
  have hn := launch_native r.launch (preparation_inactive r.preparation) (hp.2 0) hz.1 hz.2
  exact ⟨fun i => (return_dx_native r.returns i).trans (hn 0 i),
    fun i => (return_dw_native r.returns (launch_live r.launch _ (hp.2 1)) i).trans (hn 1 i),
    fun i => (return_db_native r.returns (launch_live r.launch _ (hp.2 2)) i).trans (hn 2 i)⟩

theorem body_dx_snapshot (r : BodyRoute lib extra service c b saved dy initial) (m : Storage.World (Nat → ℝ))
    (i : Fin r.preparation.data.base.dx.count) :
    Storage.realLogical (r.realFinal m) r.returns.dx.tensor i.val=
      RealProof.Spec.gradient c.columns (Execution.E.rows c.x c.columns (r.preReal m) Memory.realAPI (i.val/c.columns))
        (Execution.E.rows r.preparation.data.base.dy c.columns (r.preReal m) Memory.realAPI (i.val/c.columns))
        (Execution.E.vector c.w (r.preReal m) Memory.realAPI)
        (Execution.E.saved Memory.realAPI c.r c.mu (r.preReal m) (i.val/c.columns)).mean
        (Execution.E.saved Memory.realAPI c.r c.mu (r.preReal m) (i.val/c.columns)).inverse (i.val%c.columns) := by
  rw [BodyRoute.realFinal,return_dx_real]
  cases h : r.launch with
  | skipped hn =>
    have hz : r.preparation.data.base.dx.count=0 := by
      rcases preparation_inactive r.preparation hn with h|h <;> simp [r.ready.base.dxCount,h]
    exact False.elim (by have hi := i.isLt; omega)
  | ran sc hs lr =>
    simp only [BodyRoute.preReal,h,LaunchBranch.preReal,LaunchBranch.realFinal]
    exact launches_dx_real lr ((preparation_present r.preparation).2 0) _ i

theorem body_dw_snapshot (r : BodyRoute lib extra service c b saved dy initial) (m : Storage.World (Nat → ℝ))
    (i : Fin r.preparation.data.base.dw.count) :
    Storage.realLogical (r.realFinal m) r.returns.wr.output i.val=
      RealProof.Spec.weightGradient c.columns c.rows (Execution.E.rows c.x c.columns (r.preReal m) Memory.realAPI)
        (Execution.E.rows r.preparation.data.base.dy c.columns (r.preReal m) Memory.realAPI)
        (Execution.E.saved Memory.realAPI c.r c.mu (r.preReal m)) i.val := by
  rw [BodyRoute.realFinal,return_dw_real r.returns (launch_live r.launch _ ((preparation_present r.preparation).2 1))]
  cases h : r.launch with
  | skipped hn =>
    have hz := (preparation_zero_real r.preparation m).1 i
    change Storage.realLogical (r.preparation.transition.shadow m) _ i.val=_
    rw [hz]
    rcases preparation_inactive r.preparation hn with hm|hn
    · simp [RealProof.Spec.weightGradient,VeriTile.Bench.Repairs.FastLayernorm.RealProof.weightGradientSpec,hm]
    · have hi := i.isLt; have he := r.ready.base.dwCount; omega
  | ran sc hs lr =>
    simp only [BodyRoute.preReal,h,LaunchBranch.preReal,LaunchBranch.realFinal]
    exact launches_dw_real lr (preparation_scratch_present r.preparation sc hs) _ i

theorem body_db_snapshot (r : BodyRoute lib extra service c b saved dy initial) (m : Storage.World (Nat → ℝ))
    (i : Fin r.preparation.data.base.db.count) :
    Storage.realLogical (r.realFinal m) r.returns.br.output i.val=
      RealProof.Spec.biasGradient c.columns c.rows
        (Execution.E.rows r.preparation.data.base.dy c.columns (r.preReal m) Memory.realAPI) i.val := by
  rw [BodyRoute.realFinal,return_db_real r.returns (launch_live r.launch _ ((preparation_present r.preparation).2 2))]
  cases h : r.launch with
  | skipped hn =>
    have hz := (preparation_zero_real r.preparation m).2 i
    change Storage.realLogical (r.preparation.transition.shadow m) _ i.val=_
    rw [hz]
    rcases preparation_inactive r.preparation hn with hm|hn
    · simp [RealProof.Spec.biasGradient,VeriTile.Bench.Repairs.FastLayernorm.RealProof.biasGradientSpec,hm]
    · have hi := i.isLt; have he := r.ready.base.dbCount; omega
  | ran sc hs lr =>
    simp only [BodyRoute.preReal,h,LaunchBranch.preReal,LaunchBranch.realFinal]
    exact launches_db_real lr (preparation_scratch_present r.preparation sc hs) _ i

#axiomsClean body_native
#axiomsClean body_dx_snapshot
#axiomsClean body_dw_snapshot
#axiomsClean body_db_snapshot
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.BackwardHost
