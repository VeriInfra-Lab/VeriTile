import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.BackwardCall
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.Numerics

namespace VeriTile.Bench.Repairs.LayerNormTriton.BackwardHost
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm hiding Ops Plan
open VeriTile.Bench.Repairs.FastLayernorm.Host
open Storage
noncomputable section
variable {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32} {inputs : List Storage.Tensor}
  {Metadata : Type} {platform : Compiler.Platform} {service : Service platform api neg inputs Metadata}
  {c : SavedContext} {base : Base} {scratch : Scratch} {layout : BaseLayout base c inputs}
  {sl : ScratchLayout base scratch c inputs} {initial : State Store Metadata}

theorem launches_scratch_native (r : LaunchesRoute service c base scratch layout sl initial)
    (live : initial.live scratch.dw.tensor ∧ initial.live scratch.db.tensor)
    (group col : Nat) (hg : group<min (Geometry.groups c.columns) c.rows) (hc : col<c.columns) :
    Execution.E.rows scratch.dw c.columns r.last.s2.memory api group col=
      (groupValue (platform.scalar r.first.code.object.primitives) r.first.code.plan c.rows (Geometry.groups c.columns) group
        (Execution.E.rows c.x c.columns r.first.s2.memory api) (Execution.E.rows base.dy c.columns r.first.s2.memory api)
        (Execution.E.saved api c.r c.mu r.first.s2.memory) col).1 ∧
    Execution.E.rows scratch.db c.columns r.last.s2.memory api group col=
      (groupValue (platform.scalar r.first.code.object.primitives) r.first.code.plan c.rows (Geometry.groups c.columns) group
        (Execution.E.rows c.x c.columns r.first.s2.memory api) (Execution.E.rows base.dy c.columns r.first.s2.memory api)
        (Execution.E.saved api c.r c.mu r.first.s2.memory) col).2 := by
  have hi : group*c.columns+col<min (Geometry.groups c.columns) c.rows*c.columns := by
    have h := Nat.mul_le_mul_right c.columns (show group+1≤min (Geometry.groups c.columns) c.rows by omega)
    nlinarith
  have hwc : group*c.columns+col<scratch.dw.count := by simpa only [sl.dwCount] using hi
  have hbc : group*c.columns+col<scratch.db.count := by simpa only [sl.dbCount] using hi
  have hw := r.last.preTransition.values scratch.dw.tensor (r.first.preTransition.live _ live.1) _
    (by rw [←scratch.dw.countEq]; exact hwc)
  have hb := r.last.preTransition.values scratch.db.tensor (r.first.preTransition.live _ live.2) _
    (by rw [←scratch.db.countEq]; exact hbc)
  rw [dense_logical api neg _ _ _ hwc,dense_logical api neg _ _ _ hwc] at hw
  rw [dense_logical api neg _ _ _ hbc,dense_logical api neg _ _ _ hbc] at hb
  have hr := Execution.group_readback (platform.scalar r.first.code.object.primitives) r.first.code.plan api
    (groupBuffers base scratch layout sl) c.x c.w base.dy c.r c.mu r.first.s2.memory r.first.schedule
    (Runtime.route_coverage r.first) group col hg hc
  exact ⟨hw.trans hr.1,hb.trans hr.2⟩

def LaunchesRoute.targetValue (r : LaunchesRoute service c base scratch layout sl initial)
    (target : Numerics.Target) (k : Fin 3) (i : Nat) : UInt32 :=
  match k with
  | ⟨0,_⟩ => backwardValue (target.scalar r.first.code.object.primitives) r.first.code.plan
      (Execution.E.saved api c.r c.mu r.first.s2.memory (i/c.columns))
      (Execution.E.rows c.x c.columns r.first.s2.memory api (i/c.columns))
      (Execution.E.rows base.dy c.columns r.first.s2.memory api (i/c.columns))
      (Execution.E.vector c.w r.first.s2.memory api) (i%c.columns)
  | ⟨1,_⟩ => finalValue (target.scalar r.last.code.object.primitives) r.last.code.finalPlan
      (min (Geometry.groups c.columns) c.rows)
      (fun group => (groupValue (target.scalar r.first.code.object.primitives) r.first.code.plan c.rows (Geometry.groups c.columns) group
        (Execution.E.rows c.x c.columns r.first.s2.memory api) (Execution.E.rows base.dy c.columns r.first.s2.memory api)
        (Execution.E.saved api c.r c.mu r.first.s2.memory) i).1) i
  | ⟨2,_⟩ => finalValue (target.scalar r.last.code.object.primitives) r.last.code.finalPlan
      (min (Geometry.groups c.columns) c.rows)
      (fun group => (groupValue (target.scalar r.first.code.object.primitives) r.first.code.plan c.rows (Geometry.groups c.columns) group
        (Execution.E.rows c.x c.columns r.first.s2.memory api) (Execution.E.rows base.dy c.columns r.first.s2.memory api)
        (Execution.E.saved api c.r c.mu r.first.s2.memory) i).2) i

theorem launches_target_dw (r : LaunchesRoute service c base scratch layout sl initial) (target : Numerics.Target)
    (first : Numerics.Conformance platform target r.first.code neg)
    (last : Numerics.Conformance platform target r.last.code neg)
    (live : initial.live scratch.dw.tensor ∧ initial.live scratch.db.tensor) (i : Nat) (hi : i<c.columns) :
    r.dwValue i=r.targetValue target 1 i := by
  unfold LaunchesRoute.dwValue LaunchesRoute.targetValue
  rw [←first.primitives.operations,←last.primitives.operations]
  apply Execution.final_congr
  exact fun group hg => (launches_scratch_native r live group i hg hi).1

theorem launches_target_db (r : LaunchesRoute service c base scratch layout sl initial) (target : Numerics.Target)
    (first : Numerics.Conformance platform target r.first.code neg)
    (last : Numerics.Conformance platform target r.last.code neg)
    (live : initial.live scratch.dw.tensor ∧ initial.live scratch.db.tensor) (i : Nat) (hi : i<c.columns) :
    r.dbValue i=r.targetValue target 2 i := by
  unfold LaunchesRoute.dbValue LaunchesRoute.targetValue
  rw [←first.primitives.operations,←last.primitives.operations]
  apply Execution.final_congr
  exact fun group hg => (launches_scratch_native r live group i hg hi).2

variable {data : All} {ready : Ready data c inputs}

def LaunchBranch.conformance (r : LaunchBranch service c data ready initial) (target : Numerics.Target) : Prop :=
  match r with
  | .skipped _ => True
  | .ran _ _ r => Numerics.Conformance platform target r.first.code neg ∧ Numerics.Conformance platform target r.last.code neg

def LaunchBranch.targetValue (r : LaunchBranch service c data ready initial) (target : Numerics.Target) (k : Fin 3) (i : Nat) : UInt32 :=
  match r with | .skipped _ => 0 | .ran _ _ r => r.targetValue target k i

variable {lib : APIHost api neg inputs Metadata} {extra : APIExtra api neg inputs Metadata} {identity : Nat}
  {b : Dense} {saved : Saved c b} {dy : Storage.Tensor}

theorem returned_fp (r : Call lib extra service identity c b saved dy initial) (target : Numerics.Target)
    (laws : r.body.launch.conformance target) (k : Fin 3) (i : Fin (outputs r.body.preparation.data.base k).count) :
    Storage.logical api neg r.final.memory (r.body.returns.outputs k) i.val=r.body.launch.targetValue target k i.val := by
  rw [returned_native]
  cases h : r.body.launch with
  | skipped hn => rfl
  | ran sc hs lr =>
    have lc : Numerics.Conformance platform target lr.first.code neg ∧ Numerics.Conformance platform target lr.last.code neg := by
      simpa only [LaunchBranch.conformance,h] using laws
    have live := preparation_scratch_present r.body.preparation sc hs
    fin_cases k
    · change lr.dxValue i.val=lr.targetValue target 0 i.val
      unfold LaunchesRoute.dxValue LaunchesRoute.targetValue
      rw [lc.1.primitives.operations]
    · exact launches_target_dw lr target lc.1 lc.2 live i.val (by simpa only [outputs,r.body.ready.base.dwCount] using i.isLt)
    · exact launches_target_db lr target lc.1 lc.2 live i.val (by simpa only [outputs,r.body.ready.base.dbCount] using i.isLt)

#axiomsClean launches_scratch_native
#axiomsClean returned_fp
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.BackwardHost
