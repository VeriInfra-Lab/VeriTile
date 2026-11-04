import bench.tritonbench_g.rms_rbe_matmul.Proofs.RotarySourceBridge

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.RotaryBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RbeTritonTransform
noncomputable section
variable (r : Request) (X Y : RegionName)

theorem invocation_runs (q : Runtime.Program) (s : State ℝ) :
    (RotaryIO.invocation r X Y q).run s =
      (RbeTritonTransformCheckedLaunch.invocation ideal r X Y q).run s := by
  exact source_executes r X Y _

theorem invocation_safe (q : Runtime.Program) (bounds : RegionName → Nat) (s : State ℝ) :
    (RotaryIO.invocation r X Y q).Safe bounds s ↔
      (RbeTritonTransformCheckedLaunch.invocation ideal r X Y q).Safe bounds s := by
  exact source_safe r X Y bounds _

theorem schedule_runs (qs : List Runtime.Program) (s : State ℝ) :
    runSchedule (qs.map (RotaryIO.invocation r X Y)) s =
      runSchedule (qs.map (RbeTritonTransformCheckedLaunch.invocation ideal r X Y)) s := by
  induction qs generalizing s with
  | nil => rfl
  | cons q qs ih =>
    simp only [List.map_cons,runSchedule,invocation_runs]
    cases (RbeTritonTransformCheckedLaunch.invocation ideal r X Y q).run s <;> simp [ih]

theorem schedule_safe (qs : List Runtime.Program) (bounds : RegionName → Nat) (s : State ℝ) :
    ScheduleSafe bounds (qs.map (RotaryIO.invocation r X Y)) s ↔
      ScheduleSafe bounds (qs.map (RbeTritonTransformCheckedLaunch.invocation ideal r X Y)) s := by
  induction qs generalizing s with
  | nil => exact Iff.rfl
  | cons q qs ih =>
    simp only [List.map_cons,ScheduleSafe,invocation_runs,invocation_safe,ih]

theorem implements (qs : List Runtime.Program)
    (f : (RotaryIO.io r X Y qs).Inputs → (RotaryIO.io r X Y qs).Outputs)
    (correct : (RbeTritonTransformCheckedLaunch.rotationIO ideal r X Y qs id id).Implements f) :
    (RotaryIO.io r X Y qs).Implements f := by
  intro vals hd bounds hb initial hin
  obtain ⟨safe,final,executed,output,frame⟩ := correct vals hd bounds hb initial hin
  refine ⟨(schedule_safe r X Y qs bounds initial).mpr safe,final,?_,output,frame⟩
  exact (schedule_runs r X Y qs initial).trans executed

#axiomsClean implements
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.RotaryBridge
