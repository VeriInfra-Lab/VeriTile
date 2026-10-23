import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.ConcurrentSource
import Mathlib.Tactic.IntervalCases

namespace VeriTile.Bench.Repairs.SpinningLock.ConcurrentSource
open VeriTile.Bench.TritonBenchG.SpinningLock VeriTile.Triton SourceReader

def phase : Nat → Nat → Nat → Wide.Phase
  | 0, _, _ => .branch
  | 1, next, _ => .scan next
  | 2, next, _ => .poll next
  | 3, next, lane => .gather next lane
  | 4, _, lane => .publish lane
  | 5, _, _ => .release
  | 6, _, lane => .output lane
  | 7, _, _ => .done
  | _, _, _ => .fault

def projectState (s : Concurrent.ControlState V) : Wide.State V :=
  ⟨values s.values,fun p => ⟨s.locals p 0,phase (s.pc p) (s.locals p 1) (s.locals p 2)⟩⟩

@[simp] theorem projectState_values (s : Concurrent.ControlState V) :
    (projectState s).values = values s.values := rfl

@[simp] theorem projectState_pc (s : Concurrent.ControlState V) (p : Nat) :
    (projectState s).pc p = ⟨s.locals p 0,phase (s.pc p) (s.locals p 1) (s.locals p 2)⟩ := rfl

def Valid (s : Concurrent.ControlState V) : Prop := ∀ p, s.pc p < 9

theorem project_update (s : Concurrent.ControlState V) (v : Concurrent.State V)
    (p label : Nat) (locals : Nat → Nat) :
    projectState ⟨v,Concurrent.update s.pc p label,Concurrent.update s.locals p locals⟩ =
      ⟨values v,put (projectState s).pc p ⟨locals 0,phase label (locals 1) (locals 2)⟩⟩ := by
  unfold projectState
  congr 1
  funext q
  by_cases h : q = p
  · subst q; simp [projectState,Concurrent.update,VeriTile.Bench.TritonBenchG.SpinningLock.put]
  · simp [projectState,Concurrent.update,VeriTile.Bench.TritonBenchG.SpinningLock.put,h]

set_option maxRecDepth 4096 in
set_option maxHeartbeats 1600000 in
theorem step_projection (b : Backend V) (plan : Plan) (h : HostArgs) (stage : Phased.Stage)
    (p : Nat) (s : Concurrent.ControlState V) (hp : p < h.numSms) (valid : s.pc p < 9) :
    Option.map (fun result => (result.1,projectState result.2))
      ((controller plan h stage).step (backend b) p s) =
    some (event p (decodePlan plan h stage p (projectState s)).1,
      (stepPlan b plan h stage p (projectState s)).2) := by
  generalize hl : s.pc p = label at valid
  interval_cases label
  all_goals
    simp only [Concurrent.Controller.step,controller,hl,List.getElem?_cons_zero,
      List.getElem?_cons_succ,bind,Option.bind,Option.map_some]
    simp only [Concurrent.Code.decode,Concurrent.EventCode.instantiate,
      Concurrent.ValueCode.instantiate,go,Concurrent.Next.eval,Concurrent.Test.eval,
      lowerB_eval,lowerI_eval,lowerP_eval,Concurrent.NumExpr.eval,
      List.foldl_cons,List.foldl_nil]
    simp only [stepPlan,decodePlan,projectState_pc,projectState_values,phase,
      show ¬p ≥ h.numSms by omega,if_false,hl]
    try split_ifs
    all_goals try omega
    all_goals try simp_all only [Concurrent.Next.eval,Concurrent.Test.eval,
      lowerB_eval,lowerI_eval,lowerP_eval,Concurrent.NumExpr.eval,
      Concurrent.Code.decode,Concurrent.EventCode.instantiate,Concurrent.ValueCode.instantiate,
      go,List.foldl_cons,List.foldl_nil]
    all_goals try split_ifs
    all_goals try omega
    all_goals simp_all [project_update,phase,values,backend,cas,word,event,
      Concurrent.Next.eval,Concurrent.Test.eval,lowerB_eval,lowerI_eval,lowerP_eval,
      Concurrent.NumExpr.eval,Concurrent.execute,Concurrent.Expr.eval,
      Concurrent.update,executeSource,execute,VeriTile.Bench.TritonBenchG.SpinningLock.put]
    all_goals first | rfl | omega

end VeriTile.Bench.Repairs.SpinningLock.ConcurrentSource
