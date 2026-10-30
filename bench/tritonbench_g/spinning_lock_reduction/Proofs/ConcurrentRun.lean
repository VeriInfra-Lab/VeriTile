import bench.tritonbench_g.spinning_lock_reduction.Proofs.ConcurrentControl

namespace VeriTile.Bench.Repairs.SpinningLock.ConcurrentSource
open VeriTile.Bench.TritonBenchG.SpinningLock VeriTile.Triton SourceReader

private def nextValid : Concurrent.Next → Prop
  | .goto label _ => label < 9
  | .branch _ yes no => nextValid yes ∧ nextValid no

private theorem nextValid_eval (s : Concurrent.State V) (p : Nat) (locals : Nat → Nat)
    (next : Concurrent.Next) (valid : nextValid next) : (next.eval s p locals).1 < 9 := by
  induction next with
  | goto label assignments => exact valid
  | branch test yes no iy ino =>
      simp only [Concurrent.Next.eval]
      split <;> simp_all [nextValid]

private def codeValid : Concurrent.Code → Prop
  | .emit _ next => nextValid next
  | .branch _ yes no => codeValid yes ∧ codeValid no

private theorem codeValid_decode (s : Concurrent.State V) (p : Nat) (locals : Nat → Nat)
    (code : Concurrent.Code) (valid : codeValid code) : nextValid (code.decode s p locals).2 := by
  induction code with
  | emit e next => exact valid
  | branch test yes no iy ino =>
      simp only [Concurrent.Code.decode]
      split <;> simp_all [codeValid]

private theorem controller_codeValid (plan : Plan) (h : HostArgs) (stage : Phased.Stage)
    (code : Concurrent.Code) (member : code ∈ (controller plan h stage).code) : codeValid code := by
  simp only [controller,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp [codeValid,nextValid,go]

theorem step_valid (b : Backend V) (plan : Plan) (h : HostArgs) (stage : Phased.Stage)
    (p : Nat) (s : Concurrent.ControlState V) (valid : Valid s)
    (e : Concurrent.Event) (final : Concurrent.ControlState V)
    (run : (controller plan h stage).step (backend b) p s = some (e,final)) : Valid final := by
  cases hc : (controller plan h stage).code[s.pc p]? with
  | none => simp [Concurrent.Controller.step,hc] at run
  | some code =>
      have cv := controller_codeValid plan h stage code (List.mem_of_getElem? hc)
      have nv := codeValid_decode s.values p (s.locals p) code cv
      have bound := nextValid_eval (Concurrent.execute (backend b) (code.decode s.values p (s.locals p)).1 s.values)
        p (s.locals p) (code.decode s.values p (s.locals p)).2 nv
      simp only [Concurrent.Controller.step,hc,bind,Option.bind,Option.some.injEq,Prod.mk.injEq] at run
      rcases run with ⟨_,rfl⟩
      intro q
      by_cases hq : q = p
      · simpa [Concurrent.update,hq] using bound
      · simpa [Concurrent.update,hq] using valid q

/-- Per-event correspondence remembers the scheduled thread for CAS's old
register, while retaining the original source event and its trace position. -/
def EventsRelated (ir : List Concurrent.Event) (source : List SourceEvent) : Prop :=
  List.Forall₂ (fun a e => ∃ p, a = event p e) ir source

theorem run_projection (b : Backend V) (plan : Plan) (h : HostArgs) (stage : Phased.Stage)
    (schedule : List Nat) (s : Concurrent.ControlState V) (valid : Valid s) :
    ∃ trace final, (controller plan h stage).run (backend b) schedule s = some (trace,final) ∧
      projectState final = (runPlan b plan h stage schedule (projectState s)).2 ∧
      EventsRelated trace (runPlan b plan h stage schedule (projectState s)).1 ∧ Valid final := by
  induction schedule generalizing s with
  | nil => exact ⟨[],s,rfl,rfl,.nil,valid⟩
  | cons p ps ih =>
      have enabled : (controller plan h stage).enabled.eval s.values p (s.locals p) =
          !evalB h stage p 0 0 plan.skip := by
        simp [controller,Concurrent.Test.eval,lowerB_eval,Concurrent.NumExpr.eval]
      have threads : (controller plan h stage).threads = h.numSms := rfl
      by_cases skip : p ≥ h.numSms || evalB h stage p 0 0 plan.skip
      · obtain ⟨trace,final,exec,projection,related,vf⟩ := ih s valid
        exact ⟨trace,final,by simpa [Concurrent.Controller.run,threads,enabled,skip] using exec,
          by simpa [runPlan,skip] using projection,
          by simpa [runPlan,skip] using related,vf⟩
      · have hp : p < h.numSms := by
          by_contra bad
          exact skip (by simp [show p ≥ h.numSms by omega])
        have step := step_projection b plan h stage p s hp (valid p)
        cases hs : (controller plan h stage).step (backend b) p s with
        | none => simp [hs] at step
        | some result =>
            rcases result with ⟨e,next⟩
            simp only [hs,Option.map_some,Option.some.injEq,Prod.mk.injEq] at step
            have vn := step_valid b plan h stage p s valid e next hs
            obtain ⟨trace,final,exec,projection,related,vf⟩ := ih next vn
            refine ⟨e::trace,final,?_,?_,?_,vf⟩
            · simp only [Concurrent.Controller.run,threads,enabled,Bool.not_not,skip,if_false,hs,
                bind,Option.bind,exec]
              simp
            · simpa only [runPlan,skip,if_false,← step.2] using projection
            · simp only [runPlan,skip,if_false,← step.2]
              exact .cons ⟨p,step.1⟩ related

end VeriTile.Bench.Repairs.SpinningLock.ConcurrentSource
