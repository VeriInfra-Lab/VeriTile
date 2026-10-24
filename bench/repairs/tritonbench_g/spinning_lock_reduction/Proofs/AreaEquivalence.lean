import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.IOImplements
import bench.repairs.tritonbench_g.spinning_lock_reduction.ProgramView
import Mathlib.Tactic.IntervalCases

namespace VeriTile.Bench.Repairs.SpinningLock.AreaEquivalence
open VeriTile Triton VeriTile.Bench.TritonBenchG.SpinningLock SourceReader ConcurrentSource
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 300000

def areaPlan : Plan := (compile areaKernel).get (by decide)
theorem candidate_compiles : compile areaKernel = some areaPlan := rfl

theorem load_address (h : HostArgs) (stage : Phased.Stage) (p q l : Nat) :
    address h stage p q l sourcePlan.load = address h stage p q l areaPlan.load := by
  change 0 + ((BitVec.ofNat 64 q * BitVec.ofNat 64 h.bm) * BitVec.ofNat 64 h.bn).toInt + _ + _ =
    0 + (BitVec.ofNat 64 q * (BitVec.ofNat 64 h.bm * BitVec.ofNat 64 h.bn)).toInt + _ + _
  rw [BitVec.mul_assoc]

theorem publish_address (h : HostArgs) (stage : Phased.Stage) (p q l : Nat) :
    address h stage p q l sourcePlan.publish = address h stage p q l areaPlan.publish := by
  change 0 + ((BitVec.ofNat 64 p * BitVec.ofNat 64 h.bm) * BitVec.ofNat 64 h.bn).toInt + _ + _ =
    0 + (BitVec.ofNat 64 p * (BitVec.ofNat 64 h.bm * BitVec.ofNat 64 h.bn)).toInt + _ + _
  rw [BitVec.mul_assoc]

theorem code_decode_update (plan : Plan) (load publish : Pointer)
    (h : HostArgs) (stage : Phased.Stage) (s : Concurrent.State V)
    (p : Nat) (locals : Nat → Nat) (label : Nat)
    (loadEqual : ∀ p q l, address h stage p q l plan.load = address h stage p q l load)
    (publishEqual : ∀ p q l, address h stage p q l plan.publish = address h stage p q l publish) :
    ((controller plan h stage).code[label]?).map (Concurrent.Code.decode s p locals) =
    ((controller {plan with load:=load,publish:=publish} h stage).code[label]?).map
      (Concurrent.Code.decode s p locals) := by
  by_cases small : label < 9
  · interval_cases label <;>
      simp only [controller,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.map_some,
        Concurrent.Code.decode,Concurrent.EventCode.instantiate,Concurrent.ValueCode.instantiate,
        lowerP_eval,loadEqual,publishEqual]
  · have large : 9 ≤ label := by omega
    have a : (controller plan h stage).code.length = 9 := rfl
    have b : (controller {plan with load:=load,publish:=publish} h stage).code.length = 9 := rfl
    rw [List.getElem?_eq_none (by omega),List.getElem?_eq_none (by omega)]

theorem plan_shape : areaPlan = {sourcePlan with load:=areaPlan.load,publish:=areaPlan.publish} := rfl

theorem code_decode (h : HostArgs) (stage : Phased.Stage) (s : Concurrent.State V)
    (p : Nat) (locals : Nat → Nat) (label : Nat) :
    ((controller sourcePlan h stage).code[label]?).map (Concurrent.Code.decode s p locals) =
    ((controller areaPlan h stage).code[label]?).map (Concurrent.Code.decode s p locals) := by
  simpa only [← plan_shape] using code_decode_update sourcePlan areaPlan.load areaPlan.publish
    h stage s p locals label (load_address h stage) (publish_address h stage)

theorem step_equal (b : Backend V) (h : HostArgs) (stage : Phased.Stage)
    (p : Nat) (s : Concurrent.ControlState V) :
    (controller sourcePlan h stage).step (backend b) p s =
      (controller areaPlan h stage).step (backend b) p s := by
  let finish : Concurrent.Event × Concurrent.Next → Concurrent.Event × Concurrent.ControlState V :=
    fun (event,next) =>
      let values := Concurrent.execute (backend b) event s.values
      let (label,locals) := next.eval values p (s.locals p)
      (event,⟨values,Concurrent.update s.pc p label,Concurrent.update s.locals p locals⟩)
  have normal (plan : Plan) : (controller plan h stage).step (backend b) p s =
      (((controller plan h stage).code[s.pc p]?).map
        (Concurrent.Code.decode s.values p (s.locals p))).map finish := by
    unfold Concurrent.Controller.step
    cases (controller plan h stage).code[s.pc p]? <;> rfl
  rw [normal,normal,code_decode]

theorem run_congr (a z : Concurrent.Controller) (b : Concurrent.Backend V)
    (threads : a.threads = z.threads) (enabled : a.enabled = z.enabled)
    (steps : ∀ p s, a.step b p s = z.step b p s)
    (schedule : List Nat) (s : Concurrent.ControlState V) :
    a.run b schedule s = z.run b schedule s := by
  induction schedule generalizing s with
  | nil => rfl
  | cons p rest ih =>
    simp only [Concurrent.Controller.run,threads,enabled,steps,ih]

theorem threads_update (plan : Plan) (load publish : Pointer) (h : HostArgs) (stage : Phased.Stage) :
    (controller plan h stage).threads = (controller {plan with load:=load,publish:=publish} h stage).threads := rfl

theorem enabled_update (plan : Plan) (load publish : Pointer) (h : HostArgs) (stage : Phased.Stage) :
    (controller plan h stage).enabled = (controller {plan with load:=load,publish:=publish} h stage).enabled := rfl

theorem initial_update (plan : Plan) (load publish : Pointer) (h : HostArgs) (stage : Phased.Stage)
    (s : Concurrent.State V) :
    (controller plan h stage).initial s =
      (controller {plan with load:=load,publish:=publish} h stage).initial s := rfl

theorem done_update (plan : Plan) (load publish : Pointer) (h : HostArgs) (stage : Phased.Stage)
    (s : Concurrent.ControlState V) :
    (controller plan h stage).Done s ↔
      (controller {plan with load:=load,publish:=publish} h stage).Done s := Iff.rfl

theorem run_equal (b : Backend V) (h : HostArgs) (stage : Phased.Stage)
    (schedule : List Nat) (s : Concurrent.ControlState V) :
    (controller sourcePlan h stage).run (backend b) schedule s =
      (controller areaPlan h stage).run (backend b) schedule s := by
  have threads := threads_update sourcePlan areaPlan.load areaPlan.publish h stage
  have enabled := enabled_update sourcePlan areaPlan.load areaPlan.publish h stage
  rw [← plan_shape] at threads enabled
  exact run_congr (controller sourcePlan h stage) (controller areaPlan h stage) (backend b)
    threads enabled (step_equal b h stage) schedule s

theorem candidate_decoded (h : HostArgs) (stage : Phased.Stage) :
    ConcurrentSource.decode h stage areaKernel = some (.controller (controller areaPlan h stage)) := by
  simp only [ConcurrentSource.decode,candidate_compiles,bind,Option.bind]
  have assertion : evalB h stage 0 0 0 areaPlan.assertion = true := by
    cases stage <;> rfl
  rw [assertion]
  rfl

theorem runs_congr (a z : Concurrent.Controller) (b : Concurrent.Backend V)
    (initial : ∀ s : Concurrent.State V, a.initial s = z.initial s)
    (runs : ∀ schedule s, a.run b schedule s = z.run b schedule s)
    (done : ∀ s : Concurrent.ControlState V, a.Done s ↔ z.Done s)
    (before final : Concurrent.State V) (events : List Concurrent.Event) :
    Concurrent.Program.Runs b (.controller a) before events final ↔
      Concurrent.Program.Runs b (.controller z) before events final := by
  simp only [Concurrent.Program.Runs,initial,runs,done]

theorem runs_equal (b : Backend V) (h : HostArgs) (stage : Phased.Stage)
    (before final : Concurrent.State V) (events : List Concurrent.Event) :
    Concurrent.Program.Runs (backend b) (.controller (controller sourcePlan h stage)) before events final ↔
    Concurrent.Program.Runs (backend b) (.controller (controller areaPlan h stage)) before events final := by
  have initial : ∀ s : Concurrent.State V,
      (controller sourcePlan h stage).initial s = (controller areaPlan h stage).initial s := by
    simpa only [← plan_shape] using initial_update (V:=V) sourcePlan areaPlan.load areaPlan.publish h stage
  have done : ∀ s : Concurrent.ControlState V,
      (controller sourcePlan h stage).Done s ↔ (controller areaPlan h stage).Done s := by
    simpa only [← plan_shape] using done_update (V:=V) sourcePlan areaPlan.load areaPlan.publish h stage
  exact runs_congr (controller sourcePlan h stage) (controller areaPlan h stage) (backend b)
    initial (run_equal b h stage) done before final events

theorem equivalent (b : Backend V) (addzero : b.add b.zero b.zero = b.zero)
    (h : HostArgs) (buf : Wide.Buffers h) (stage : Phased.Stage) :
    ProgramView.Equivalent b
      (⟨kernel⟩ : ProgramView.Implementation b h buf.cElements stage) ⟨areaKernel⟩ := by
  intro input admitted before binding
  obtain ⟨program,decoded,events,final,ran,safe,output,frame⟩ :=
    IOProof.implements b addzero h buf stage input admitted before binding
  change ConcurrentSource.decode h stage kernel = some program at decoded
  have decodedOriginal := ConcurrentSource.source_decoded h stage
  have programEqual : program = .controller (controller sourcePlan h stage) := by
    rw [decoded] at decodedOriginal
    exact Option.some.inj decodedOriginal
  subst program
  refine ⟨.controller (controller sourcePlan h stage),.controller (controller areaPlan h stage),
    events,final,decoded,candidate_decoded h stage,ran,
    (runs_equal b h stage before final events).mp ran,safe,safe,output,output,frame,frame⟩

#axiomsClean run_equal
#axiomsClean runs_equal
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.SpinningLock.AreaEquivalence
