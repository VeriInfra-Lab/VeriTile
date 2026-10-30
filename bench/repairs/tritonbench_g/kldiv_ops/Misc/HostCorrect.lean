import bench.repairs.tritonbench_g.kldiv_ops.Misc.ConditionalCorrect
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Checked.Real
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Checked.Direct

namespace VeriTile.Bench.Repairs.KldivOps.Contiguous.Correct
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open KldivTriton.Host
noncomputable section
set_option autoImplicit false

/-- Whole forward API in the independent Real interpretation. Ordinary host
failure is an explicit outcome with the complete caller frame. Empty means use
Real's total division convention; this is not a claim about native NaNs. -/
theorem forward {M β : Type} (ops : Ops M ℝ β) (rt : Runtime M)
    (lib : Library ops KldivTriton.RealHost.negation KldivTriton.RealHost.arithmetic rt)
    (a b : Tensor) (call : ForwardCall) (eps : ℝ) (initial : World M)
    (admit : ForwardContract a b Bool.true) (ya : Binding a call.prediction.view)
    (ta : Binding b call.target.view) (domain : ForwardDomain call initial) :
    Ensures (fun out after => Real.ForwardTarget ops call eps initial out after ∧
      Frame ops call.caller initial after ∧ after.device=initial.device)
      (fun _ after => Frame ops call.caller initial after)
      (runForward ops Real.adapter rt a b call eps initial) := by
  have frame := runForward_kept ops KldivTriton.RealHost.negation Real.adapter KldivTriton.RealHost.arithmetic rt lib a b call eps initial call.caller domain.live
  cases e : runForward ops Real.adapter rt a b call eps initial with
  | error why after => simpa only [e,Result.state] using frame
  | ok out after =>
    obtain ⟨output,shape,result,kept,device⟩ :=
      forward_success ops KldivTriton.RealHost.negation Real.adapter KldivTriton.RealHost.arithmetic rt lib a b call eps initial after out admit ya ta domain e
    exact ⟨Real.forward_target ops call eps initial after output out shape result,kept,device⟩

/-- Independent target for the actual backward API: unscaled gradient written
into the supplied input, then either that exact object for scalar-one or the
actual broadcast product. Caller supplies any forward-reduction normalization. -/
def BackwardTarget {M β : Type} (ops : Ops M ℝ β) (call : BackwardCall) (initial : World M)
    (out : View) (after : World M) : Prop :=
  BackwardReturned ops KldivTriton.RealHost.negation KldivTriton.RealHost.arithmetic call.input call.gradient.shape
    (fun i => RealExecution.backward call.flag (logical ops KldivTriton.RealHost.negation initial.memory call.target.view i))
    (fun i => logical ops KldivTriton.RealHost.negation initial.memory call.gradient i) out after

theorem backward {M β : Type} (ops : Ops M ℝ β) (rt : Runtime M)
    (lib : Library ops KldivTriton.RealHost.negation KldivTriton.RealHost.arithmetic rt)
    (a b g : Tensor) (call : BackwardCall) (initial : World M)
    (admit : BackwardContract a b g) (ia : Binding a call.input.view)
    (ta : Binding b call.target.view) (ga : Binding g call.gradient)
    (domain : BackwardDomain call initial) :
    Ensures (fun out after => BackwardTarget ops call initial out after ∧
      Complement ops call.input.view call.caller initial after ∧ after.device=initial.device)
      (fun _ after => Complement ops call.input.view call.caller initial after)
      (runBackward ops Real.adapter rt a b g call initial) := by
  have frame := runBackward_kept ops KldivTriton.RealHost.negation Real.adapter KldivTriton.RealHost.arithmetic rt lib a b g call initial call.caller domain.live
  cases e : runBackward ops Real.adapter rt a b g call initial with
  | error why after => simpa only [e,Result.state] using frame
  | ok out after =>
    obtain ⟨result,_,device⟩ := backward_success ops KldivTriton.RealHost.negation Real.adapter KldivTriton.RealHost.arithmetic rt lib a b g call initial after out admit ia ta ga domain e
    have values : expectedBackward ops KldivTriton.RealHost.negation Real.adapter call initial=
        fun i => RealExecution.backward call.flag (logical ops KldivTriton.RealHost.negation initial.memory call.target.view i) := by
      funext i
      exact Real.backward_target _ _
    rw [values] at result
    refine ⟨result,?_,device⟩
    simpa only [e,Result.state] using frame

#specNonCircular BackwardTarget avoiding [expectedBackward,derivative,Backward.grid]
abbrev python_forward_checks_iff_preconditions := python_forward_checks_iff

abbrev python_backward_checks_iff_preconditions := python_backward_checks_iff

#axiomsClean forward
#axiomsClean backward
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Contiguous.Correct
