import bench.repairs.tritonbench_g.kldiv_ops.Misc.ConditionalFPEquiv
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Checked.Frames
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Checked.Direct
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Checked.NativeBytes

namespace VeriTile.Bench.Repairs.KldivOps.Contiguous.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormBwd.Physical (Matrix)
open KldivTriton.Host
open KldivTriton.Native (Word)
noncomputable section
set_option autoImplicit false

def ForwardOutcome {M : Type} (ops : Ops M Word SwigluTriton.Words.Byte) (p : Native.Ports) (host : KldivTriton.NativeHost.TorchPorts)
    (plans : Native.Plans) (call : ForwardCall) (eps : BitVec 64) (initial : World M) (out : View) (after : World M) : Prop :=
  ∃ output : View,output.shape=outputShape call ∧
    ForwardReturned ops SwigluTriton.Words.signNegation (KldivTriton.NativeHost.arithmetic host) call output (Native.forward ops p plans call eps initial) out after ∧
    Native.ForwardBytes ops p host plans call eps initial output out after ∧ Frame ops call.caller initial after ∧ after.device=initial.device

def BackwardOutcome {M : Type} (ops : Ops M Word SwigluTriton.Words.Byte) (p : Native.Ports) (host : KldivTriton.NativeHost.TorchPorts)
    (plans : Native.Plans) (call : BackwardCall) (initial : World M) (out : View) (after : World M) : Prop :=
  BackwardReturned ops SwigluTriton.Words.signNegation (KldivTriton.NativeHost.arithmetic host) call.input call.gradient.shape
    (Native.backward ops p call initial) (logical ops SwigluTriton.Words.signNegation initial.memory call.gradient) out after ∧
  Native.BackwardBytes ops p host call initial out after ∧ Complement ops call.input.view call.caller initial after ∧
  after.device=initial.device

/-- Whole selected FP32 API: raw words and all four bytes of the actual output,
explicit native/Torch reduction schedules, returned identity and caller frame.
The primitive ports remain explicit; no GPU conformance is asserted. -/
theorem forward {M : Type} (ops : Ops M Word SwigluTriton.Words.Byte) (codec : SwigluTriton.Words.Codec ops)
    (p : Native.Ports) (host : KldivTriton.NativeHost.TorchPorts) (rt : Runtime M)
    (_launch : LaunchContract rt)
    (lib : Library ops SwigluTriton.Words.signNegation (KldivTriton.NativeHost.arithmetic host) rt)
    (a b : Tensor) (plans : Native.Plans) (call : ForwardCall) (eps : BitVec 64) (initial : World M)
    (admit : ForwardContract a b Bool.true) (ya : Binding a call.prediction.view)
    (ta : Binding b call.target.view) (domain : ForwardDomain call initial) :
    Ensures (ForwardOutcome ops p host plans call eps initial) (fun _ after => Frame ops call.caller initial after)
      (runForward ops (Native.adapter p plans) rt a b call (p.bindEpsilon eps) initial) := by
  have frame := runForward_kept ops SwigluTriton.Words.signNegation (Native.adapter p plans) (KldivTriton.NativeHost.arithmetic host) rt lib a b call (p.bindEpsilon eps) initial call.caller domain.live
  cases e : runForward ops (Native.adapter p plans) rt a b call (p.bindEpsilon eps) initial with
  | error why after => simpa only [e,Result.state] using frame
  | ok out after =>
    obtain ⟨output,shape,result,kept,device⟩ :=
      forward_success ops SwigluTriton.Words.signNegation (Native.adapter p plans) (KldivTriton.NativeHost.arithmetic host) rt lib a b call (p.bindEpsilon eps) initial after out admit ya ta domain e
    rw [Native.forward_native] at result
    exact ⟨output,shape,result,Native.forward_bytes ops codec p host plans call eps initial output out after result,kept,device⟩

theorem backward {M : Type} (ops : Ops M Word SwigluTriton.Words.Byte) (codec : SwigluTriton.Words.Codec ops)
    (p : Native.Ports) (host : KldivTriton.NativeHost.TorchPorts) (rt : Runtime M)
    (_launch : LaunchContract rt)
    (lib : Library ops SwigluTriton.Words.signNegation (KldivTriton.NativeHost.arithmetic host) rt)
    (a b g : Tensor) (plans : Native.Plans) (call : BackwardCall) (initial : World M)
    (admit : BackwardContract a b g) (ia : Binding a call.input.view)
    (ta : Binding b call.target.view) (ga : Binding g call.gradient)
    (domain : BackwardDomain call initial) :
    Ensures (BackwardOutcome ops p host plans call initial) (fun _ after => Complement ops call.input.view call.caller initial after)
      (runBackward ops (Native.adapter p plans) rt a b g call initial) := by
  have frame := runBackward_kept ops SwigluTriton.Words.signNegation (Native.adapter p plans) (KldivTriton.NativeHost.arithmetic host) rt lib a b g call initial call.caller domain.live
  cases e : runBackward ops (Native.adapter p plans) rt a b g call initial with
  | error why after => simpa only [e,Result.state] using frame
  | ok out after =>
    obtain ⟨result,_,device⟩ := backward_success ops SwigluTriton.Words.signNegation (Native.adapter p plans) (KldivTriton.NativeHost.arithmetic host) rt lib a b g call initial after out admit ia ta ga domain e
    rw [Native.backward_native] at result
    refine ⟨result,Native.backward_bytes ops codec p host call initial out after result,?_,device⟩
    simpa only [e,Result.state] using frame

/-- Optional target-primitive equivalence. Every deferred hypothesis is a
local scalar instruction/Torch primitive equality; all program, index, frame,
snapshot, return and failure behavior has already been proved above. -/
theorem forward_reference {M : Type} (ops : Ops M Word SwigluTriton.Words.Byte) (codec : SwigluTriton.Words.Codec ops)
    (actual target : Native.Ports) (actualHost targetHost : KldivTriton.NativeHost.TorchPorts)
    (kernel : Native.LocalPorts actual target) (torch : KldivTriton.NativeHost.Obligations actualHost targetHost)
    (rt : Runtime M) (launch : LaunchContract rt)
    (lib : Library ops SwigluTriton.Words.signNegation (KldivTriton.NativeHost.arithmetic actualHost) rt)
    (a b : Tensor) (plans : Native.Plans) (call : ForwardCall) (eps : BitVec 64) (initial : World M)
    (admit : ForwardContract a b Bool.true) (ya : Binding a call.prediction.view)
    (ta : Binding b call.target.view) (domain : ForwardDomain call initial) :
    Ensures (ForwardOutcome ops target targetHost plans call eps initial) (fun _ after => Frame ops call.caller initial after)
      (runForward ops (Native.adapter actual plans) rt a b call (actual.bindEpsilon eps) initial) := by
  rw [Native.ports_eq actual target kernel]
  rw [KldivTriton.NativeHost.ports_eq actualHost targetHost torch] at lib
  exact forward ops codec target targetHost rt launch lib a b plans call eps initial admit ya ta domain

theorem backward_reference {M : Type} (ops : Ops M Word SwigluTriton.Words.Byte) (codec : SwigluTriton.Words.Codec ops)
    (actual target : Native.Ports) (actualHost targetHost : KldivTriton.NativeHost.TorchPorts)
    (kernel : Native.LocalPorts actual target) (torch : KldivTriton.NativeHost.Obligations actualHost targetHost)
    (rt : Runtime M) (launch : LaunchContract rt)
    (lib : Library ops SwigluTriton.Words.signNegation (KldivTriton.NativeHost.arithmetic actualHost) rt)
    (a b g : Tensor) (plans : Native.Plans) (call : BackwardCall) (initial : World M)
    (admit : BackwardContract a b g) (ia : Binding a call.input.view)
    (ta : Binding b call.target.view) (ga : Binding g call.gradient)
    (domain : BackwardDomain call initial) :
    Ensures (BackwardOutcome ops target targetHost plans call initial) (fun _ after => Complement ops call.input.view call.caller initial after)
      (runBackward ops (Native.adapter actual plans) rt a b g call initial) := by
  rw [Native.ports_eq actual target kernel]
  rw [KldivTriton.NativeHost.ports_eq actualHost targetHost torch] at lib
  exact backward ops codec target targetHost rt launch lib a b g plans call initial admit ia ta ga domain

#axiomsClean forward
#axiomsClean backward
#axiomsClean forward_reference
#axiomsClean backward_reference
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Contiguous.FPEquiv
