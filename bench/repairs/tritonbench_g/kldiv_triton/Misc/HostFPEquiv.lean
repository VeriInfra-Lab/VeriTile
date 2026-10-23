import bench.repairs.tritonbench_g.kldiv_triton.Misc.ConditionalFPEquiv
import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Checked.Frames
import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Checked.Execution
import bench.repairs.tritonbench_g.kldiv_triton.Proofs.NativeBytes
import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Launch
import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Context
import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Grid

namespace VeriTile.Bench.Repairs.KldivTriton.Contiguous.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormBwd.Physical (Matrix)
open Host
open Native (Word)
noncomputable section
set_option autoImplicit false

def ForwardOutcome {M : Type} (ops : Ops M Word SwigluTriton.Words.Byte) (p : Native.Instructions) (host : NativeHost.TorchPorts)
    (call : ForwardCall) (initial : World M) (out : View) (after : World M) : Prop :=
  ∃ output : Matrix,output.rows=call.prediction.rows ∧ output.cols=outputCols call ∧
    ForwardReturned ops SwigluTriton.Words.signNegation (NativeHost.arithmetic host) call output (NativeHost.forward ops p call initial) out after ∧
    NativeHost.ForwardBytes ops p host call initial output out after ∧ Frame ops call.caller initial after ∧ after.device=initial.device

def BackwardOutcome {M : Type} (ops : Ops M Word SwigluTriton.Words.Byte) (p : Native.Instructions) (host : NativeHost.TorchPorts)
    (call : BackwardCall) (initial : World M) (out : View) (after : World M) : Prop :=
  BackwardReturned ops SwigluTriton.Words.signNegation (NativeHost.arithmetic host) call.input call.gradient.shape
    (NativeHost.backward ops p call initial) (logical ops SwigluTriton.Words.signNegation initial.memory call.gradient) out after ∧
  NativeHost.BackwardBytes ops p host call initial out after ∧ Complement ops call.input.view call.caller initial after ∧
  after.device=initial.device

/-- Whole selected FP32 API: raw words and all four bytes of the actual output,
explicit native/Torch reduction schedules, returned identity and caller frame.
The primitive ports remain explicit; no GPU conformance is asserted. -/
theorem forward {M : Type} (ops : Ops M Word SwigluTriton.Words.Byte) (codec : SwigluTriton.Words.Codec ops)
    (p : Native.Instructions) (host : NativeHost.TorchPorts) (rt : Runtime M)
    (_allocation : AllocationContract rt) (_launch : Launch.Contract rt)
    (lib : Library ops SwigluTriton.Words.signNegation (NativeHost.arithmetic host) rt)
    (a b : Tensor) (call : ForwardCall) (initial : World M)
    (admit : ForwardContract a b) (ya : MatrixBinding a call.prediction)
    (ta : MatrixBinding b call.target) (domain : ForwardDomain call initial) :
    Ensures (ForwardOutcome ops p host call initial) (fun _ after => Frame ops call.caller initial after)
      (runForward ops (Storage.nativeForward p (blockSize call.prediction.cols) call.flag) rt a b call initial) := by
  have frame := runForward_frame ops SwigluTriton.Words.signNegation (Storage.nativeForward p (blockSize call.prediction.cols) call.flag) (NativeHost.arithmetic host) rt lib a b call initial domain
  cases e : runForward ops (Storage.nativeForward p (blockSize call.prediction.cols) call.flag) rt a b call initial with
  | error why after => simpa only [e,Result.state] using frame
  | ok out after =>
    obtain ⟨output,rows,cols,result,kept,device⟩ :=
      forward_success ops SwigluTriton.Words.signNegation (Storage.nativeForward p (blockSize call.prediction.cols) call.flag) (NativeHost.arithmetic host) rt lib a b call initial after out admit ya ta domain e
    rw [NativeHost.forward_native] at result
    exact ⟨output,rows,cols,result,NativeHost.forward_bytes ops codec p host call initial output out after result,kept,device⟩

theorem backward {M : Type} (ops : Ops M Word SwigluTriton.Words.Byte) (codec : SwigluTriton.Words.Codec ops)
    (p : Native.Instructions) (host : NativeHost.TorchPorts) (rt : Runtime M)
    (_allocation : AllocationContract rt) (_launch : Launch.Contract rt)
    (lib : Library ops SwigluTriton.Words.signNegation (NativeHost.arithmetic host) rt)
    (a b g : Tensor) (call : BackwardCall) (initial : World M)
    (admit : BackwardContract a b g) (ia : MatrixBinding a call.input)
    (ta : MatrixBinding b call.target) (ga : ViewBinding g call.gradient)
    (domain : BackwardDomain call initial) :
    Ensures (BackwardOutcome ops p host call initial) (fun _ after => Complement ops call.input.view call.caller initial after)
      (runBackward ops (Storage.native p) rt a b g call initial) := by
  have frame := runBackward_complement ops SwigluTriton.Words.signNegation (Storage.native p) (NativeHost.arithmetic host) rt lib a b g call initial domain
  cases e : runBackward ops (Storage.native p) rt a b g call initial with
  | error why after => simpa only [e,Result.state] using frame
  | ok out after =>
    obtain ⟨result,_,device⟩ := backward_success ops SwigluTriton.Words.signNegation (Storage.native p) (NativeHost.arithmetic host) rt lib a b g call initial after out admit ia ta ga domain e
    rw [NativeHost.backward_native] at result
    refine ⟨result,NativeHost.backward_bytes ops codec p host call initial out after result,?_,device⟩
    simpa only [e,Result.state] using frame

/-- Optional target-primitive equivalence. Every deferred hypothesis is a
local scalar instruction/Torch primitive equality; all program, index, frame,
snapshot, return and failure behavior has already been proved above. -/
theorem forward_reference {M : Type} (ops : Ops M Word SwigluTriton.Words.Byte) (codec : SwigluTriton.Words.Codec ops)
    (actual target : Native.Instructions) (actualHost targetHost : NativeHost.TorchPorts)
    (kernel : Native.Obligations actual target) (torch : NativeHost.Obligations actualHost targetHost)
    (rt : Runtime M) (allocation : AllocationContract rt) (launch : Launch.Contract rt)
    (lib : Library ops SwigluTriton.Words.signNegation (NativeHost.arithmetic actualHost) rt)
    (a b : Tensor) (call : ForwardCall) (initial : World M)
    (admit : ForwardContract a b) (ya : MatrixBinding a call.prediction)
    (ta : MatrixBinding b call.target) (domain : ForwardDomain call initial) :
    Ensures (ForwardOutcome ops target targetHost call initial) (fun _ after => Frame ops call.caller initial after)
      (runForward ops (Storage.nativeForward actual (blockSize call.prediction.cols) call.flag) rt a b call initial) := by
  rw [Native.instructions_eq actual target kernel]
  rw [NativeHost.ports_eq actualHost targetHost torch] at lib
  exact forward ops codec target targetHost rt allocation launch lib a b call initial admit ya ta domain

theorem backward_reference {M : Type} (ops : Ops M Word SwigluTriton.Words.Byte) (codec : SwigluTriton.Words.Codec ops)
    (actual target : Native.Instructions) (actualHost targetHost : NativeHost.TorchPorts)
    (kernel : Native.Obligations actual target) (torch : NativeHost.Obligations actualHost targetHost)
    (rt : Runtime M) (allocation : AllocationContract rt) (launch : Launch.Contract rt)
    (lib : Library ops SwigluTriton.Words.signNegation (NativeHost.arithmetic actualHost) rt)
    (a b g : Tensor) (call : BackwardCall) (initial : World M)
    (admit : BackwardContract a b g) (ia : MatrixBinding a call.input)
    (ta : MatrixBinding b call.target) (ga : ViewBinding g call.gradient)
    (domain : BackwardDomain call initial) :
    Ensures (BackwardOutcome ops target targetHost call initial) (fun _ after => Complement ops call.input.view call.caller initial after)
      (runBackward ops (Storage.native actual) rt a b g call initial) := by
  rw [Native.instructions_eq actual target kernel]
  rw [NativeHost.ports_eq actualHost targetHost torch] at lib
  exact backward ops codec target targetHost rt allocation launch lib a b g call initial admit ia ta ga domain

#axiomsClean forward
#axiomsClean backward
#axiomsClean forward_reference
#axiomsClean backward_reference
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Contiguous.FPEquiv
