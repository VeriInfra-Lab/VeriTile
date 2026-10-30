import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Checked.Native

namespace VeriTile.Bench.Repairs.KldivOps.Contiguous.Native
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open KldivTriton.Host (World ForwardCall BackwardCall)
open KldivTriton.Native (Word)
noncomputable section
set_option autoImplicit false

/-- Independent raw-word graph on the original logical inputs. Epsilon is the
actual Python binary64 scalar, converted by one explicitly registered ABI port. -/
def forward {M β : Type} (ops : Ops M Word β) (p : Ports) (plans : Plans)
    (call : ForwardCall) (eps : BitVec 64) (initial : World M) (i : Nat) : Word :=
  if call.mode = .none then loss p call.flag (p.bindEpsilon eps)
    (logical ops SwigluTriton.Words.signNegation initial.memory call.prediction.view i)
    (logical ops SwigluTriton.Words.signNegation initial.memory call.target.view i)
  else row p plans call.prediction.cols call.flag (p.bindEpsilon eps)
    (fun c => logical ops SwigluTriton.Words.signNegation initial.memory call.prediction.view (i*call.prediction.cols+c))
    (fun c => logical ops SwigluTriton.Words.signNegation initial.memory call.target.view (i*call.prediction.cols+c))

def backward {M β : Type} (ops : Ops M Word β) (p : Ports)
    (call : BackwardCall) (initial : World M) (i : Nat) : Word :=
  derivative p call.flag (logical ops SwigluTriton.Words.signNegation initial.memory call.target.view i)

theorem forward_native {M β : Type} (ops : Ops M Word β) (p : Ports) (plans : Plans)
    (call : ForwardCall) (eps : BitVec 64) (initial : World M) :
    expected ops SwigluTriton.Words.signNegation (adapter p plans) call (p.bindEpsilon eps) initial =
      forward ops p plans call eps initial := by
  funext i
  unfold expected forwardExpected forward
  have mode : (reduction call.mode = Host.Reduction.none) ↔ call.mode = .none := by
    cases call.mode <;> simp [reduction]
  by_cases h : call.mode = .none
  · simp only [h,reduction,if_true]
    exact congrArg KldivTriton.Native.Node.value (scalar_lowering p plans call.flag (p.bindEpsilon eps) _ _)
  · simp only [show reduction call.mode ≠ Host.Reduction.none from fun eq => h (mode.mp eq),h,if_false]
    exact row_lowering p plans _ _ _ _ _

theorem backward_native {M β : Type} (ops : Ops M Word β) (p : Ports) (plans : Plans)
    (call : BackwardCall) (initial : World M) :
    expectedBackward ops SwigluTriton.Words.signNegation (adapter p plans) call initial =
      backward ops p call initial := by
  funext i
  exact congrArg KldivTriton.Native.Node.value (backward_lowering p plans call.flag _)

#specNonCircular forward avoiding [forwardKernel,forwardExpected,expected,Program.grid]
#specNonCircular backward avoiding [backwardKernel,expectedBackward,Backward.grid]
#axiomsClean forward_native
#axiomsClean backward_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Contiguous.Native
