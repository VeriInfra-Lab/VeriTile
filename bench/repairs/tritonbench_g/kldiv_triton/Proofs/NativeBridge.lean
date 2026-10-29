import bench.repairs.tritonbench_g.kldiv_triton.Proofs.BackSafety

namespace VeriTile.Bench.Repairs.KldivTriton.NativeHost
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormBwd.Physical (Matrix)
open Host
open Native (Word)
noncomputable section
set_option autoImplicit false
set_option maxHeartbeats 5000000

/-- Exact ordinary Torch FP32 scalar ports, separate from Triton intrinsics.
The integer division port retains Python scalar conversion, zero denominators,
and actual Torch scalar division. The reduction tree is supplied separately. -/
structure TorchPorts where
  add : Word → Word → Word
  mul : Word → Word → Word
  equal : Word → Word → Bool
  dividePythonNat : Word → Nat → Word

def arithmetic (p : TorchPorts) : Arithmetic Word :=
  ⟨Native.word 0,Native.word 0x3f800000,p.add,p.mul,p.equal,p.dividePythonNat⟩

def row {M β : Type} (ops : Ops M Word β) (s : World M) (m : Matrix) (r c : Nat) : Word :=
  logical ops SwigluTriton.Words.signNegation s.memory m.view (r*m.cols+c)

/-- Independent raw-word reference, retaining per-chunk stores, source/native
reduction order and all explicit padding. No source interpreter occurs here. -/
def forward {M β : Type} (ops : Ops M Word β) (p : Native.Instructions)
    (call : ForwardCall) (initial : World M) (i : Nat) : Word :=
  let r := i/outputCols call
  let c := i%outputCols call
  if call.mode=.none then Native.forwardPoint p (blockSize call.prediction.cols) call.flag (row ops initial call.prediction r c) (row ops initial call.target r c)
  else Native.chunk p call.flag call.prediction.cols (blockSize call.prediction.cols) c
    (row ops initial call.prediction r) (row ops initial call.target r)

def backward {M β : Type} (ops : Ops M Word β) (p : Native.Instructions)
    (call : BackwardCall) (initial : World M) (i : Nat) : Word :=
  Native.backwardPoint p call.flag (logical ops SwigluTriton.Words.signNegation initial.memory call.target.view i)

theorem point_native (p : Native.Instructions) (B : Nat) (flag : Bool) (y t : Word) :
    point (Storage.nativeForward p B flag) flag y t=Native.forwardPoint p B flag y t :=
  congrArg Native.Node.value (Native.forward_lowering p B flag y t)

theorem chunk_native (p : Native.Instructions) (flag : Bool) (N B k : Nat) (ys ts : Nat → Word) :
    chunk (Storage.nativeForward p B flag) flag N B k ys ts=Native.chunk p flag N B k ys ts :=
  congrArg Native.Node.value (Native.chunk_lowering p flag N B k ys ts)

theorem derivative_native (p : Native.Instructions) (flag : Bool) (t : Word) :
    derivative (Storage.native p) flag t=Native.backwardPoint p flag t :=
  congrArg Native.Node.value (Native.backward_lowering p flag t)

theorem forward_native {M β : Type} (ops : Ops M Word β) (p : Native.Instructions)
    (call : ForwardCall) (initial : World M) :
    forwardExpected ops SwigluTriton.Words.signNegation (Storage.nativeForward p (blockSize call.prediction.cols) call.flag) call initial=forward ops p call initial := by
  funext i
  unfold forwardExpected forward
  split
  · exact point_native p _ _ _ _
  · exact chunk_native p _ _ _ _ _ _

theorem backward_native {M β : Type} (ops : Ops M Word β) (p : Native.Instructions)
    (call : BackwardCall) (initial : World M) :
    backwardExpected ops SwigluTriton.Words.signNegation (Storage.native p) call initial=backward ops p call initial := by
  funext i
  exact derivative_native p _ _

/-- The optional reference bridge has scalar primitive deferrals only. It does
not assume a host sum equation, output agreement, memory/index law, successful
kernel, broadcast identity, or context behavior. -/
structure Obligations (actual target : TorchPorts) : Prop where
  add : ∀ a b,actual.add a b=target.add a b
  mul : ∀ a b,actual.mul a b=target.mul a b
  equal : ∀ a b,actual.equal a b=target.equal a b
  dividePythonNat : ∀ a n,actual.dividePythonNat a n=target.dividePythonNat a n

theorem ports_eq (actual target : TorchPorts) (h : Obligations actual target) : actual=target := by
  cases actual;cases target
  simp only [TorchPorts.mk.injEq]
  exact ⟨funext fun a => funext (h.add a),funext fun a => funext (h.mul a),
    funext fun a => funext (h.equal a),funext fun a => funext (h.dividePythonNat a)⟩

#specNonCircular forward avoiding [forwardExpected,Storage.forwardValue,Tile.sourceRun]
#specNonCircular backward avoiding [backwardExpected,Storage.backwardValue,Backward.sourceRun]
#axiomsClean forward_native
#axiomsClean backward_native
#axiomsClean ports_eq
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.NativeHost
