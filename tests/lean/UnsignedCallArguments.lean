import VeriTile.Triton.DSL.UnsignedCallArguments
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.UnsignedCast

namespace VeriTile.Tests.UnsignedCallArguments
open VeriTile.Triton
noncomputable section

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := [("n",.i64),("stride",.i64)]
  constexpr := ["B"]
  readPorts := ["X"]
  writePorts := ["Y"]

def copied (X Y : RegionName) (n : Nat) (stride : Nat) (B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [n := n, stride := stride] in triton {
    n = tl.cast(n, tl.uint64)
    offsets = tl.arange(0, B).to(tl.uint64) * stride
    x = tl.load(X + offsets, mask=offsets < n, other=0.0)
    tl.store(Y + offsets, x, mask=offsets < n)
  }

/-- Full typed program, for any natural call values, including values wrapping
at 2^64. The self-cast instruction and every later use retain unsigned modulo. -/
theorem complete_program (X Y : RegionName) (n stride B : Nat) :
    copied X Y n stride B =
      (let call : CallSignature := {signature with scalars := []}
       with_triton_signature call in triton {
         n = tl.cast($(n), tl.uint64)
         offsets = tl.arange(0, B).to(tl.uint64) * $(stride)
         x = tl.load(X + offsets, mask=offsets < tl.cast($(n), tl.uint64), other=0.0)
         tl.store(Y + offsets, x, mask=offsets < tl.cast($(n), tl.uint64))
       }) := rfl

def signed (X Y : RegionName) (n : Int) (B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [n := n, stride := (1:Nat)] in triton {
    n = n.to(tl.uint64)
    offsets = tl.arange(0, B)
    x = tl.load(X + offsets, mask=offsets < n, other=0.0)
    tl.store(Y + offsets, x, mask=offsets < n)
  }

theorem signed_program (X Y : RegionName) (n : Int) (B : Nat) :
    signed X Y n B =
      (let call : CallSignature := {signature with scalars := []}
       with_triton_signature call in triton {
         n = tl.cast($((n : Int)), tl.uint64)
         offsets = tl.arange(0, B)
         x = tl.load(X + offsets, mask=offsets < tl.cast($((n : Int)), tl.uint64), other=0.0)
         tl.store(Y + offsets, x, mask=offsets < tl.cast($((n : Int)), tl.uint64))
       }) := rfl

/-- Signed-to-unsigned casts use modulo conversion, including negative inputs;
no natural clipping or dropped cast is introduced. -/
theorem signed_cast (X Y : RegionName) (n : Int) (B : Nat) :
    (signed X Y n B).surfaceBody.head? =
      some (.assign .nat [] "n" (.alg (.intToUnsigned 64 (.constInt n)))) := rfl

/-- Concrete negative and overflow edges under arbitrary floating backends. -/
theorem negative_cast {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (u : FP.Structural.State α) :
    FP.Structural.evalOp M none (.intToUnsigned 64 (.constInt (-1))) u =
      some (fun _ => 18446744073709551615) := by
  rw [FP.Structural.eval_intToUnsigned]
  rfl

theorem overflow_cast {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (u : FP.Structural.State α) :
    FP.Structural.evalOp M none (.natToUnsigned 64 (.constNat (2^64+3))) u =
      some (fun _ => 3) := by
  norm_num [Op.natToUnsigned,FP.Structural.evalOp,FP.Structural.bop,
    FP.Structural.integralMod,IntegerExpr.sameShape]
  funext i
  cases i
  norm_num [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]

/-- error: unsigned call specialization requires one top-level self-cast and no later mutation -/
#guard_msgs in
def mutation (X Y : RegionName) (n : Nat) (B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [n := n, stride := (1:Nat)] in triton {
    n = tl.cast(n, tl.uint64)
    if True:
        n += 1
    tl.store(Y, 0.0)
  }

/-- error: a loop cannot shadow an unsigned call argument -/
#guard_msgs in
def loopShadow (X Y : RegionName) (n : Nat) (B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [n := n, stride := (1:Nat)] in triton {
    n = tl.cast(n, tl.uint64)
    for n in range(B):
        tl.store(Y, 0.0)
  }

/-- error: duplicate unsigned argument self-cast -/
#guard_msgs in
def duplicate (X Y : RegionName) (n : Nat) (B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [n := n, stride := (1:Nat)] in triton {
    n = tl.cast(n, tl.uint64)
    n = tl.cast(n, tl.uint64)
    tl.store(Y, 0.0)
  }

#axiomsClean complete_program
#axiomsClean signed_program
#axiomsClean signed_cast
#axiomsClean negative_cast
#axiomsClean overflow_cast
end
end VeriTile.Tests.UnsignedCallArguments
