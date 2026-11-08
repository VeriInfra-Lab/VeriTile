import VeriTile.Triton.DSL.FP32BooleanArithmetic
import VeriTile.Triton.Float.ScalarBackend
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic.NormNum

namespace VeriTile.Tests.FP32BooleanArithmetic
open VeriTile.Triton FP.Structural
noncomputable section
instance : Inhabited (FP.Value .fp32) := ⟨FP.Value.zero .fp32⟩

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := []
  constexpr := ["B"]
  readPorts := ["X"]
  writePorts := ["Y"]

def copied (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    offsets = tl.arange(0, B)
    values = tl.load(X + offsets)
    flag = offsets < 0
    result = values * flag
    tl.store(Y + offsets, result)
  }

/-- Boolean promotion preserves the complete typed program, with the
floating multiplication still present at every lane and every source shape. -/
theorem complete_program (X Y : RegionName) (B : Nat) : copied X Y B =
    (with_triton_signature signature in triton {
      offsets = tl.arange(0, B)
      values = tl.load(X + offsets)
      flag = offsets < 0
      result = values * tl.where(flag, 1.0, 0.0)
      tl.store(Y + offsets, result)
    }) := rfl

def booleanLeft (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    offsets = tl.arange(0, B)
    values = tl.load(X + offsets)
    flag = offsets < B
    saved = flag
    result = saved * values
    tl.store(Y + offsets, result)
  }

theorem operand_order (X Y : RegionName) (B : Nat) : booleanLeft X Y B =
    (with_triton_signature signature in triton {
      offsets = tl.arange(0, B)
      values = tl.load(X + offsets)
      flag = offsets < B
      saved = flag
      result = tl.where(saved, 1.0, 0.0) * values
      tl.store(Y + offsets, result)
    }) := rfl

def rest : Algebra (FP.Value .fp32) where
  literal := fun _ _ _ => FP.Value.zero .fp32
  negInf := FP.Value.infinity .fp32 Bool.true
  binary := fun _ _ _ a _ => a
  unary := fun _ _ a => a
  cast := fun _ _ _ a => a
  fromNat := fun _ _ => FP.Value.zero .fp32
  fromInt := fun _ _ => FP.Value.zero .fp32
  fp32Bits := fun b => ⟨b.bits⟩
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => FP.Value.zero .fp32
  reduceSum := fun _ {_} _ _ _ _ => FP.Value.zero .fp32

def initial : State (FP.Value .fp32) where
  mem := fun r _ => .mk .real (if r="X" then FP.Value.qNaN .fp32 else FP.Value.zero .fp32)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

/-- Concrete software FP32 NaN multiplied by a false Boolean remains NaN.
Replacing the multiplication with lane selection would instead store zero. -/
theorem nan_times_false :
    ∃ t, exec (FP.Software.scalarFP32 rest) (copied "X" "Y" 1) initial = some t ∧
      t.mem "Y" 0 = .mk .real (FP.Value.qNaN .fp32) := by
  refine ⟨?_,?_,?_⟩
  rotate_left
  · simp [copied,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,Option.bind,store,bop,numericLt,ComputeDType.eraseDType,Region.cast]
    rfl
  · rfl

private theorem multiply_signed_zero :
    FP.mul FP.Config.ieeeValue (FP.Value.zero .fp32 Bool.true) (FP.Value.zero .fp32) =
      FP.Value.zero .fp32 Bool.true := by decide +kernel

/-- Signed zero also observes the retained multiplication: -0 times the
promoted +0 remains -0 in the nearest-even software profile. -/
theorem negative_zero_times_false :
    let start : State (FP.Value .fp32) := { initial with
      mem := fun r _ => .mk .real (if r="X" then FP.Value.zero .fp32 Bool.true else FP.Value.zero .fp32) }
    ∃ t, exec (FP.Software.scalarFP32 rest) (copied "X" "Y" 1) start = some t ∧
      t.mem "Y" 0 = .mk .real (FP.Value.zero .fp32 Bool.true) := by
  refine ⟨?_,?_,?_⟩
  rotate_left
  · simp [copied,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,Option.bind,store,bop,numericLt,ComputeDType.eraseDType,Region.cast]
    rfl
  · simp only [FP.Software.scalarFP32, FP.Software.fp32Context,
      show (0.0 : ℝ) = 0 by norm_num, ite_true,
      bop, Broadcast.leftIndex, Broadcast.rightIndex, numeric,
      Nat.lt_irrefl, decide_false, Bool.false_eq_true, if_false, id_eq, multiply_signed_zero]
    rfl

def fp64Signature : CallSignature where
  pointers := [("X",.fp64),("Y",.fp64)]
  scalars := []
  constexpr := ["B"]

/-- error: numeric Boolean frontend requires FP32 pointer storage -/
#guard_msgs in
def unsupportedPrecision (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature fp64Signature in triton {
    flag = tl.arange(0, B) < B
    value = tl.load(X)
    tl.store(Y, value * flag)
  }

/-- error: numeric Boolean frontend currently requires straight-line assignments and stores -/
#guard_msgs in
def dynamicMerge (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    if tl.program_id(0) > 0:
        value = tl.load(X)
    tl.store(Y, 0.0)
  }

#axiomsClean complete_program
#axiomsClean operand_order
#axiomsClean nan_times_false
#axiomsClean negative_zero_times_false
end
end VeriTile.Tests.FP32BooleanArithmetic
