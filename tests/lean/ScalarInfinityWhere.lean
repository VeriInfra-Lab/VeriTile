import VeriTile.Triton.DSL.ScalarInfinityWhere
import VeriTile.Triton.Float.StructuralSafety
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.ScalarInfinityWhere
open VeriTile.Triton FP.Structural

def halfSignature : CallSignature where
  pointers := [("X",.fp16),("Y",.fp16)]
  constexpr := ["B"]
  readPorts := ["X"]
  writePorts := ["Y"]
def bf16Signature : CallSignature where
  pointers := [("X",.bf16),("Y",.bf16)]
  constexpr := ["B"]
  readPorts := ["X"]
  writePorts := ["Y"]

def negativeHalf (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_call halfSignature call_args [] scalar_infinity_where in triton {
    x = tl.load(X + tl.arange(0, B))
    y = tl.where(tl.arange(0, B) < B, x, float("-inf"))
    tl.store(Y + tl.arange(0, B), y)
  }

def negativeHalfReference (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_call halfSignature call_args [] in triton {
    x = tl.load(X + tl.arange(0, B))
    y = tl.where(tl.arange(0, B) < B, x, (float("-inf")).to(tl.float16))
    tl.store(Y + tl.arange(0, B), y)
  }

theorem negativeHalf_program (X Y : RegionName) (B : Nat) :
    negativeHalf X Y B = negativeHalfReference X Y B := by rfl

theorem negativeHalf_execution {α : Type} [Inhabited α] (M : Algebra α)
    (X Y : RegionName) (B : Nat) (entry : State α) :
    exec M (negativeHalf X Y B) entry = exec M (negativeHalfReference X Y B) entry := by
  rw [negativeHalf_program]

theorem negativeHalf_safety {α : Type} [Inhabited α] (M : Algebra α) (bounds : RegionName → Nat)
    (X Y : RegionName) (B : Nat) (entry : State α) :
    TraceSafe M bounds (negativeHalf X Y B) entry ↔ TraceSafe M bounds (negativeHalfReference X Y B) entry := by
  rw [negativeHalf_program]

#axiomsClean negativeHalf_program
#axiomsClean negativeHalf_execution
#axiomsClean negativeHalf_safety

def negativeLeftHalf (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_call halfSignature call_args [] scalar_infinity_where in triton {
    x = tl.load(X + tl.arange(0, B))
    y = tl.where(tl.arange(0, B) < B, float("-inf"), x)
    tl.store(Y + tl.arange(0, B), y)
  }

def negativeLeftHalfReference (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_call halfSignature call_args [] in triton {
    x = tl.load(X + tl.arange(0, B))
    y = tl.where(tl.arange(0, B) < B, (float("-inf")).to(tl.float16), x)
    tl.store(Y + tl.arange(0, B), y)
  }

theorem negativeLeftHalf_program (X Y : RegionName) (B : Nat) :
    negativeLeftHalf X Y B = negativeLeftHalfReference X Y B := by rfl

theorem negativeLeftHalf_execution {α : Type} [Inhabited α] (M : Algebra α)
    (X Y : RegionName) (B : Nat) (entry : State α) :
    exec M (negativeLeftHalf X Y B) entry = exec M (negativeLeftHalfReference X Y B) entry := by
  rw [negativeLeftHalf_program]

theorem negativeLeftHalf_safety {α : Type} [Inhabited α] (M : Algebra α) (bounds : RegionName → Nat)
    (X Y : RegionName) (B : Nat) (entry : State α) :
    TraceSafe M bounds (negativeLeftHalf X Y B) entry ↔ TraceSafe M bounds (negativeLeftHalfReference X Y B) entry := by
  rw [negativeLeftHalf_program]

#axiomsClean negativeLeftHalf_program
#axiomsClean negativeLeftHalf_execution
#axiomsClean negativeLeftHalf_safety

def negativeBF16 (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_call bf16Signature call_args [] scalar_infinity_where in triton {
    x = tl.load(X + tl.arange(0, B))
    y = tl.where(tl.arange(0, B) < B, x, -float("inf"))
    tl.store(Y + tl.arange(0, B), y)
  }

def negativeBF16Reference (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_call bf16Signature call_args [] in triton {
    x = tl.load(X + tl.arange(0, B))
    y = tl.where(tl.arange(0, B) < B, x, (-float("inf")).to(tl.bfloat16))
    tl.store(Y + tl.arange(0, B), y)
  }

theorem negativeBF16_program (X Y : RegionName) (B : Nat) :
    negativeBF16 X Y B = negativeBF16Reference X Y B := by rfl

theorem negativeBF16_execution {α : Type} [Inhabited α] (M : Algebra α)
    (X Y : RegionName) (B : Nat) (entry : State α) :
    exec M (negativeBF16 X Y B) entry = exec M (negativeBF16Reference X Y B) entry := by
  rw [negativeBF16_program]

theorem negativeBF16_safety {α : Type} [Inhabited α] (M : Algebra α) (bounds : RegionName → Nat)
    (X Y : RegionName) (B : Nat) (entry : State α) :
    TraceSafe M bounds (negativeBF16 X Y B) entry ↔ TraceSafe M bounds (negativeBF16Reference X Y B) entry := by
  rw [negativeBF16_program]

#axiomsClean negativeBF16_program
#axiomsClean negativeBF16_execution
#axiomsClean negativeBF16_safety

def mixedSignature : CallSignature where
  pointers := [("X",.fp16),("Z",.fp32),("Y",.fp32)]
  constexpr := ["B"]
  readPorts := ["X","Z"]
  writePorts := ["Y"]

def rebound32 (X Z Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_call mixedSignature call_args [] scalar_infinity_where in triton {
    x = tl.load(X + tl.arange(0, B))
    x = tl.load(Z + tl.arange(0, B))
    y = tl.where(tl.arange(0, B) < B, x, float("-inf"))
    tl.store(Y + tl.arange(0, B), y)
  }
def rebound32Reference (X Z Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_call mixedSignature call_args [] in triton {
    x = tl.load(X + tl.arange(0, B))
    x = tl.load(Z + tl.arange(0, B))
    y = tl.where(tl.arange(0, B) < B, x, float("-inf"))
    tl.store(Y + tl.arange(0, B), y)
  }
theorem rebound32_program (X Z Y : RegionName) (B : Nat) :
    rebound32 X Z Y B = rebound32Reference X Z Y B := by rfl
#axiomsClean rebound32_program

def strong32 (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_call halfSignature call_args [] scalar_infinity_where in triton {
    x = tl.load(X + tl.arange(0, B))
    y = tl.where(tl.arange(0, B) < B, x, tl.full([], float("-inf"), dtype=tl.float32))
    tl.store(Y + tl.arange(0, B), y)
  }
def strong32Reference (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_call halfSignature call_args [] in triton {
    x = tl.load(X + tl.arange(0, B))
    y = tl.where(tl.arange(0, B) < B, x, tl.full([], float("-inf"), dtype=tl.float32))
    tl.store(Y + tl.arange(0, B), y)
  }
theorem strong32_program (X Y : RegionName) (B : Nat) :
    strong32 X Y B = strong32Reference X Y B := by rfl
#axiomsClean strong32_program

/-- error: scalar infinity where requires a statically known floating tensor branch -/
#guard_msgs in
def unknownType (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_call halfSignature call_args [] scalar_infinity_where in triton {
    y = tl.where(tl.arange(0, B) < B, unknown, float("-inf"))
  }

/-- error: scalar infinity where requires a statically known floating tensor branch -/
#guard_msgs in
def shadowedPointer (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_call halfSignature call_args [] scalar_infinity_where in triton {
    lo = 0
    hi = 2
    stride = 1
    for X in range(lo, hi, stride):
        x = tl.load(X)
        y = tl.where(tl.arange(0, B) < B, x, float("-inf"))
  }

/-- error: scalar infinity typing requires straight-line statements or a three-argument range -/
#guard_msgs in
def conditionalRebinding (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_call halfSignature call_args [] scalar_infinity_where in triton {
    x = tl.load(X + tl.arange(0, B))
    if True:
        x = 0.0
    y = tl.where(tl.arange(0, B) < B, x, float("-inf"))
  }
end VeriTile.Tests.ScalarInfinityWhere
