import VeriTile.Triton.DSL.OutputDTypeMetadata
import VeriTile.Triton.Float.StructuralSafety
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.OutputDTypeMetadata
open VeriTile.Triton FP.Structural
open VeriTile.Triton.DSL.OutputDTypeMetadata

def signature : CallSignature where
  pointers := [("X",.fp16),("Y",.fp16)]
  scalars := [("N",.integer)]
  constexpr := ["B","OUT_DTYPE"]
  readPorts := ["X"]
  writePorts := ["Y"]

def copied (X Y : RegionName) (N B : Nat) («OUT_DTYPE» : FloatDType) : ComputeKernel :=
  with_triton_call signature call_args [N := N] in triton {
    pointer = tl.make_block_ptr(base=X, shape=(N + B, B), strides=(B, 1), offsets=(0, 0), block_shape=(B, B), order=(1, 0))
    value = tl.load(pointer, boundary_check=(0, 1), padding_option="zero")
    scaled = value.to(OUT_DTYPE)
    acc = tl.zeros([B, B], dtype=OUT_DTYPE)
    acc += tl.dot(scaled, tl.trans(scaled))
    tl.store(Y + tl.arange(0, B)[:, None] * B + tl.arange(0, B)[None, :], acc.to(OUT_DTYPE))
  }

def reference16 (X Y : RegionName) (N B : Nat) : ComputeKernel :=
  with_triton_call (withoutOutputType signature) call_args [N := N] in triton {
    pointer = tl.make_block_ptr(base=X, shape=(N + B, B), strides=(B, 1), offsets=(0, 0), block_shape=(B, B), order=(1, 0))
    value = tl.load(pointer, boundary_check=(0, 1), padding_option="zero")
    scaled = value.to(tl.float16)
    acc = tl.zeros([B, B], dtype=tl.float16)
    acc += tl.dot(scaled, tl.trans(scaled))
    tl.store(Y + tl.arange(0, B)[:, None] * B + tl.arange(0, B)[None, :], acc.to(tl.float16))
  }

def referenceBF16 (X Y : RegionName) (N B : Nat) : ComputeKernel :=
  with_triton_call (withoutOutputType signature) call_args [N := N] in triton {
    pointer = tl.make_block_ptr(base=X, shape=(N + B, B), strides=(B, 1), offsets=(0, 0), block_shape=(B, B), order=(1, 0))
    value = tl.load(pointer, boundary_check=(0, 1), padding_option="zero")
    scaled = value.to(tl.bfloat16)
    acc = tl.zeros([B, B], dtype=tl.bfloat16)
    acc += tl.dot(scaled, tl.trans(scaled))
    tl.store(Y + tl.arange(0, B)[:, None] * B + tl.arange(0, B)[None, :], acc.to(tl.bfloat16))
  }

theorem full_program16 (X Y : RegionName) (N B : Nat) :
    copied X Y N B .fp16 = reference16 X Y N B := by rfl

theorem full_programBF16 (X Y : RegionName) (N B : Nat) :
    copied X Y N B .bf16 = referenceBF16 X Y N B := by rfl

theorem full_execution16 {α : Type} [Inhabited α] (M : Algebra α)
    (X Y : RegionName) (N B : Nat) (entry : State α) :
    exec M (copied X Y N B .fp16) entry = exec M (reference16 X Y N B) entry := by
  rw [full_program16]

theorem full_executionBF16 {α : Type} [Inhabited α] (M : Algebra α)
    (X Y : RegionName) (N B : Nat) (entry : State α) :
    exec M (copied X Y N B .bf16) entry = exec M (referenceBF16 X Y N B) entry := by
  rw [full_programBF16]

theorem full_safety16 {α : Type} [Inhabited α] (M : Algebra α) (bounds : RegionName → Nat)
    (X Y : RegionName) (N B : Nat) (entry : State α) :
    TraceSafe M bounds (copied X Y N B .fp16) entry ↔ TraceSafe M bounds (reference16 X Y N B) entry := by
  rw [full_program16]

theorem full_safetyBF16 {α : Type} [Inhabited α] (M : Algebra α) (bounds : RegionName → Nat)
    (X Y : RegionName) (N B : Nat) (entry : State α) :
    TraceSafe M bounds (copied X Y N B .bf16) entry ↔ TraceSafe M bounds (referenceBF16 X Y N B) entry := by
  rw [full_programBF16]

/-- error: <input>:2:0: expected '}' -/
#guard_msgs in
def mutated (X Y : RegionName) (N B : Nat) («OUT_DTYPE» : FloatDType) : ComputeKernel :=
  with_triton_call signature call_args [N := N] in triton {
    OUT_DTYPE = 0
    value = tl.load(X).to(OUT_DTYPE)
  }

/-- error: <input>:2:4: expected identifier -/
#guard_msgs in
def loop_shadow (X Y : RegionName) (N B : Nat) («OUT_DTYPE» : FloatDType) : ComputeKernel :=
  with_triton_call signature call_args [N := N] in triton {
    for OUT_DTYPE in range(2):
        value = tl.load(X).to(OUT_DTYPE)
  }

/-- error: <input>:2:8: expected 'tl.max(' or tritonExpr -/
#guard_msgs in
def runtime_use (X Y : RegionName) (N B : Nat) («OUT_DTYPE» : FloatDType) : ComputeKernel :=
  with_triton_call signature call_args [N := N] in triton {
    value = OUT_DTYPE
  }

/-- error: OUT_DTYPE must be declared exactly once -/
#guard_msgs in
def duplicate (X Y : RegionName) (N B : Nat) («OUT_DTYPE» : FloatDType) : ComputeKernel :=
  with_triton_call {signature with constexpr := ["B","OUT_DTYPE","OUT_DTYPE"]} call_args [N := N] in triton {
    value = tl.load(X).to(OUT_DTYPE)
  }

/-- error: OUT_DTYPE must be a distinct constexpr parameter -/
#guard_msgs in
def scalar_collision (X Y : RegionName) (N B : Nat) («OUT_DTYPE» : FloatDType) : ComputeKernel :=
  with_triton_call {signature with scalars := [("N",.integer),("OUT_DTYPE",.fp32)]} call_args [N := N] in triton {
    value = tl.load(X).to(OUT_DTYPE)
  }

#axiomsClean full_program16
#axiomsClean full_programBF16
#axiomsClean full_execution16
#axiomsClean full_executionBF16
#axiomsClean full_safety16
#axiomsClean full_safetyBF16
end VeriTile.Tests.OutputDTypeMetadata

namespace VeriTile.Tests.OutputDTypeMetadata
open VeriTile.Triton FP.Structural
open VeriTile.Triton.DSL.OutputDTypeMetadata

/-- A closed FP16 dot result must not elaborate invalid unused dtype branches. -/
def closedHalfDot (X Y : RegionName) (N B : Nat) : ComputeKernel :=
  let «OUT_DTYPE» : FloatDType := .fp16
  with_triton_call signature call_args [N := N] in triton {
    values = tl.load(X + tl.arange(0, B)[:, None] * B + tl.arange(0, B)[None, :])
    acc = tl.zeros([B, B], dtype=OUT_DTYPE)
    acc += tl.dot(values, values, out_dtype=OUT_DTYPE)
    tl.store(Y + tl.arange(0, B)[:, None] * B + tl.arange(0, B)[None, :], acc.to(OUT_DTYPE))
  }

def directHalfDot (X Y : RegionName) (N B : Nat) : ComputeKernel :=
  with_triton_call (withoutOutputType signature) call_args [N := N] in triton {
    values = tl.load(X + tl.arange(0, B)[:, None] * B + tl.arange(0, B)[None, :])
    acc = tl.zeros([B, B], dtype=tl.float16)
    acc += tl.dot(values, values, out_dtype=tl.float16)
    tl.store(Y + tl.arange(0, B)[:, None] * B + tl.arange(0, B)[None, :], acc.to(tl.float16))
  }

theorem closed_half_program (X Y : RegionName) (N B : Nat) :
    closedHalfDot X Y N B = directHalfDot X Y N B := by rfl

theorem closed_half_execution {α : Type} [Inhabited α] (M : Algebra α)
    (X Y : RegionName) (N B : Nat) (entry : State α) :
    exec M (closedHalfDot X Y N B) entry = exec M (directHalfDot X Y N B) entry := by
  rw [closed_half_program]

theorem closed_half_safety {α : Type} [Inhabited α] (M : Algebra α) (bounds : RegionName → Nat)
    (X Y : RegionName) (N B : Nat) (entry : State α) :
    TraceSafe M bounds (closedHalfDot X Y N B) entry ↔ TraceSafe M bounds (directHalfDot X Y N B) entry := by
  rw [closed_half_program]

/-- error: tl.dot: bfloat16 output requires a float32/float16 dot followed by a cast -/
#guard_msgs in
def invalidClosedDot (X Y : RegionName) (N B : Nat) : ComputeKernel :=
  let «OUT_DTYPE» : FloatDType := .bf16
  with_triton_call signature call_args [N := N] in triton {
    values = tl.load(X + tl.arange(0, B)[:, None] * B + tl.arange(0, B)[None, :])
    acc = tl.dot(values, values, out_dtype=OUT_DTYPE)
  }

#axiomsClean closed_half_program
#axiomsClean closed_half_execution
#axiomsClean closed_half_safety
end VeriTile.Tests.OutputDTypeMetadata
