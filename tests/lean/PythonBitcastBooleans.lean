import VeriTile.Triton.DSL.PythonBitcastBooleans
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.PythonBitcastBooleans
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2500000
set_option maxRecDepth 24000
set_option linter.unusedVariables false

def halfSignature : CallSignature where
  pointers := [("X",.fp16),("Y",.fp16)]
  scalars := [("N",.integer)]
  constexpr := ["USE_FP8"]
def byteSignature : CallSignature where
  pointers := [("X",.i8),("Y",.fp16)]
  scalars := [("N",.integer)]
  constexpr := ["USE_FP8"]

def halfSource (X Y : RegionName) (N : Nat) : ComputeKernel :=
  let USE_FP8 := Bool.false
  with_triton_call halfSignature call_args [N := N] in triton {
    for i in range(0, N):
      x = tl.load(X + i)
      if USE_FP8:
        x = x.to(tl.float8e5, bitcast=True)
        x = x.to(tl.float32)
        x = x.to(tl.float16)
      tl.store(Y + i, x)
  }
def halfExpected (X Y : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_call {halfSignature with constexpr:=[]} call_args [N := N] in triton {
    for i in range(0, N) {
      x = tl.load(X + i)
      tl.store(Y + i, x)
    }
  }
/-- The disabled byte path is never typechecked as half storage; the runtime
loop, all active loads and implicit half stores remain in the complete IR. -/
theorem half_complete_program (X Y : RegionName) (N : Nat) :
    halfSource X Y N=halfExpected X Y N := rfl

def byteSource (X Y : RegionName) (N : Nat) : ComputeKernel :=
  let USE_FP8 := Bool.true
  with_triton_call byteSignature call_args [N := N] in triton {
    for i in range(0, N):
      x = tl.load(X + i)
      if USE_FP8:
        x = x.to(tl.float8e5, bitcast=True)
        x = x.to(tl.float32)
        x = x.to(tl.float16)
      tl.store(Y + i, x)
  }
def byteExpected (X Y : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_call {byteSignature with constexpr:=[]} call_args [N := N] in triton {
    for i in range(0, N) {
      x = tl.load(X + i)
      x = tl.bitcast(x, tl.float8e5)
      x = x.to(tl.float32)
      x = x.to(tl.float16)
      tl.store(Y + i, x)
    }
  }
/-- Bit reinterpretation and both subsequent rounded conversion stages are
retained, rather than replaced with numeric int8 conversion or a direct cast. -/
theorem byte_complete_program (X Y : RegionName) (N : Nat) :
    byteSource X Y N=byteExpected X Y N := rfl

example (X Y : RegionName) (N : Nat) : True := by
  fail_if_success
    let bad : ComputeKernel :=
      let USE_FP8 := Bool.false
      with_triton_call halfSignature call_args [N := N] in triton {
        USE_FP8 = True
        x = tl.load(X)
        tl.store(Y, x)
      }
  trivial

def symbolicSignature : CallSignature where
  pointers := [("X",.fp16),("Y",.fp16)]
  scalars := [("N",.integer)]
  constexpr := ["FLAG"]
def symbolicSource (X Y : RegionName) (N : Nat) (FLAG : Bool) : ComputeKernel :=
  with_triton_call symbolicSignature call_args [N := N] in triton {
    for i in range(0, N):
      x = tl.load(X + i)
      if FLAG:
        x = x + x
      if i < N:
        tl.store(Y + i, x)
  }
def symbolicExpected (X Y : RegionName) (N : Nat) (FLAG : Bool) : ComputeKernel :=
  if FLAG then
    with_triton_call {halfSignature with constexpr:=[]} call_args [N := N] in triton {
      for i in range(0, N):
        x = tl.load(X + i)
        x = x + x
        if i < N:
          tl.store(Y + i, x)
    }
  else
    with_triton_call {halfSignature with constexpr:=[]} call_args [N := N] in triton {
      for i in range(0, N):
        x = tl.load(X + i)
        if i < N:
          tl.store(Y + i, x)
    }
/-- Symbolic Boolean arguments keep both specializations; the independent
runtime branch remains present in each complete program. -/
theorem symbolic_and_runtime_control (X Y : RegionName) (N : Nat) (FLAG : Bool) :
    symbolicSource X Y N FLAG=symbolicExpected X Y N FLAG := rfl

example (X Y : RegionName) (N : Nat) : True := by
  fail_if_success
    let bad : ComputeKernel :=
      let USE_FP8 := Bool.true
      with_triton_call byteSignature call_args [N := N] in triton {
        x = tl.load(X)
        x = x.to(tl.float8e5, bitcast=False)
        tl.store(Y, x)
      }
  trivial

#axiomsClean half_complete_program
#axiomsClean byte_complete_program
#axiomsClean symbolic_and_runtime_control

end
end VeriTile.Tests.PythonBitcastBooleans
