import VeriTile.Triton.DSL.PythonBlockAdvances
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.PythonBlockAdvances
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000

def abi : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  readPorts := ["X"]
  writePorts := ["Y"]
def source (X Y : RegionName) : ComputeKernel :=
  with_python_block_advances in
  with_triton_call abi call_args [] in triton {
    ptr = tl.make_block_ptr(base=X, shape=(4, 4), strides=(4, 1), offsets=(2, 0), block_shape=(1, 1), order=(1, 0))
    out = tl.make_block_ptr(base=Y, shape=(4, 4), strides=(4, 1), offsets=(0, 0), block_shape=(1, 1), order=(1, 0))
    for i in range(0, 2):
        ptr = tl.advance(ptr, [-1, 1])
    do = tl.load(ptr, boundary_check=(0, 1))
    tl.store(out, do)
  }

def expected {α : Type} [Inhabited α] (M : Algebra α) (s : State α) : Cell α :=
  .mk .real (M.fp32Load ((s.mem "X" 2).read .real))

/-- Two signed, ordered pointer advances reach row zero, column two.
The original Python name `do`, the loaded value and the complete frame survive. -/
theorem signed_loop {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    ∃ t, exec M (source "X" "Y") s=some t ∧
      t.mem=(s.write "Y" 0 (expected M s)).mem := by
  simp [source,expected,exec,ComputeKernel.surfaceBody,run,step,range,
    evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,
    ComputeDType.eraseDType,Option.bind,numeric,numericLt,numericLe,numericEq,
    bop,Broadcast.leftIndex,Broadcast.rightIndex,State.setReg,State.write,
    TileShape.allIndices,List.finRange_succ,Region.cast,ofFloat,toFloat]

/-- Keep the existing rejection of a pointer alias changed into a tensor. -/
theorem reject_retyped_alias : True := by
  fail_if_success
    let invalid : ComputeKernel :=
      let X := "X"
      let Y := "Y"
      with_python_block_advances in
      with_triton_call abi call_args [] in triton {
        ptr = tl.make_block_ptr(base=X, shape=(4, 4), strides=(4, 1), offsets=(0, 0), block_shape=(1, 1), order=(1, 0))
        for i in range(0, 2):
            ptr = tl.load(ptr)
        tl.store(Y, 0.0)
      }
  trivial
def shapeABI : CallSignature where
  constexpr := ["N","ALT"]
def equalShape (N : Nat) : ComputeKernel :=
  let ALT := N
  with_python_block_advances in
  with_triton_call shapeABI call_args [] in triton {
    value = tl.zeros([N, ALT], dtype=tl.float32)
    value += tl.zeros([ALT, N], dtype=tl.float32)
  }

/-- Equality is established by call metadata, never assumed for two parameters. -/
theorem equal_shape_executes {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (N : Nat) :
    ∃ t, exec M (equalShape N) s=some t ∧ ∃ v, t.regs .real [N,N] "value"=some v := by
  simp [equalShape,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,ComputeDType.eraseDType,Option.bind,numeric,bop,State.setReg]

theorem reject_independent_shapes : True := by
  fail_if_success
    let invalid : Nat → Nat → ComputeKernel := fun N ALT =>
      with_python_block_advances in
      with_triton_call shapeABI call_args [] in triton {
        value = tl.zeros([N, ALT], dtype=tl.float32)
        value += tl.zeros([ALT, N], dtype=tl.float32)
      }
  trivial

theorem reject_mutated_constexpr : True := by
  fail_if_success
    let invalid : Nat → ComputeKernel := fun N =>
      let ALT := N
      with_python_block_advances in
      with_triton_call shapeABI call_args [] in triton {
        ALT = ALT + 1
        value = tl.zeros([N, ALT], dtype=tl.float32)
      }
  trivial

def staticABI : CallSignature where
  constexpr := ["N"]
def checkedStatic : ComputeKernel :=
  let N : Nat := 2
  with_python_block_advances in
  with_triton_call staticABI call_args [] in triton {
    tl.static_assert(N == 2)
    value = tl.zeros([N], dtype=tl.float32)
  }

/-- A verified source assertion has no runtime instruction. -/
theorem assertion_is_compile_time : checkedStatic.surfaceBody.length=1 := rfl

theorem reject_false_assertion : True := by
  fail_if_success
    let invalid : ComputeKernel :=
      let N : Nat := 3
      with_python_block_advances in
      with_triton_call staticABI call_args [] in triton {
        tl.static_assert(N == 2)
        value = tl.zeros([N], dtype=tl.float32)
      }
  trivial

theorem reject_runtime_assertion : True := by
  fail_if_success
    let invalid : ComputeKernel :=
      let N : Nat := 2
      with_python_block_advances in
      with_triton_call ({scalars:=[("N",.integer)]} : CallSignature) call_args [N:=N] in triton {
        tl.static_assert(N == 2)
      }
  trivial

#axiomsClean assertion_is_compile_time
#axiomsClean equal_shape_executes
#axiomsClean signed_loop
end
end VeriTile.Tests.PythonBlockAdvances
