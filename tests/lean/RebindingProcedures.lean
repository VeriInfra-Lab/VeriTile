import VeriTile.Triton.DSL.RebindingProcedures
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.RebindingProcedures
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 2000000

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := [("stride",.integer)]
  constexpr := ["N"]

def helper : ProcedureJitFunction :=
  triton_rebinding_procedure signature (X,Y,stride,N) constexpr_args [N] {
    stride = stride + 1
    value = tl.load(X + stride)
    tl.store(Y, value)
    return
  }

def called (X Y : RegionName) (stride N : Nat) : ComputeKernel :=
  with_triton_call signature call_args [stride := stride]
    rebinding_jit_procedures [copy := helper] in triton {
      copy(X, Y, stride, N=N)
      saved_stride = stride
    }

/-- The original argument survives a callee's local reassignment. Its store
uses the updated local value and retains the actual typed FP32 load. -/
theorem parameter_scope {α : Type} [Inhabited α] (M : Algebra α)
    (X Y : RegionName) (stride N : Nat) (s : State α) :
    ∃ t, exec M (called X Y stride N) s=some t ∧
      t.regs .nat [] "saved_stride"=some (fun _=>stride) ∧
      t.mem=(s.write Y 0 (.mk .real (M.fp32Load ((s.mem X (stride+1)).read .real)))).mem := by
  simp [called,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,numeric,bop,State.setReg,store,ComputeDType.eraseDType,
    ofFloat,toFloat,State.write]

#axiomsClean parameter_scope

def shapeSignature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := [("size",.integer)]
  constexpr := ["N"]
def shapeHelper : ProcedureJitFunction :=
  triton_rebinding_procedure shapeSignature (X,Y,size,N) constexpr_args [N] {
    size = tl.cast(size, tl.int64)
    pointer = tl.make_block_ptr(X, (size, N), (N, 1), (0, 0), (1, 1), (1, 0))
    value = tl.load(pointer, boundary_check=(0, 1), padding_option="zero")
    output = tl.make_block_ptr(Y, (size, N), (N, 1), (0, 0), (1, 1), (1, 0))
    tl.store(output, value, boundary_check=(0, 1))
    return
  }
def shapeCalled (X Y : RegionName) (size N : Nat) : ComputeKernel :=
  with_triton_call shapeSignature call_args [size := size]
    rebinding_jit_procedures [copy := shapeHelper] in triton {
      size = tl.cast(size, tl.int64)
      copy(X, Y, size, N=N)
      saved_size = size
    }

/-- Widened parent extents and helper arguments agree for every size, including
empty dimensions; both explicit widening assignments are still executed. -/
theorem widened_parent {α : Type} [Inhabited α] (M : Algebra α)
    (X Y : RegionName) (size N : Nat) (s : State α) :
    ∃ t, exec M (shapeCalled X Y size N) s=some t ∧
      t.regs .int [] "saved_size"=some (fun _=>(size : Int)) ∧
      t.mem=(if 0<size ∧ 0<N then
        s.write Y 0 (.mk .real (M.fp32Load ((s.mem X 0).read .real))) else s).mem := by
  by_cases active : 0<size ∧ 0<N <;>
    simp [shapeCalled,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,numeric,bop,State.setReg,store,ComputeDType.eraseDType,
    ofFloat,toFloat,State.write,evalOffsets,BlockPtr.address_2d_offsets,
    BlockPtr.inBounds_2d_offsets,TileShape.indexToList,TileShape.allIndices,
      List.finRange_succ,List.finRange_zero,active]

#axiomsClean widened_parent


def narrowed (X Y : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_call shapeSignature call_args [size := (263 : Nat)]
    rebinding_jit_procedures [] in triton {
      size = tl.cast(size, tl.int64)
      pointer = tl.make_block_ptr(X, (1, 8), (8, 1), (0, tl.cast(size, tl.int8)), (1, 1), (1, 0))
      value = tl.load(pointer, boundary_check=(0, 1), padding_option="zero")
      output = tl.make_block_ptr(Y, (1, 1), (1, 1), (0, 0), (1, 1), (1, 0))
      tl.store(output, value, boundary_check=(0, 1))
    }
/-- The source signed8 cast keeps the low word: the actual load is at 7,
not at the original 64-bit argument 2^8+7. -/
theorem narrowed_address {α : Type} [Inhabited α] (M : Algebra α)
    (X Y : RegionName) (N : Nat) (s : State α) :
    ∃ t, exec M (narrowed X Y N) s=some t ∧
      t.mem=(s.write Y 0 (.mk .real (M.fp32Load ((s.mem X 7).read .real)))).mem := by
  simp [narrowed,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,numeric,bop,State.setReg,store,ComputeDType.eraseDType,
    ofFloat,toFloat,State.write,evalOffsets,BlockPtr.address_2d_offsets,
    BlockPtr.inBounds_2d_offsets,TileShape.indexToList,TileShape.allIndices,
    List.finRange_succ,List.finRange_zero,integralMod,numericLt,natLt]
#axiomsClean narrowed_address


def weakFloat (X Y : RegionName) (size N : Nat) : ComputeKernel :=
  with_triton_call shapeSignature call_args [size := size]
    rebinding_jit_procedures [] in triton {
      size = tl.cast(size, tl.int64)
      value = tl.math.exp2(-5 - size * 1.0)
      tl.store(Y, value)
    }
/-- Python's default float literal supplies the actual FP32 operation context,
including conversion of the signed integer tensor. No backend identity is assumed. -/
theorem weak_float_precision {α : Type} [Inhabited α] (M : Algebra α)
    (X Y : RegionName) (size N : Nat) (s : State α) :
    ∃ t, exec M (weakFloat X Y size N) s=some t ∧
      t.mem=(s.write Y 0 (.mk .real (M.unary (some .fp32) .exp2
        (M.binary (some .fp32) .real .sub (M.literal (some .fp32) .real (-5))
          (M.binary (some .fp32) .real .mul (M.fromInt (some .fp32) (size : Int))
            (M.literal (some .fp32) .real 1.0)))))).mem := by
  simp [weakFloat,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,numeric,bop,State.setReg,store,ComputeDType.eraseDType,
    ofFloat,toFloat,State.write]
#axiomsClean weak_float_precision


/-- Unknown mixed-format weak promotion must not silently choose FP32. -/
theorem reject_mixed_weak_literals (X Y : RegionName) (size N : Nat) : True := by
  fail_if_success
    let bad : ComputeKernel :=
      with_triton_call ({shapeSignature with pointers := [("X",.fp16),("Y",.fp16)]} : CallSignature)
        call_args [size := size] rebinding_jit_procedures [] in triton {
          size = tl.cast(size, tl.int64)
          value = size * 1.0
          tl.store(Y, value)
        }
  trivial
#axiomsClean reject_mixed_weak_literals

end
end VeriTile.Tests.RebindingProcedures
