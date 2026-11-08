import VeriTile.Triton.DSL.TupleScalarCasts
import VeriTile.Triton.Float.StructuralSafety
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.TupleScalarCasts
open VeriTile.Triton FP.Structural
noncomputable section
set_option linter.unusedSimpArgs false

def signature : CallSignature where
  pointers := [("Out",.i64)]
  scalars := [("M",.integer),("N",.integer),("K",.integer)]
  constexpr := ["BLOCK"]
  writePorts := ["Out"]

def source (Out : RegionName) (M N K BLOCK : Nat) : ComputeKernel :=
  with_triton_call signature call_args [M := M,N := N,K := K] in triton {
    M, N, K = tl.cast(M, tl.int64), tl.cast(N, tl.int64), tl.cast(K, tl.int64)
    trips = M // BLOCK + (M % BLOCK != 0)
    tl.store(Out, M + N + K + trips)
  }

/-- Independent scalar statement form for the only supported tuple case. -/
def sequential (Out : RegionName) (M N K BLOCK : Nat) : ComputeKernel :=
  with_triton_call signature call_args [M := M,N := N,K := K] in triton {
    M = tl.cast(M, tl.int64)
    N = tl.cast(N, tl.int64)
    K = tl.cast(K, tl.int64)
    trips = M // BLOCK + (M % BLOCK != 0)
    tl.store(Out, M + N + K + trips)
  }

theorem complete_kernel (Out : RegionName) (M N K BLOCK : Nat) :
    source Out M N K BLOCK = sequential Out M N K BLOCK := rfl

theorem complete_execution {α : Type} [Inhabited α] (model : Algebra α)
    (Out : RegionName) (M N K BLOCK : Nat) (s : State α) :
    exec model (source Out M N K BLOCK) s = exec model (sequential Out M N K BLOCK) s := by
  rw [complete_kernel]

theorem complete_safety {α : Type} [Inhabited α] (model : Algebra α)
    (Out : RegionName) (M N K BLOCK : Nat) (s : State α) (bounds : RegionName → Nat) :
    Safety.traceSafeList model bounds (source Out M N K BLOCK).surfaceBody s ↔
      Safety.traceSafeList model bounds (sequential Out M N K BLOCK).surfaceBody s := by
  rw [complete_kernel]

/-- The three original argument values survive simultaneous casts and feed
both signed-Boolean arithmetic and an actual typed store, with an exact frame. -/
theorem successful_store {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    ∃ t, exec M (source "Out" 5 3 4 4) s = some t ∧
      t.mem = (s.write "Out" 0 (.mk .int 14)).mem := by
  simp [source,exec,run,step,stepAlg,evalExpr,evalOp_unfold,Option.bind,
    State.setReg,store,TileShape.allIndices,numeric,bop,numericLt,numericLe,numericEq,integralFloorDiv,integralMod,Region.cast,State.write]

theorem zero_remainder {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    ∃ t, exec M (source "Out" 0 3 4 4) s = some t ∧
      t.mem = (s.write "Out" 0 (.mk .int 7)).mem := by
  simp [source,exec,run,step,stepAlg,evalExpr,evalOp_unfold,Option.bind,
    State.setReg,store,TileShape.allIndices,numeric,bop,numericLt,numericLe,numericEq,integralFloorDiv,integralMod,Region.cast,State.write]

/-- error: caller tuple casts require distinct integer arguments with matching int64 self-casts -/
#guard_msgs in
def swapped (Out : RegionName) (M N K BLOCK : Nat) : ComputeKernel :=
  with_triton_call signature call_args [M := M,N := N,K := K] in triton {
    M, N = tl.cast(N, tl.int64), tl.cast(M, tl.int64)
  }

/-- error: caller tuple casts require distinct integer arguments with matching int64 self-casts -/
#guard_msgs in
def duplicated (Out : RegionName) (M N K BLOCK : Nat) : ComputeKernel :=
  with_triton_call signature call_args [M := M,N := N,K := K] in triton {
    M, M = tl.cast(M, tl.int64), tl.cast(M, tl.int64)
  }

/-- error: caller tuple casts require distinct integer arguments with matching int64 self-casts -/
#guard_msgs in
def other_width (Out : RegionName) (M N K BLOCK : Nat) : ComputeKernel :=
  with_triton_call signature call_args [M := M,N := N,K := K] in triton {
    M, N = tl.cast(M, tl.int32), tl.cast(N, tl.int32)
  }

/-- error: caller tuple casts require distinct integer arguments with matching int64 self-casts -/
#guard_msgs in
def load_in_tuple (Out : RegionName) (M N K BLOCK : Nat) : ComputeKernel :=
  with_triton_call signature call_args [M := M,N := N,K := K] in triton {
    M, N = tl.load(Out), tl.cast(N, tl.int64)
  }

#axiomsClean complete_kernel
#axiomsClean complete_execution
#axiomsClean complete_safety
#axiomsClean successful_store
#axiomsClean zero_remainder
end
end VeriTile.Tests.TupleScalarCasts
