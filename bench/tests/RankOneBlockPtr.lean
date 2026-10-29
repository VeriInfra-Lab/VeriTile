import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.RankOneBlockPtr
open VeriTile.Triton
noncomputable section

def signature : CallSignature where
  pointers := [("X",.i32),("Y",.i32)]
  scalars := [("N",.integer),("stride",.integer)]
  constexpr := ["B"]

def pasted (X Y : RegionName) (B N : Nat) (stride origin : Int) : ComputeKernel :=
  with_triton_call signature call_args [N := N,stride := stride]
    pointer_offsets [X := origin] in triton {
    offset = (0).to(tl.int64)
    parent = N
    p = tl.make_block_ptr(X, (parent,), (stride,), (offset,), (B,), order=(0,))
    offset = (99).to(tl.int64)
    parent = 0
    x = tl.load(p, boundary_check=(0,))
    lanes = tl.arange(0, B)
    tl.store(Y + lanes, x)
  }

def expanded (X Y : RegionName) (B N : Nat) (stride origin : Int) : ComputeKernel :=
  triton {
    tl.region X = tl.int32, Y = tl.int32
    offset = (0).to(tl.int64)
    parent = $((N : Nat))
    __veritile_block_p_coordinates = $((0 : Int)) + (offset).to(tl.int64) + tl.arange(0, $((B : Nat))).to(tl.int64)
    __veritile_block_p_shape = (parent).to(tl.int64)
    p = ($((origin : Int))).to(tl.int64) + __veritile_block_p_coordinates * ($((stride : Int))).to(tl.int64)
    offset = (99).to(tl.int64)
    parent = tl.full([], $((0 : Int)), dtype=tl.int32)
    x = tl.load(X + p, mask=($((0 : Int)) <= __veritile_block_p_coordinates) & (__veritile_block_p_coordinates < __veritile_block_p_shape))
    lanes = tl.arange(0, $((B : Nat)))
    tl.store(Y + lanes, x)
  }

/-- Equality of complete executable kernels, for arbitrary signed origin/stride.
Later writes to offset and parent cannot change the captured pointer bounds. -/
theorem signed_and_captured (X Y : RegionName) (B N : Nat) (stride origin : Int) :
    pasted X Y B N stride origin = expanded X Y B N stride origin := rfl

/-- error: rank-one block-pointer axis must be zero -/
#guard_msgs in
def wrongOrder (X Y : RegionName) (B N : Nat) (stride : Int) : ComputeKernel :=
  with_triton_call signature call_args [N := N,stride := stride] in triton {
    p = tl.make_block_ptr(X, (N,), (stride,), (0,), (B,), order=(1,))
    x = tl.load(p, boundary_check=(0,))
    tl.store(Y, x)
  }

/-- error: unsupported or escaping rank-one block-pointer use -/
#guard_msgs in
def cannotCopyOpaquePointer (X Y : RegionName) (B N : Nat) (stride : Int) : ComputeKernel :=
  with_triton_call signature call_args [N := N,stride := stride] in triton {
    p = tl.make_block_ptr(X, (N,), (stride,), (0,), (B,), order=(0,))
    q = p
    x = tl.load(q)
    tl.store(Y, x)
  }

/-- error: rank-one block-pointer axis must be zero -/
#guard_msgs in
def wrongBoundary (X Y : RegionName) (B N : Nat) (stride : Int) : ComputeKernel :=
  with_triton_call signature call_args [N := N,stride := stride] in triton {
    p = tl.make_block_ptr(X, (N,), (stride,), (0,), (B,), order=(0,))
    x = tl.load(p, boundary_check=(1,))
    tl.store(Y, x)
  }

def signedOffsetSignature : CallSignature where
  pointers := [("X",.i32),("Y",.i32)]
  scalars := [("N",.integer),("stride",.integer),("initial",.integer)]
  constexpr := ["B"]

def signedOffset (X Y : RegionName) (B N : Nat) (stride initial origin : Int) : ComputeKernel :=
  with_triton_call signedOffsetSignature call_args [N := N,stride := stride,initial := initial]
    pointer_offsets [X := origin] in triton {
    p = tl.make_block_ptr(X, (N,), (stride,), (initial,), (B,), order=(0,))
    x = tl.load(p, boundary_check=(0,))
    lanes = tl.arange(0, B)
    tl.store(Y + lanes, x)
  }

/-- A logical coordinate below zero stays inactive, even when its physical
signed-stride address is a valid positive cell. Undefined padding is preserved. -/
theorem negative_coordinate_uses_undefined (M : FP.Structural.Algebra Unit)
    (s : FP.Structural.State Unit) :
    (FP.Structural.exec M (signedOffset ⟨"X"⟩ ⟨"Y"⟩ 4 3 (-1) (-1) 5) s).map
      (fun t => t.mem ⟨"Y"⟩ 0) =
      some (FP.Structural.Cell.mk .int (s.undef .int ⟨"X"⟩ 6)) := by
  simp [signedOffset,ComputeKernel.surfaceBody,FP.Structural.exec,FP.Structural.run,
    FP.Structural.step,FP.Structural.stepAlg,FP.Structural.evalExpr,
    FP.Structural.evalOp_unfold,FP.Structural.numeric,FP.Structural.numericLt,
    FP.Structural.numericLe,FP.Structural.natLt,FP.Structural.natLe,
    FP.Structural.bop,FP.Structural.store,TileShape.allIndices,Region.cast]
  rfl

#axiomsClean negative_coordinate_uses_undefined
#axiomsClean signed_and_captured
end
end VeriTile.Tests.RankOneBlockPtr
