import VeriTile.Triton.Float.Structural
import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit

/-! Block-pointer rebinding and advance preserve source storage types. Max
promotes FP16 to FP32, then only the true narrowing store conversion remains. -/
namespace VeriTile.Tests.BlockPointerDType
open VeriTile.Triton

def profile16 : CallSignature where
  pointers := [("X",.fp16),("Y",.fp16)]
  constexpr := ["M","N"]
  readPorts := ["X"]
  writePorts := ["Y"]

def source16 (X Y : RegionName) (M N : Nat) : ComputeKernel :=
  with_triton_signature profile16 in triton {
    X = tl.make_block_ptr(base=X, shape=(M, N), strides=(N, 1),
      offsets=(0, 0), block_shape=(M, N), order=(1, 0))
    ptr = tl.advance(X, (0, 0))
    x = tl.load(ptr)
    y = tl.max(x, axis=1)
    tl.store(Y + tl.arange(0, M), y)
  }

def expanded16 (X Y : RegionName) (M N : Nat) : ComputeKernel := triton {
  tl.region X = tl.float16, Y = tl.float16
  X = tl.make_block_ptr(base=X, shape=($(M), $(N)), strides=($(N), $(1)),
    offsets=($(0), $(0)), block_shape=($(M), $(N)), order=(1, 0))
  ptr = tl.advance(X, ($(0), $(0)))
  x = tl.load(ptr)
  y = tl.max(x, axis=1)
  tl.store(Y + tl.arange(0, $(M)), y.to(tl.float16))
}

theorem complete_source16 (X Y : RegionName) (M N : Nat) :
    source16 X Y M N = expanded16 X Y M N := rfl
#axiomsClean complete_source16

def profile32 : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  constexpr := ["M","N"]
  readPorts := ["X"]
  writePorts := ["Y"]

def source32 (X Y : RegionName) (M N : Nat) : ComputeKernel :=
  with_triton_signature profile32 in triton {
    X = tl.make_block_ptr(base=X, shape=(M, N), strides=(N, 1),
      offsets=(0, 0), block_shape=(M, N), order=(1, 0))
    ptr = tl.advance(X, (0, 0))
    x = tl.load(ptr)
    y = tl.max(x, axis=1)
    tl.store(Y + tl.arange(0, M), y)
  }

def expanded32 (X Y : RegionName) (M N : Nat) : ComputeKernel := triton {
  tl.region X = tl.float32, Y = tl.float32
  X = tl.make_block_ptr(base=X, shape=($(M), $(N)), strides=($(N), $(1)),
    offsets=($(0), $(0)), block_shape=($(M), $(N)), order=(1, 0))
  ptr = tl.advance(X, ($(0), $(0)))
  x = tl.load(ptr)
  y = tl.max(x, axis=1)
  tl.store(Y + tl.arange(0, $(M)), y)
}

theorem complete_source32 (X Y : RegionName) (M N : Nat) :
    source32 X Y M N = expanded32 X Y M N := rfl
#axiomsClean complete_source32

def profile64 : CallSignature where
  pointers := [("X",.fp64),("Y",.fp64)]
  constexpr := ["M","N"]
  readPorts := ["X"]
  writePorts := ["Y"]

def source64 (X Y : RegionName) (M N : Nat) : ComputeKernel :=
  with_triton_signature profile64 in triton {
    X = tl.make_block_ptr(base=X, shape=(M, N), strides=(N, 1),
      offsets=(0, 0), block_shape=(M, N), order=(1, 0))
    ptr = tl.advance(X, (0, 0))
    x = tl.load(ptr)
    y = tl.max(x, axis=1)
    tl.store(Y + tl.arange(0, M), y)
  }

def expanded64 (X Y : RegionName) (M N : Nat) : ComputeKernel := triton {
  tl.region X = tl.float64, Y = tl.float64
  X = tl.make_block_ptr(base=X, shape=($(M), $(N)), strides=($(N), $(1)),
    offsets=($(0), $(0)), block_shape=($(M), $(N)), order=(1, 0))
  ptr = tl.advance(X, ($(0), $(0)))
  x = tl.load(ptr)
  y = tl.max(x, axis=1)
  tl.store(Y + tl.arange(0, $(M)), y)
}

theorem complete_source64 (X Y : RegionName) (M N : Nat) :
    source64 X Y M N = expanded64 X Y M N := rfl
#axiomsClean complete_source64

end VeriTile.Tests.BlockPointerDType

namespace VeriTile.Tests.PositionalBlockPtr
open VeriTile.Triton
noncomputable section

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := [("N",.integer)]
  constexpr := ["B","D"]
  readPorts := ["X","Y"]
  writePorts := ["Y"]

def pasted (X Y : RegionName) (N B D : Nat) : ComputeKernel :=
  with_triton_call signature call_args [N := N] in triton {
    source = tl.make_block_ptr(X, (N * B, D), (D, 1), (0, 0), (B, D), (1, 0))
    target = tl.make_block_ptr(Y, (B, D), (D, 1), (0, 0), (B, D), (1, 0))
    value = tl.load(source, boundary_check=(0,), padding_option="zero")
    tl.store(target, value.to(target.dtype.element_ty), boundary_check=(1,))
  }

/-- The caller's dimension product, different load/store axes, padding and
explicit destination cast survive the complete positional spelling. -/
def expected (X Y : RegionName) (N B D : Nat) : ComputeKernel := triton {
  tl.ports inputs=[X, Y], outputs=[Y]
  tl.region X = tl.float32, Y = tl.float32
  source = tl.make_block_ptr(base=X, shape=($(N * B), $(D)), strides=($(D), 1),
    offsets=(0, 0), block_shape=($(B), $(D)), order=(1, 0))
  target = tl.make_block_ptr(base=Y, shape=($(B), $(D)), strides=($(D), 1),
    offsets=(0, 0), block_shape=($(B), $(D)), order=(1, 0))
  value = tl.load(source, boundary_check=([0] : List Nat), padding_option="zero")
  tl.store(target, value.to(tl.float32), boundary_check=([1] : List Nat))
}

theorem complete_program (X Y : RegionName) (N B D : Nat) :
    pasted X Y N B D = expected X Y N B D := rfl

/-- Zero logical rows select padding for every lane, even if the input
allocation contains data. The explicit FP32 store cast remains observable. -/
theorem empty_rows_use_padding {A : Type} [Inhabited A]
    (M : FP.Structural.Algebra A) (s : FP.Structural.State A) :
    (FP.Structural.exec M (pasted ⟨"X"⟩ ⟨"Y"⟩ 0 2 2) s).map
      (fun t => t.mem ⟨"Y"⟩ 0) =
      some (FP.Structural.Cell.mk .real
        (M.cast (some .fp32) .real .real
          (M.fp32Load (M.literal (some .fp32) .real 0)))) := by
  simp [pasted, FP.Structural.exec, ComputeKernel.surfaceBody, FP.Structural.run,
    FP.Structural.step, FP.Structural.evalExpr, FP.Structural.evalComputeOp,
    FP.Structural.evalOp_unfold, FP.Structural.store,
    FP.Structural.State.setReg, TileShape.allIndices, BlockPtr.inBounds,
    BlockPtr.address, FP.Structural.ofFloat, FP.Structural.toFloat,
    Region.cast]
  rfl

#axiomsClean complete_program
#axiomsClean empty_rows_use_padding
end
end VeriTile.Tests.PositionalBlockPtr
