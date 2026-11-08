import VeriTile.Triton.DSL.InlineAsmWordAdd
import bench.tritonbench_g.int8_matmul_kernel.Proofs.WordAdd

namespace VeriTile.Tests.InlineAsmWordAdd
open VeriTile.Triton
noncomputable section

def signature : CallSignature where
  pointers := [("X",.i32),("Y",.i32)]

def source (X Y : RegionName) : ComputeKernel := with_triton_signature signature in triton {
  x = tl.load(X)
  y = tl.load(Y)
  result = tl.inline_asm_elementwise("add.u32 $0, $1, $2;",
    constraints="=r,r,r", args=[x, y], dtype=tl.int32, is_pure=True, pack=1)
}

/-- The assembly ABI produces the existing strict low-word expression. -/
theorem complete_program (X Y : RegionName) : source X Y = (triton {
    tl.region X = tl.int32, Y = tl.int32
    x = tl.load(X)
    y = tl.load(Y)
    result = ((x + y).to(tl.uint32)).to(tl.int32)
  }) := rfl

/-- The decoded output agrees with the separately defined PTX word operation,
including overflow, for every pair of mathematical input carriers. -/
theorem word_semantics (x y : Int) :
    VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.signedAdd32 x y =
      (BitVec.ofInt 32 (x+y)).toInt :=
  VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.signedAdd32_low32 x y

/-- error: unsupported inline assembly: expected pure add.u32 with =r,r,r, two inputs, int32 and pack=1 -/
#guard_msgs in
example (X Y : RegionName) : ComputeKernel := with_triton_signature signature in triton {
  result = tl.inline_asm_elementwise("sub.u32 $0, $1, $2;",
    constraints="=r,r,r", args=[x, y], dtype=tl.int32, is_pure=True, pack=1)
}

/-- error: unsupported inline assembly: expected pure add.u32 with =r,r,r, two inputs, int32 and pack=1 -/
#guard_msgs in
example (X Y : RegionName) : ComputeKernel := with_triton_signature signature in triton {
  result = tl.inline_asm_elementwise("add.u32 $0, $1, $2;",
    constraints="=r,r,r", args=[x, y], dtype=tl.int32, is_pure=False, pack=1)
}

/-- error: unsupported inline assembly: expected pure add.u32 with =r,r,r, two inputs, int32 and pack=1 -/
#guard_msgs in
example (X Y : RegionName) : ComputeKernel := with_triton_signature signature in triton {
  result = tl.inline_asm_elementwise("add.u32 $0, $1, $2;",
    constraints="=r,r,r", args=[x, y], dtype=tl.int32, is_pure=True, pack=2)
}

#axiomsClean complete_program
#axiomsClean word_semantics
end
end VeriTile.Tests.InlineAsmWordAdd
