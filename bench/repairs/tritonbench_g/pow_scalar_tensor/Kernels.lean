import VeriTile.Triton.DSL
import VeriTile.Triton.DSL.CallSignature

/-! Shared readable scalar-base FP32 power source. Signed block pointers are
lowered to their complete signed cell formula and logical boundary mask.
The original and candidate differ only by commuting the integer tile-offset
product; all FP32 loads, scalar conversion, libdevice pow and stores remain.
`base` is the runtime scalar parameter, represented by the DSL scalar argument
slot; FP word semantics binds that slot to the launch ABI's FP32 word. -/
namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked
open VeriTile.Triton

structure View where
  region : RegionName
  base : Int
  stride : Int
structure Parameters where
  input : View
  output : View
  n : Nat
  base : ℝ
  B : Nat
  tiles : Nat
  oneTile : Bool

def powerFunction : SourceFunction :=
  triton_function pow_func_scalar_tensor(x, exponent) in triton {
    return _pow(x.to(tl.float32), exponent)
  }

/-- The launch supplies the CUDA libdevice binding selected by the first import
branch in the original Python module. Fallback import choices remain recorded
by the source checker; they are not silently substituted for this binding. -/
def callSignature : CallSignature where
  pointers := [("in0_ptr",.fp32),("out0_ptr",.fp32)]
  scalars := [("val0",.fp32),("in0_stride0",.integer),("out0_stride0",.integer),
    ("s0",.integer),("num_tasks",.integer),("tiles_per_cta",.integer)]
  constexpr := ["in0_stride_order0","out0_stride_order0","tile_size0","one_tile_per_cta"]
  typedFP32Storage := Bool.true
  functions := [powerFunction]
  binaryImports := [("_pow","triton.language.extra.cuda.libdevice.pow")]

def originalBody (c : Parameters) : ComputeKernel :=
  let in0_ptr := c.input.region
  let out0_ptr := c.output.region
  let in0_stride_order0 : Nat := 0
  let out0_stride_order0 : Nat := 0
  let tile_size0 := c.B
  let one_tile_per_cta := c.oneTile
  with_triton_call callSignature call_args [val0 := c.base,
    in0_stride0 := c.input.stride, out0_stride0 := c.output.stride,
    s0 := c.n, num_tasks := c.n, tiles_per_cta := c.tiles]
    pointer_offsets [in0_ptr := c.input.base, out0_ptr := c.output.base] in triton {
    pid = tl.program_id(0)
    num_tiles0 = tl.cdiv(s0, tile_size0)
    if one_tile_per_cta: # monolitic kernel style
        tile_id = pid
        # pid multi index recontruction: we use c ordering, right axes changes fastest
        tile_id0 = tile_id

        # tile offsets
        offset0 = tile_id0 * tile_size0
        # loads
        in0_bptr = tl.make_block_ptr(in0_ptr, (s0,), (in0_stride0,), (offset0,), (tile_size0,), order=(in0_stride_order0,))
        in0 = tl.load(in0_bptr, boundary_check=(in0_stride_order0,)).to(in0_ptr.type.element_ty) # workaround the bug on bool, we should use the original pointer's dtype(instead of block pointer's)

        # compute
        out0 = pow_func_scalar_tensor(val0, in0)

        # stores, note that store to block pointer does not automatically cast the value to the pointer's dtype
        out0_bptr = tl.make_block_ptr(out0_ptr, (s0,), (out0_stride0,), (offset0,), (tile_size0,), order=(out0_stride_order0,))
        tl.store(out0_bptr, out0.to(out0_bptr.type.element_ty), boundary_check=(out0_stride_order0,))
    else: # grid-stride-loop style kernel
        num_ctas = tl.num_programs(0)
        for j in range(0, tiles_per_cta):
            tile_id = pid + j * num_ctas
            # pid multi index recontruction: we use c ordering, right axes changes fastest
            tile_id0 = tile_id

            # tile offsets
            offset0 = tile_id0 * tile_size0
            # loads
            in0_bptr = tl.make_block_ptr(in0_ptr, (s0,), (in0_stride0,), (offset0,), (tile_size0,), order=(in0_stride_order0,))
            in0 = tl.load(in0_bptr, boundary_check=(in0_stride_order0,)).to(in0_ptr.type.element_ty) # workaround the bug on bool, we should use the original pointer's dtype(instead of block pointer's)

            # compute
            out0 = pow_func_scalar_tensor(val0, in0)

            # stores, note that store to block pointer does not automatically cast the value to the pointer's dtype
            out0_bptr = tl.make_block_ptr(out0_ptr, (s0,), (out0_stride0,), (offset0,), (tile_size0,), order=(out0_stride_order0,))
            tl.store(out0_bptr, out0.to(out0_bptr.type.element_ty), boundary_check=(out0_stride_order0,))
  }

def commutedBody (c : Parameters) : ComputeKernel :=
  let in0_ptr := c.input.region
  let out0_ptr := c.output.region
  let in0_stride_order0 : Nat := 0
  let out0_stride_order0 : Nat := 0
  let tile_size0 := c.B
  let one_tile_per_cta := c.oneTile
  with_triton_call callSignature call_args [val0 := c.base,
    in0_stride0 := c.input.stride, out0_stride0 := c.output.stride,
    s0 := c.n, num_tasks := c.n, tiles_per_cta := c.tiles]
    pointer_offsets [in0_ptr := c.input.base, out0_ptr := c.output.base] in triton {
    pid = tl.program_id(0)
    num_tiles0 = tl.cdiv(s0, tile_size0)
    if one_tile_per_cta: # monolitic kernel style
        tile_id = pid
        # pid multi index recontruction: we use c ordering, right axes changes fastest
        tile_id0 = tile_id

        # tile offsets
        offset0 = tile_size0 * tile_id0
        # loads
        in0_bptr = tl.make_block_ptr(in0_ptr, (s0,), (in0_stride0,), (offset0,), (tile_size0,), order=(in0_stride_order0,))
        in0 = tl.load(in0_bptr, boundary_check=(in0_stride_order0,)).to(in0_ptr.type.element_ty) # workaround the bug on bool, we should use the original pointer's dtype(instead of block pointer's)

        # compute
        out0 = pow_func_scalar_tensor(val0, in0)

        # stores, note that store to block pointer does not automatically cast the value to the pointer's dtype
        out0_bptr = tl.make_block_ptr(out0_ptr, (s0,), (out0_stride0,), (offset0,), (tile_size0,), order=(out0_stride_order0,))
        tl.store(out0_bptr, out0.to(out0_bptr.type.element_ty), boundary_check=(out0_stride_order0,))
    else: # grid-stride-loop style kernel
        num_ctas = tl.num_programs(0)
        for j in range(0, tiles_per_cta):
            tile_id = pid + j * num_ctas
            # pid multi index recontruction: we use c ordering, right axes changes fastest
            tile_id0 = tile_id

            # tile offsets
            offset0 = tile_size0 * tile_id0
            # loads
            in0_bptr = tl.make_block_ptr(in0_ptr, (s0,), (in0_stride0,), (offset0,), (tile_size0,), order=(in0_stride_order0,))
            in0 = tl.load(in0_bptr, boundary_check=(in0_stride_order0,)).to(in0_ptr.type.element_ty) # workaround the bug on bool, we should use the original pointer's dtype(instead of block pointer's)

            # compute
            out0 = pow_func_scalar_tensor(val0, in0)

            # stores, note that store to block pointer does not automatically cast the value to the pointer's dtype
            out0_bptr = tl.make_block_ptr(out0_ptr, (s0,), (out0_stride0,), (offset0,), (tile_size0,), order=(out0_stride_order0,))
            tl.store(out0_bptr, out0.to(out0_bptr.type.element_ty), boundary_check=(out0_stride_order0,))
  }

def oneTileKernel (c : Parameters) : ComputeKernel := originalBody {c with oneTile := Bool.true}
def gridStrideKernel (c : Parameters) : ComputeKernel := originalBody {c with oneTile := Bool.false}
def oneTileCommuted (c : Parameters) : ComputeKernel := commutedBody {c with oneTile := Bool.true}
def gridStrideCommuted (c : Parameters) : ComputeKernel := commutedBody {c with oneTile := Bool.false}
def kernel (c : Parameters) : ComputeKernel := originalBody c
def commutedKernel (c : Parameters) : ComputeKernel := commutedBody c

end VeriTile.Bench.Repairs.PowScalarTensor.Checked
