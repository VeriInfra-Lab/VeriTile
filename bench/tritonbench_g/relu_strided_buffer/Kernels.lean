import bench.tritonbench_g.relu_strided_buffer.Proofs.Support.SourceTypes
import VeriTile.Triton.DSL.SignedRankOneBlockPointers

namespace VeriTile.Bench.Repairs.ReluStridedBuffer
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

/-- Copied original unary JIT body, shared by both callers. -/
def reluForwardSource : PureJitFunction := with_triton_params (x) constexpr_args [] in triton {
    return tl.where(x > 0, x, 0)
}

/-- Selected FP32 pointer storage; source scalars and constexprs come from the invocation. -/
def signature : CallSignature where
  pointers := []
  floatingPointers := [("in0_ptr",.fp32),("out0_ptr",.fp32)]
  scalars := [("in0_stride0",.integer),("out0_stride0",.integer),("s0",.integer),
    ("num_tasks",.integer),("tiles_per_cta",.integer)]
  constexpr := ["in0_stride_order0","out0_stride_order0","tile_size0","one_tile_per_cta"]
  readPorts := ["in0_ptr"]
  writePorts := ["out0_ptr"]

def candidateSignature : CallSignature where
  pointers := []
  floatingPointers := [("in0_ptr",.fp32),("out0_ptr",.fp32)]
  scalars := [("in0_stride0",.integer),("out0_stride0",.integer),("s0",.integer),
    ("num_tasks",.integer),("tiles_per_cta",.integer)]
  constexpr := ["in0_stride_order0","out0_stride_order0","tile_size0","one_tile_per_cta","folded_num_tiles0"]
  readPorts := ["in0_ptr"]
  writePorts := ["out0_ptr"]

def originalKernel (in0_ptr out0_ptr : RegionName) (input_base output_base in0_stride0 out0_stride0 : Int)
    (s0 num_tasks tiles_per_cta tile_size0 : Nat) (one_tile_per_cta : Bool) : ComputeKernel :=
  let in0_stride_order0 : Nat := 0
  let out0_stride_order0 : Nat := 0
  with_triton_call signature call_args [in0_stride0 := in0_stride0,out0_stride0 := out0_stride0,
    s0 := s0,num_tasks := num_tasks,tiles_per_cta := tiles_per_cta]
    signed_pointer_offsets [in0_ptr := input_base,out0_ptr := output_base]
    pure_functions [relu_forward := reluForwardSource] in triton {
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
        out0 = relu_forward(in0)

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
            out0 = relu_forward(in0)

            # stores, note that store to block pointer does not automatically cast the value to the pointer's dtype
            out0_bptr = tl.make_block_ptr(out0_ptr, (s0,), (out0_stride0,), (offset0,), (tile_size0,), order=(out0_stride_order0,))
            tl.store(out0_bptr, out0.to(out0_bptr.type.element_ty), boundary_check=(out0_stride_order0,))
  }

def candidateKernel (in0_ptr out0_ptr : RegionName) (input_base output_base in0_stride0 out0_stride0 : Int)
    (s0 num_tasks tiles_per_cta tile_size0 : Nat) (one_tile_per_cta : Bool) (folded_num_tiles0 : Nat) : ComputeKernel :=
  let in0_stride_order0 : Nat := 0
  let out0_stride_order0 : Nat := 0
  with_triton_call candidateSignature call_args [in0_stride0 := in0_stride0,out0_stride0 := out0_stride0,
    s0 := s0,num_tasks := num_tasks,tiles_per_cta := tiles_per_cta]
    signed_pointer_offsets [in0_ptr := input_base,out0_ptr := output_base]
    pure_functions [relu_forward := reluForwardSource] in triton {
    pid = tl.program_id(0)
    num_tiles0 = folded_num_tiles0
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
        out0 = relu_forward(in0)

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
            out0 = relu_forward(in0)

            # stores, note that store to block pointer does not automatically cast the value to the pointer's dtype
            out0_bptr = tl.make_block_ptr(out0_ptr, (s0,), (out0_stride0,), (offset0,), (tile_size0,), order=(out0_stride_order0,))
            tl.store(out0_bptr, out0.to(out0_bptr.type.element_ty), boundary_check=(out0_stride_order0,))
  }

def oneTileKernel (c : Parameters) : ComputeKernel :=
  originalKernel c.input.region c.output.region c.input.base c.output.base c.input.stride c.output.stride
    c.n c.n c.tiles c.B Bool.true

def gridStrideKernel (c : Parameters) : ComputeKernel :=
  originalKernel c.input.region c.output.region c.input.base c.output.base c.input.stride c.output.stride
    c.n c.n c.tiles c.B Bool.false

def oneTileFolded (c : Parameters) : ComputeKernel :=
  candidateKernel c.input.region c.output.region c.input.base c.output.base c.input.stride c.output.stride
    c.n c.n c.tiles c.B Bool.true ((c.n+c.B-1)/c.B)

def gridStrideFolded (c : Parameters) : ComputeKernel :=
  candidateKernel c.input.region c.output.region c.input.base c.output.base c.input.stride c.output.stride
    c.n c.n c.tiles c.B Bool.false ((c.n+c.B-1)/c.B)

def kernel (c : Parameters) : ComputeKernel :=
  if c.oneTile then oneTileKernel c else gridStrideKernel c

def foldedKernel (c : Parameters) : ComputeKernel :=
  if c.oneTile then oneTileFolded c else gridStrideFolded c

end
end VeriTile.Bench.Repairs.ReluStridedBuffer
