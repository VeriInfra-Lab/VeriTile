/-
Source-audit target: isfinite_func_kernel_rank_1
Source-audit Lean: isfiniteSource%
Source-audit args: storage=tl.float32
Source-audit helpers: {"kind":"isfinite-dtype-dispatch","python":["isfinite_func"],"lean":{"source":"isfiniteSource%","args":{"storage":"tl.float32"}}}
-/
import bench.tritonbench_g.isfinite_kernel.Proofs.Parameters
import VeriTile.Triton.DSL
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit

/-! Complete original and candidate typed source quotations. The candidate
changes only the order of the grid tile-index natural product. The exact
finite helper, both source branches, all parameters, boundary checks and
pointer element casts remain. These sources have no Real projection. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedSource
open VeriTile.Triton Lean

syntax "isfiniteSource%" "[" tritonDType "]" : term
set_option hygiene false in
macro_rules
  | `(isfiniteSource%[$storage:tritonDType]) =>
    `(fun (in0_ptr out0_ptr : RegionName)
      (in0_stride0 in0_stride_order0 out0_stride0 out0_stride_order0 : Nat)
      (s0 num_tasks tiles_per_cta tile_size0 : Nat) (one_tile_per_cta : Bool)
      (in_order_zero : in0_stride_order0 = 0)
      (out_order_zero : out0_stride_order0 = 0) =>
      ComputeKernel.mk [in0_ptr] [out0_ptr] (triton {
        tl.source_alias _isfinited = triton.language.extra.cuda.libdevice.isfinited
        tl.source_alias _finitef = triton.language.extra.cuda.libdevice.finitef
        tl.source_helper isfinite_func(x) {
          return (_isfinited(x) if x.dtype.is_fp64() else _finitef(x.to(tl.float32)))
        }
        tl.region in0_ptr = $storage, out0_ptr = tl.int1
        pid = tl.program_id(0)
        num_tiles0 = tl.cdiv($(s0), $(tile_size0))
        if one_tile_per_cta {
          tile_id = pid
          tile_id0 = tile_id
          offset0 = tile_id0 * $(tile_size0)
          in0_bptr = tl.make_block_ptr(base=in0_ptr, shape=($(s0)),
            strides=($(in0_stride0)), offsets=(offset0), block_shape=($(tile_size0)),
            order=(0))
          in0 = tl.load(in0_bptr,
            boundary_check=([0] : List Nat)).to(in0_ptr.type.element_ty)
          out0 = isfinite_func(in0)
          out0_bptr = tl.make_block_ptr(base=out0_ptr, shape=($(s0)),
            strides=($(out0_stride0)), offsets=(offset0), block_shape=($(tile_size0)),
            order=(0))
          tl.store(out0_bptr, out0.to(out0_bptr.type.element_ty),
            boundary_check=([0] : List Nat))
        } else {
          num_ctas = tl.num_programs(0)
          for j in range($(0), $(tiles_per_cta)) {
            tile_id = pid + j * num_ctas
            tile_id0 = tile_id
            offset0 = tile_id0 * $(tile_size0)
            in0_bptr = tl.make_block_ptr(base=in0_ptr, shape=($(s0)),
              strides=($(in0_stride0)), offsets=(offset0), block_shape=($(tile_size0)),
              order=(0))
            in0 = tl.load(in0_bptr,
              boundary_check=([0] : List Nat)).to(in0_ptr.type.element_ty)
            out0 = isfinite_func(in0)
            out0_bptr = tl.make_block_ptr(base=out0_ptr, shape=($(s0)),
              strides=($(out0_stride0)), offsets=(offset0), block_shape=($(tile_size0)),
              order=(0))
            tl.store(out0_bptr, out0.to(out0_bptr.type.element_ty),
              boundary_check=([0] : List Nat))
          }
        }
      }).surfaceBody)

def source32 := isfiniteSource%[tl.float32]
def source64 := isfiniteSource%[tl.float64]
def sourceBfloat16 := isfiniteSource%[tl.bfloat16]

end VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedSource
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.Parametric
open VeriTile.Triton Lean

syntax "isfiniteCandidate%" "[" tritonDType "]" : term
set_option hygiene false in
macro_rules
  | `(isfiniteCandidate%[$storage:tritonDType]) =>
    `(fun (in0_ptr out0_ptr : RegionName)
      (in0_stride0 in0_stride_order0 out0_stride0 out0_stride_order0 : Nat)
      (s0 num_tasks tiles_per_cta tile_size0 : Nat) (one_tile_per_cta : Bool)
      (in_order_zero : in0_stride_order0 = 0)
      (out_order_zero : out0_stride_order0 = 0) =>
      ComputeKernel.mk [in0_ptr] [out0_ptr] (triton {
        tl.source_alias _isfinited = triton.language.extra.cuda.libdevice.isfinited
        tl.source_alias _finitef = triton.language.extra.cuda.libdevice.finitef
        tl.source_helper isfinite_func(x) {
          return (_isfinited(x) if x.dtype.is_fp64() else _finitef(x.to(tl.float32)))
        }
        tl.region in0_ptr = $storage, out0_ptr = tl.int1
        pid = tl.program_id(0)
        num_tiles0 = tl.cdiv($(s0), $(tile_size0))
        if one_tile_per_cta {
          tile_id = pid
          tile_id0 = tile_id
          offset0 = tile_id0 * $(tile_size0)
          in0_bptr = tl.make_block_ptr(base=in0_ptr, shape=($(s0)),
            strides=($(in0_stride0)), offsets=(offset0), block_shape=($(tile_size0)),
            order=(0))
          in0 = tl.load(in0_bptr,
            boundary_check=([0] : List Nat)).to(in0_ptr.type.element_ty)
          out0 = isfinite_func(in0)
          out0_bptr = tl.make_block_ptr(base=out0_ptr, shape=($(s0)),
            strides=($(out0_stride0)), offsets=(offset0), block_shape=($(tile_size0)),
            order=(0))
          tl.store(out0_bptr, out0.to(out0_bptr.type.element_ty),
            boundary_check=([0] : List Nat))
        } else {
          num_ctas = tl.num_programs(0)
          for j in range($(0), $(tiles_per_cta)) {
            tile_id = pid + num_ctas * j
            tile_id0 = tile_id
            offset0 = tile_id0 * $(tile_size0)
            in0_bptr = tl.make_block_ptr(base=in0_ptr, shape=($(s0)),
              strides=($(in0_stride0)), offsets=(offset0), block_shape=($(tile_size0)),
              order=(0))
            in0 = tl.load(in0_bptr,
              boundary_check=([0] : List Nat)).to(in0_ptr.type.element_ty)
            out0 = isfinite_func(in0)
            out0_bptr = tl.make_block_ptr(base=out0_ptr, shape=($(s0)),
              strides=($(out0_stride0)), offsets=(offset0), block_shape=($(tile_size0)),
              order=(0))
            tl.store(out0_bptr, out0.to(out0_bptr.type.element_ty),
              boundary_check=([0] : List Nat))
          }
        }
      }).surfaceBody)

def candidateSource32 := isfiniteCandidate%[tl.float32]
def candidateSource64 := isfiniteCandidate%[tl.float64]

def original32 (p : Parameters) : ComputeKernel :=
  TypedSource.source32 p.input p.output p.inputStride p.inputOrder
    p.outputStride p.outputOrder p.size p.numTasks p.iterations p.block
    p.oneTile p.inputOrderZero p.outputOrderZero

def original64 (p : Parameters) : ComputeKernel :=
  TypedSource.source64 p.input p.output p.inputStride p.inputOrder
    p.outputStride p.outputOrder p.size p.numTasks p.iterations p.block
    p.oneTile p.inputOrderZero p.outputOrderZero

def candidate32 (p : Parameters) : ComputeKernel :=
  candidateSource32 p.input p.output p.inputStride p.inputOrder
    p.outputStride p.outputOrder p.size p.numTasks p.iterations p.block
    p.oneTile p.inputOrderZero p.outputOrderZero

def candidate64 (p : Parameters) : ComputeKernel :=
  candidateSource64 p.input p.output p.inputStride p.inputOrder
    p.outputStride p.outputOrder p.size p.numTasks p.iterations p.block
    p.oneTile p.inputOrderZero p.outputOrderZero

end VeriTile.Bench.TritonBenchG.IsfiniteKernel.Parametric

namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.Kernels

abbrev original32 := Parametric.original32
abbrev original64 := Parametric.original64
abbrev indexCommuted32 := Parametric.candidate32
abbrev indexCommuted64 := Parametric.candidate64

end VeriTile.Bench.TritonBenchG.IsfiniteKernel.Kernels
