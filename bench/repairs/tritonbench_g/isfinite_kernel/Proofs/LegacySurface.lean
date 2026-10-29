import bench.repairs.tritonbench_g.isfinite_kernel.Kernels

namespace VeriTile.Bench.Repairs.IsfiniteKernel.LegacySurface
open VeriTile.Triton
syntax "isfiniteDirect%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(isfiniteDirect%[$storage:tritonDType, $variant:ident]) => do
    let blockStart ← if variant.getId == `commuted then
      `(tritonExpr| $(B) * pid) else `(tritonExpr| pid * $(B))
    `(fun (input output : View) (n B : Nat) =>
      let in0_ptr : RegionName := "x"
      let out0_ptr : RegionName := "y"
      triton {
        tl.source_alias _isfinited = triton.language.extra.cuda.libdevice.isfinited
        tl.source_alias _finitef = triton.language.extra.cuda.libdevice.finitef
        tl.source_helper isfinite_func(x) {
          return (_isfinited(x) if x.dtype.is_fp64() else _finitef(x.to(tl.float32)))
        }
        tl.region in0_ptr = $storage, out0_ptr = tl.int1
        pid = tl.program_id(0)
        offsets = $blockStart:tritonExpr + tl.arange(0, $(B))
        mask = offsets < $(n)
        indices = offsets.to(tl.int64)
        in0 = tl.load(in0_ptr + ($((input.pointer : Int)) + indices * $((input.byteStride : Int))),
          mask=mask).to(in0_ptr.type.element_ty)
        out0 = isfinite_func(in0)
        tl.store(out0_ptr + ($((output.pointer : Int)) + indices * $((output.byteStride : Int))),
          out0.to(out0_ptr.type.element_ty), mask=mask)
      })

def source32 := isfiniteDirect%[tl.float32, original]
def source64 := isfiniteDirect%[tl.float64, original]
def candidate32 := isfiniteDirect%[tl.float32, commuted]
def candidate64 := isfiniteDirect%[tl.float64, commuted]

def source (precision : Precision) (input output : View) (n B : Nat) : ComputeKernel :=
  match precision with
  | .single => source32 input output n B
  | .double => source64 input output n B

def indexCommuted (precision : Precision) (input output : View) (n B : Nat) : ComputeKernel :=
  match precision with
  | .single => candidate32 input output n B
  | .double => candidate64 input output n B

end VeriTile.Bench.Repairs.IsfiniteKernel.LegacySurface
