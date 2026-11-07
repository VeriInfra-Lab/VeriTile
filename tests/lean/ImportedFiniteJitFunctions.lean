import tests.lean.ImportedFiniteJitReference
import bench.tritonbench_g.isfinite_kernel.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.ImportedFiniteJitFunctions
open VeriTile.Triton VeriTile.Bench.Repairs.IsfiniteKernel

/-- Complete programs include the actual signed address, original pointer
casts, selected dtype classifier and implicit destination conversion. -/
theorem single_program (input output : View) (n B : Nat) :
    source32 input output n B = ImportedFiniteJitReference.source32 input output n B := by rfl

theorem double_program (input output : View) (n B : Nat) :
    source64 input output n B = ImportedFiniteJitReference.source64 input output n B := by rfl

theorem single_candidate (input output : View) (n B : Nat) :
    candidate32 input output n B = ImportedFiniteJitReference.candidate32 input output n B := by rfl

theorem double_candidate (input output : View) (n B : Nat) :
    candidate64 input output n B = ImportedFiniteJitReference.candidate64 input output n B := by rfl

/-- error: finite source alias requires the explicit CUDA libdevice finitef/isfinited import -/
#guard_msgs in
def unsupportedImport : ImportedFiniteJitFunction :=
  with_triton_imports [_isfinited := "triton.language.math.isfinited",
    _finitef := "triton.language.extra.cuda.libdevice.finitef"] in triton_function (x) {
    return _isfinited(x) if x.dtype.is_fp64() else _finitef(x.to(tl.float32))
  }

/-- error: finite helper narrow branch requires its explicit float32 cast -/
#guard_msgs in
def missingCast : ImportedFiniteJitFunction :=
  with_triton_imports [_isfinited := "triton.language.extra.cuda.libdevice.isfinited",
    _finitef := "triton.language.extra.cuda.libdevice.finitef"] in triton_function (x) {
    return _isfinited(x) if x.dtype.is_fp64() else _finitef(x)
  }

/-- error: signed pointer arguments must remain immutable -/
#guard_msgs in
def mutated (in0_ptr out0_ptr : RegionName) : ComputeKernel :=
  with_triton_call (callSignature Precision.single) call_args [in0_stride := 1, out0_stride := 1, n := 1]
    signed_pointer_offsets [in0_ptr := 0, out0_ptr := 0]
    pure_functions [isfinite_func := isfiniteSource] in triton {
    in0_ptr += 1
  }

/-- error: signed pointer binding requires one complete displacement -/
#guard_msgs in
def nestedAddress (in0_ptr out0_ptr : RegionName) : ComputeKernel :=
  with_triton_call (callSignature Precision.single) call_args [in0_stride := 1, out0_stride := 1, n := 1]
    signed_pointer_offsets [in0_ptr := 0, out0_ptr := 0]
    pure_functions [isfinite_func := isfiniteSource] in triton {
    indices = tl.arange(0, 4).to(tl.int64)
    in0 = tl.load((in0_ptr + indices) + indices)
  }

/-- error: finite source declarations must precede the writer body at top level -/
#guard_msgs in
def declarationsAfterRuntime : ComputeKernel := triton {
  pid = tl.program_id(0)
  tl.source_alias _isfinited = triton.language.extra.cuda.libdevice.isfinited
}

#axiomsClean single_program
#axiomsClean double_program
#axiomsClean single_candidate
#axiomsClean double_candidate
end VeriTile.Tests.ImportedFiniteJitFunctions
