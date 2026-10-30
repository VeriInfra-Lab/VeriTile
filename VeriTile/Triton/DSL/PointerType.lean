import VeriTile.Triton.DSL.SourceDType

/-! Invocation pointer storage. A floating family refers to the caller's
existing FloatDType channel, retaining all of its cases rather than fixing
FP32. This descriptor emits metadata, never a source instruction. -/
namespace VeriTile.Triton
inductive PointerType where
  | fp16 | bf16 | fp32 | fp64 | f8e4 | f8e5
  | i8 | i16 | i32 | i64 | u8 | u16 | u32 | u64 | boolean
  | inputFloat | outputFloat
  deriving DecidableEq, Repr

namespace DSL.PointerType
open Lean

def dtypeName : Triton.PointerType → Name
  | .fp16 => `tl.float16 | .bf16 => `tl.bfloat16 | .fp32 => `tl.float32 | .fp64 => `tl.float64
  | .f8e4 => `tl.float8e4nv | .f8e5 => `tl.float8e5
  | .i8 => `tl.int8 | .i16 => `tl.int16 | .i32 => `tl.int32 | .i64 => `tl.int64
  | .u8 => `tl.uint8 | .u16 => `tl.uint16 | .u32 => `tl.uint32 | .u64 => `tl.uint64
  | .boolean => `tl.int1
  | .inputFloat => `DTYPE | .outputFloat => `OUT_DTYPE

end DSL.PointerType
end VeriTile.Triton
