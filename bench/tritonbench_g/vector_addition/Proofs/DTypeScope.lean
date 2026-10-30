import bench.tritonbench_g.vector_addition.Proofs.TypedFloat

/-! Finite dispatch audit for the 17 canonical scalar signatures recognized
by the inspected Torch/Triton environment. `observedMatrix` is frozen probe
evidence, not a proved semantics of Python or the compiler. The kernel checks
that the explicit rule below reproduces every one of its 289 entries and that
the five typed-source formats use the same promotion. Native support/errors,
integer arithmetic/casts and physical storage remain separate obligations.
-/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.DTypeScope
open VeriTile.Triton
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000

inductive DType where
  | b1 | i8 | u8 | i16 | u16 | i32 | u32 | i64 | u64
  | fp16 | bf16 | fp32 | fp64 | fp8e5 | fp8e4nv | fp8e5b16 | fp8e4b8
  deriving DecidableEq, Repr

def DType.width : DType → Nat
  | .b1 => 1 | .i8 | .u8 => 8 | .i16 | .u16 => 16
  | .i32 | .u32 => 32 | .i64 | .u64 => 64 | _ => 0

def DType.signed : DType → Bool
  | .i8 | .i16 | .i32 | .i64 => Bool.true | _ => Bool.false

def DType.fp8 : DType → Bool
  | .fp8e5 | .fp8e4nv | .fp8e5b16 | .fp8e4b8 => Bool.true | _ => Bool.false

/-- Promotion of tensor operands, including the frontend fp8/integer error.
No inference that a promoted pair has successful native lowering is made. -/
def promote (x y : DType) : Option DType :=
  if x = .fp64 ∨ y = .fp64 then some .fp64
  else if x = .fp32 ∨ y = .fp32 then some .fp32
  else if x = .fp16 ∨ y = .fp16 then some .fp16
  else if x = .bf16 ∧ y = .bf16 then some .bf16
  else if x = .bf16 ∨ y = .bf16 then some .fp32
  else if x.fp8 ∨ y.fp8 then
    if x.fp8 ∧ y.fp8 then some (if x = y then x else .fp16) else none
  else if x.signed = y.signed then some (if y.width ≤ x.width then x else y)
  else if x.signed then some (if y.width < x.width then x else y)
  else some (if x.width < y.width then y else x)

/-- Native tl.store reinterprets a bool pointer as int8 before conversion. -/
def storeElement (x : DType) : DType := if x = .b1 then .i8 else x

def profiles : List DType :=
  [.b1, .i8, .u8, .i16, .u16, .i32, .u32, .i64, .u64, .fp16, .bf16, .fp32, .fp64, .fp8e5, .fp8e4nv, .fp8e5b16, .fp8e4b8]

/-- Captured from the 2026-10-15 offline semantic probe, in profiles order.
Source matrix SHA256: 6049ee18bc252e0b8aab035fde83c219be9da7735b58a2779a0cccf4e98bc854. -/
def observedMatrix : List (List (Option DType)) := [
  [some .b1, some .i8, some .u8, some .i16, some .u16, some .i32, some .u32, some .i64, some .u64, some .fp16, some .fp32, some .fp32, some .fp64, none, none, none, none],
  [some .i8, some .i8, some .u8, some .i16, some .u16, some .i32, some .u32, some .i64, some .u64, some .fp16, some .fp32, some .fp32, some .fp64, none, none, none, none],
  [some .u8, some .u8, some .u8, some .i16, some .u16, some .i32, some .u32, some .i64, some .u64, some .fp16, some .fp32, some .fp32, some .fp64, none, none, none, none],
  [some .i16, some .i16, some .i16, some .i16, some .u16, some .i32, some .u32, some .i64, some .u64, some .fp16, some .fp32, some .fp32, some .fp64, none, none, none, none],
  [some .u16, some .u16, some .u16, some .u16, some .u16, some .i32, some .u32, some .i64, some .u64, some .fp16, some .fp32, some .fp32, some .fp64, none, none, none, none],
  [some .i32, some .i32, some .i32, some .i32, some .i32, some .i32, some .u32, some .i64, some .u64, some .fp16, some .fp32, some .fp32, some .fp64, none, none, none, none],
  [some .u32, some .u32, some .u32, some .u32, some .u32, some .u32, some .u32, some .i64, some .u64, some .fp16, some .fp32, some .fp32, some .fp64, none, none, none, none],
  [some .i64, some .i64, some .i64, some .i64, some .i64, some .i64, some .i64, some .i64, some .u64, some .fp16, some .fp32, some .fp32, some .fp64, none, none, none, none],
  [some .u64, some .u64, some .u64, some .u64, some .u64, some .u64, some .u64, some .u64, some .u64, some .fp16, some .fp32, some .fp32, some .fp64, none, none, none, none],
  [some .fp16, some .fp16, some .fp16, some .fp16, some .fp16, some .fp16, some .fp16, some .fp16, some .fp16, some .fp16, some .fp16, some .fp32, some .fp64, some .fp16, some .fp16, some .fp16, some .fp16],
  [some .fp32, some .fp32, some .fp32, some .fp32, some .fp32, some .fp32, some .fp32, some .fp32, some .fp32, some .fp16, some .bf16, some .fp32, some .fp64, some .fp32, some .fp32, some .fp32, some .fp32],
  [some .fp32, some .fp32, some .fp32, some .fp32, some .fp32, some .fp32, some .fp32, some .fp32, some .fp32, some .fp32, some .fp32, some .fp32, some .fp64, some .fp32, some .fp32, some .fp32, some .fp32],
  [some .fp64, some .fp64, some .fp64, some .fp64, some .fp64, some .fp64, some .fp64, some .fp64, some .fp64, some .fp64, some .fp64, some .fp64, some .fp64, some .fp64, some .fp64, some .fp64, some .fp64],
  [none, none, none, none, none, none, none, none, none, some .fp16, some .fp32, some .fp32, some .fp64, some .fp8e5, some .fp16, some .fp16, some .fp16],
  [none, none, none, none, none, none, none, none, none, some .fp16, some .fp32, some .fp32, some .fp64, some .fp16, some .fp8e4nv, some .fp16, some .fp16],
  [none, none, none, none, none, none, none, none, none, some .fp16, some .fp32, some .fp32, some .fp64, some .fp16, some .fp16, some .fp8e5b16, some .fp16],
  [none, none, none, none, none, none, none, none, none, some .fp16, some .fp32, some .fp32, some .fp64, some .fp16, some .fp16, some .fp16, some .fp8e4b8]]

theorem promotion_matches_all_289_observations :
    profiles.map (fun x => profiles.map (fun y => promote x y)) = observedMatrix := by decide

theorem observed_success_count :
    (observedMatrix.flatten.filter Option.isSome).length = 217 := by decide

theorem observed_rejection_count :
    (observedMatrix.flatten.filter Option.isNone).length = 72 := by decide

def embed : TypedFloat.Storage → DType
  | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32
  | .f8e4 => .fp8e4nv | .f8e5 => .fp8e5

theorem typed_promotion_agrees (x y : TypedFloat.Storage) :
    promote (embed x) (embed y) = some (embed (TypedFloat.promote x y)) := by
  cases x <;> cases y <;> decide

#axiomsClean promotion_matches_all_289_observations
#axiomsClean observed_success_count
#axiomsClean observed_rejection_count
#axiomsClean typed_promotion_agrees
end VeriTile.Bench.TritonBenchG.VectorAddition.DTypeScope
