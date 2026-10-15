import bench.tritonbench_g.vector_addition.Proofs.HostWrapper
import bench.tritonbench_g.vector_addition.Proofs.DTypeAdmission
import bench.tritonbench_g.vector_addition.IntegerWordGridCorrect
import bench.tritonbench_g.vector_addition.TypedGridCorrect
import bench.tritonbench_g.vector_addition.MixedFP32ValueCorrect
import bench.tritonbench_g.vector_addition.MixedSmallValueCorrect
import bench.tritonbench_g.vector_addition.FP64GridCorrect

/-! Exact finite dispatch to existing source execution families. No family is
reimplemented, and the successful dtype count is not a value-totality claim. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.TypedHost
open VeriTile.Triton DTypeScope
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000

inductive Route where
  | integer (x y : IntegerSource.Canonical)
  | floating (x y : TypedFloat.Storage)
  | mixed32 (integerFirst : Bool) (d : IntegerSource.Canonical)
  | mixedSmall (f : MixedSmall.FloatStorage) (integerFirst : Bool) (d : IntegerSource.Canonical)
  | wide (x y : FP64TypedCore.Storage)
  deriving DecidableEq, Repr

def integerStorage : DType → Option IntegerSource.Canonical
  | .b1 => some .b1 | .i8 => some .i8 | .u8 => some .u8
  | .i16 => some .i16 | .u16 => some .u16 | .i32 => some .i32 | .u32 => some .u32
  | .i64 => some .i64 | .u64 => some .u64 | _ => none

def floatStorage : DType → Option TypedFloat.Storage
  | .fp16 => some .fp16 | .bf16 => some .bf16 | .fp32 => some .fp32
  | .fp8e4nv => some .f8e4 | .fp8e5 => some .f8e5 | _ => none

def wideStorage : DType → Option FP64TypedCore.Storage
  | .b1 => some .boolean
  | .i8 => some (.signed .w8) | .u8 => some (.unsigned .w8)
  | .i16 => some (.signed .w16) | .u16 => some (.unsigned .w16)
  | .i32 => some (.signed .w32) | .u32 => some (.unsigned .w32)
  | .i64 => some (.signed .w64) | .u64 => some (.unsigned .w64)
  | .fp16 => some .fp16 | .bf16 => some .bf16 | .fp32 => some .fp32 | .fp64 => some .fp64
  | .fp8e4nv => some .f8e4fn | .fp8e5 => some .f8e5 | _ => none

def wideDType : FP64TypedCore.Storage → DType
  | .boolean => .b1
  | .signed .w8 => .i8 | .unsigned .w8 => .u8
  | .signed .w16 => .i16 | .unsigned .w16 => .u16
  | .signed .w32 => .i32 | .unsigned .w32 => .u32
  | .signed .w64 => .i64 | .unsigned .w64 => .u64
  | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32 | .fp64 => .fp64
  | .f8e4fn => .fp8e4nv | .f8e5 => .fp8e5

def mixedRoute (first : Bool) (d : IntegerSource.Canonical) : TypedFloat.Storage → Option Route
  | .fp32 => some (.mixed32 first d)
  | .fp16 => some (.mixedSmall .fp16 first d)
  | .bf16 => some (.mixedSmall .bf16 first d)
  | _ => none

/-- Admission is evaluated before selection, preserving signature rejection
precedence even for the 28 rejected fnuz pairs whose standalone promotion works. -/
def route (x y : DType) : Option Route :=
  if (DTypeAdmission.admission x y).stage = .admitted then
    if x = .fp64 ∨ y = .fp64 then do
      let a ← wideStorage x
      let b ← wideStorage y
      pure (.wide a b)
    else match integerStorage x, integerStorage y, floatStorage x, floatStorage y with
      | some a, some b, _, _ => some (.integer a b)
      | _, _, some a, some b => some (.floating a b)
      | some a, _, _, some b => mixedRoute Bool.true a b
      | _, some b, some a, _ => mixedRoute Bool.false b a
      | _, _, _, _ => none
  else none

def Route.signature : Route → DType × DType
  | .integer x y => (x.dtype, y.dtype)
  | .floating x y => (embed x, embed y)
  | .mixed32 first d => if first then (d.dtype, .fp32) else (.fp32, d.dtype)
  | .mixedSmall f first d => if first then (d.dtype, f.canonical) else (f.canonical, d.dtype)
  | .wide x y => (wideDType x, wideDType y)

def Route.family : Route → Nat
  | .integer _ _ => 0 | .floating _ _ => 1 | .mixed32 _ _ => 2
  | .mixedSmall _ _ _ => 3 | .wide _ _ => 4

theorem route_matches_admission (x y : DType) :
    (route x y).isSome = decide ((DTypeAdmission.admission x y).stage = .admitted) := by
  cases x <;> cases y <;> decide +kernel

theorem route_signature_checked (x y : DType) :
    (route x y).all (fun r => decide (r.signature = (x,y))) = Bool.true := by
  cases x <;> cases y <;> decide +kernel

theorem route_signature (x y : DType) (r : Route) (selected : route x y = some r) :
    r.signature = (x,y) := by
  have h := route_signature_checked x y
  simpa [selected] using h

theorem wide_route_supported (x y : DType) :
    (route x y).all (fun r => match r with
      | .wide a b => decide (FP64TypedCore.Supported a b)
      | _ => Bool.true) = Bool.true := by
  cases x <;> cases y <;> decide +kernel

def routes : List Route := DTypeAdmission.pairs.filterMap (fun p => route p.1 p.2)

theorem route_inventory :
    routes.length = 189 ∧
    (routes.filter (fun r => r.family == 0)).length = 81 ∧
    (routes.filter (fun r => r.family == 1)).length = 25 ∧
    (routes.filter (fun r => r.family == 2)).length = 18 ∧
    (routes.filter (fun r => r.family == 3)).length = 36 ∧
    (routes.filter (fun r => r.family == 4)).length = 29 := by decide +kernel

def dtypeName : DType → String
  | .b1 => "bool" | .i8 => "int8" | .u8 => "uint8"
  | .i16 => "int16" | .u16 => "uint16" | .i32 => "int32" | .u32 => "uint32"
  | .i64 => "int64" | .u64 => "uint64" | .fp16 => "float16" | .bf16 => "bfloat16"
  | .fp32 => "float32" | .fp64 => "float64" | .fp8e5 => "float8_e5m2"
  | .fp8e4nv => "float8_e4m3fn" | .fp8e5b16 => "float8_e5m2fnuz" | .fp8e4b8 => "float8_e4m3fnuz"

def parseDType : String → Option DType
  | "bool" => some .b1 | "int8" => some .i8 | "uint8" => some .u8
  | "int16" => some .i16 | "uint16" => some .u16 | "int32" => some .i32 | "uint32" => some .u32
  | "int64" => some .i64 | "uint64" => some .u64 | "float16" => some .fp16 | "bfloat16" => some .bf16
  | "float32" => some .fp32 | "float64" => some .fp64 | "float8_e5m2" => some .fp8e5
  | "float8_e4m3fn" => some .fp8e4nv | "float8_e5m2fnuz" => some .fp8e5b16
  | "float8_e4m3fnuz" => some .fp8e4b8 | _ => none

theorem parse_dtype (d : DType) : parseDType (dtypeName d) = some d := by cases d <;> rfl

#axiomsClean route_matches_admission
#axiomsClean route_signature_checked
#axiomsClean route_signature
#axiomsClean wide_route_supported
#axiomsClean route_inventory
#axiomsClean parse_dtype
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.VectorAddition.TypedHost
