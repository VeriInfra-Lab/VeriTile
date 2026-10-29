import bench.tritonbench_g.isfinite_kernel.Proofs.BitClassification
import VeriTile.Triton.Float.ProfiledRewrite

/-! The full isfinite writer uses no floating arithmetic. These backends
therefore retain every unused operation of an arbitrary supplied algebra.
The only installed numerical behavior is bit-preserving, explicitly tagged
same-precision casts/loads and IEEE exponent-field finite classification.
An unsupported classifier precision fails. No Real projection is involved.
These are source-side semantics, not native libdevice refinement claims. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
open VeriTile.Triton FP.Structural FP.ProfiledRewrite BitClassification
noncomputable section

def backend32 (rest : Algebra Float32Bits) : Algebra Float32Bits :=
  { rest with
    cast := fun p src dst x => match p, src, dst with
      | some .fp32, .real, .real => x
      | _, _, _ => rest.cast p src dst x
    fp32Load := id
    isFinite := fun p => match p with
      | .fp32 => some finite32
      | .fp64 => none }

def backend64 (rest : Algebra Float64Bits) : Algebra Float64Bits :=
  { rest with
    cast := fun p src dst x => match p, src, dst with
      | some .fp64, .real, .real => x
      | _, _, _ => rest.cast p src dst x
    isFinite := fun p => match p with
      | .fp32 => none
      | .fp64 => some finite64 }

theorem cast32_preserves_bits (rest : Algebra Float32Bits) (x : Float32Bits) :
    (engine (backend32 rest)).cast (some .fp32) .real .real x = x := rfl

theorem cast64_preserves_bits (rest : Algebra Float64Bits) (x : Float64Bits) :
    (engine (backend64 rest)).cast (some .fp64) .real .real x = x := rfl

theorem load32_preserves_bits (rest : Algebra Float32Bits) (x : Float32Bits) :
    (engine (backend32 rest)).fp32Load x = x := rfl

theorem classification32 (rest : Algebra Float32Bits) :
    (engine (backend32 rest)).isFinite .fp32 = some finite32 := rfl

theorem classification64 (rest : Algebra Float64Bits) :
    (engine (backend64 rest)).isFinite .fp64 = some finite64 := rfl

theorem classifier32_rejects_fp64 (rest : Algebra Float32Bits) :
    (engine (backend32 rest)).isFinite .fp64 = none := rfl

theorem classifier64_rejects_fp32 (rest : Algebra Float64Bits) :
    (engine (backend64 rest)).isFinite .fp32 = none := rfl

/-- The complete sample includes both signs of zero, minimum subnormal, one,
maximum finite, infinity, quiet NaN, and signaling NaN. -/
def sample32 (n : Nat) : Float32Bits := bits32 (match n with
  | 0 => 0x00000000 | 1 => 0x80000000
  | 2 => 0x00000001 | 3 => 0x80000001
  | 4 => 0x3f800000 | 5 => 0xbf800000
  | 6 => 0x7f7fffff | 7 => 0xff7fffff
  | 8 => 0x7f800000 | 9 => 0xff800000
  | 10 => 0x7fc00001 | 11 => 0xffc00001
  | 12 => 0x7f800001 | _ => 0xff800001)

def sample64 (n : Nat) : Float64Bits := bits64 (match n with
  | 0 => 0x0000000000000000 | 1 => 0x8000000000000000
  | 2 => 0x0000000000000001 | 3 => 0x8000000000000001
  | 4 => 0x3ff0000000000000 | 5 => 0xbff0000000000000
  | 6 => 0x7fefffffffffffff | 7 => 0xffefffffffffffff
  | 8 => 0x7ff0000000000000 | 9 => 0xfff0000000000000
  | 10 => 0x7ff8000000000001 | 11 => 0xfff8000000000001
  | 12 => 0x7ff0000000000001 | _ => 0xfff0000000000001)

def expected : List Bool :=
  [true, true, true, true, true, true, true, true,
   false, false, false, false, false, false]

theorem samples32_classify :
    (List.range 14).map (fun i => finite32 (sample32 i)) = expected := by decide

theorem samples64_classify :
    (List.range 14).map (fun i => finite64 (sample64 i)) = expected := by decide

instance : Inhabited Float32Bits := ⟨bits32 0⟩
instance : Inhabited Float64Bits := ⟨bits64 0⟩

/-- Inactive loads receive a signaling NaN, so successful finite results
cannot be an artifact of zero-filled omitted padding. -/
def padding32 : (d : TileDType) → Value Float32Bits d
  | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 => bits32 0x7f800001
  | d => defaultValue d

def padding64 : (d : TileDType) → Value Float64Bits d
  | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 => bits64 0x7ff0000000000001
  | d => defaultValue d

def initial32 (values : Nat → Float32Bits := sample32) : State Float32Bits where
  mem r i := if r = "input" then .mk .real (values i) else .mk .nat 1729
  regs _ _ _ := none
  pids _ := 0
  numPids _ := 1
  undef d _ _ := padding32 d

def initial64 (values : Nat → Float64Bits := sample64) : State Float64Bits where
  mem r i := if r = "input" then .mk .real (values i) else .mk .nat 1729
  regs _ _ _ := none
  pids _ := 0
  numPids _ := 1
  undef d _ _ := padding64 d

/-- Preserve the written cell's Bool tag in observations. -/
def boolCell {α : Type} : Cell α → Option Bool
  | .mk .bool b => some b
  | _ => none

def natCell {α : Type} : Cell α → Option Nat
  | .mk .nat n => some n
  | _ => none

def outputValues {α : Type} (s : State α) : List (Option Bool) :=
  (List.range 14).map (fun i => boolCell (s.mem "output" (2 * i)))

/-- All fourteen odd-address holes, the two masked tail lanes, and a foreign
region are observed with their original Nat tag and value. -/
def sentinelValues {α : Type} (s : State α) : List (Option Nat) :=
  (List.range 14).map (fun i => natCell (s.mem "output" (2 * i + 1))) ++
  [natCell (s.mem "output" 28), natCell (s.mem "output" 30),
   natCell (s.mem "unrelated" 0)]

#axiomsClean cast32_preserves_bits
#axiomsClean cast64_preserves_bits
#axiomsClean load32_preserves_bits
#axiomsClean classification32
#axiomsClean classification64
#axiomsClean classifier32_rejects_fp64
#axiomsClean classifier64_rejects_fp32
#axiomsClean samples32_classify
#axiomsClean samples64_classify
end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
