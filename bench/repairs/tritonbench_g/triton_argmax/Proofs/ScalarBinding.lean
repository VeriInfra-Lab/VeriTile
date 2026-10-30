import bench.repairs.tritonbench_g.triton_argmax.Proofs.Formats

/-! Concrete source constants and ordered IEEE FP32 compare words. The safe
integer graph is instantiated with these constants before its final scalar
conversion callback. Floating store conversions remain format-specific. -/
namespace VeriTile.Bench.Repairs.TritonArgmax.ScalarBinding
open Source Formats
open VeriTile.Bench.TritonBenchG.MatrixReduction.MachineMax

def lowerBits : IntegerKind → Word
  | .i8 | .boolean => 0xc3000000
  | .i16 => 0xc7000000
  | .i32 => 0xcf000000
  | .i64 => 0xdf000000
  | _ => 0

def upperBits : IntegerKind → Word
  | .i8 | .boolean => 0x42fe0000
  | .i16 => 0x46fffe00
  | .i32 => 0x4effffff
  | .i64 => 0x5effffff
  | .u8 => 0x437f0000
  | .u16 => 0x477fff00
  | .u32 => 0x4f7fffff
  | .u64 => 0x5f7fffff

def gt32 (x y : Word) : Bool :=
  !isNaN .fp32 x && !isNaN .fp32 y && decide (rank .fp32 y < rank .fp32 x)

def eq32 (x y : Word) : Bool :=
  !isNaN .fp32 x && !isNaN .fp32 y && decide (rank .fp32 x=rank .fp32 y)

/-- The remaining hooks are exactly LLVM maxnum/minnum and a reached safe
FP32-to-integer truncation. These do not select reduction indices. -/
structure IntegerPrimitives (k : IntegerKind) where
  maximum32 : Word → Word → Word
  minimum32 : Word → Word → Word
  truncate : Word → BitVec k.bits

def storePrimitives (k : IntegerKind) (p : IntegerPrimitives k) : StorePrimitives k where
  eq32 := eq32
  zero32 := 0
  lower32 := lowerBits k
  upper32 := upperBits k
  maximum32 := p.maximum32
  minimum32 := p.minimum32
  truncate := p.truncate

def scalars (k : IntegerKind) (p : IntegerPrimitives k) : Scalars (format k) :=
  integerScalars k (storePrimitives k p) gt32 0xff800000

theorem no_nan_bounds (k : IntegerKind) :
    isNaN .fp32 (lowerBits k)=Bool.false ∧ isNaN .fp32 (upperBits k)=Bool.false := by
  cases k <;> decide

theorem sanitize_nan (k : IntegerKind) (p : IntegerPrimitives k) (x : Word)
    (nan : isNaN .fp32 x=Bool.true) :
    store k (storePrimitives k p) x=
      p.truncate (p.minimum32 (p.maximum32 0 (lowerBits k)) (upperBits k)) := by
  simp [store,storePrimitives,eq32,nan]

theorem stored_bool_byte (p : IntegerPrimitives .boolean) (x : Word) :
    decodeStored .boolean ((scalars .boolean p).storeCast x)=
      if (store .boolean (storePrimitives .boolean p) x).toNat=0 then 0 else 1 := by rfl

#axiomsClean no_nan_bounds
#axiomsClean sanitize_nan
#axiomsClean stored_bool_byte
end VeriTile.Bench.Repairs.TritonArgmax.ScalarBinding
