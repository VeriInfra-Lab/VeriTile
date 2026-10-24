import bench.repairs.tritonbench_g.triton_argmax.Proofs.ScalarBinding
import bench.repairs.tritonbench_g.triton_argmax.Proofs.Memory
import bench.repairs.tritonbench_g.triton_argmax.Proofs.Real

/-! Named local saturating conversion. Its native interpretation retains NaN0,
the exact FP32 bounds, maxnum/minnum, signed/unsigned truncation, storage width
and boolean byte decoding. Its idealized Real interpretation is identity by
policy; this is not literal Real evaluation of finite source clamps. A selected
source/emitted graph must refine this primitive before a whole-program shadow
may erase it. This module proves only local scalar and read/write facts. -/
namespace VeriTile.Bench.Repairs.TritonArgmax.ConversionMacro
open Source Formats Memory
open VeriTile.Bench.TritonBenchG.MatrixReduction.MachineMax

structure SaturatingConversion where
  kind : IntegerKind
  deriving DecidableEq,Repr

def ordered32 (a b : Word) : Bool := !isNaN .fp32 a && !isNaN .fp32 b

theorem self_equal_ordered_zero (x : Word) : ScalarBinding.eq32 x x=ordered32 x 0 := by
  have zero : isNaN .fp32 (0 : Word)=Bool.false := by decide
  unfold ScalarBinding.eq32 ordered32
  rw [zero]
  simp

/-- Exactly two reached primitive facts, at the pinned +0/lower/upper words.
They are not a cast, reduction, memory or output agreement assumption. -/
structure ZeroClamp (k : IntegerKind) (p : ScalarBinding.IntegerPrimitives k) : Prop where
  lower : p.maximum32 0 (ScalarBinding.lowerBits k)=0
  upper : p.minimum32 0 (ScalarBinding.upperBits k)=0

def SaturatingConversion.sourceStore (conversion : SaturatingConversion)
    (p : ScalarBinding.IntegerPrimitives conversion.kind) (x : Word) : BitVec conversion.kind.bits :=
  Formats.store conversion.kind (ScalarBinding.storePrimitives conversion.kind p) x

/-- The optimizer places the NaN-select after the two clamp instructions in
the pinned emitted LLVM. In particular NaN is tested on the original input. -/
def SaturatingConversion.emittedStore (conversion : SaturatingConversion)
    (p : ScalarBinding.IntegerPrimitives conversion.kind) (x : Word) : BitVec conversion.kind.bits :=
  let lower := p.maximum32 x (ScalarBinding.lowerBits conversion.kind)
  let upper := p.minimum32 lower (ScalarBinding.upperBits conversion.kind)
  p.truncate (if ordered32 x 0 then upper else 0)

theorem emitted_source (conversion : SaturatingConversion)
    (p : ScalarBinding.IntegerPrimitives conversion.kind) (zero : ZeroClamp conversion.kind p) (x : Word) :
    conversion.emittedStore p x=conversion.sourceStore p x := by
  unfold SaturatingConversion.emittedStore SaturatingConversion.sourceStore Formats.store
  simp only [ScalarBinding.storePrimitives,self_equal_ordered_zero]
  cases ordered32 x 0 <;>
    simp only [Bool.false_eq_true,if_false,if_true,zero.lower,zero.upper]

def SaturatingConversion.native (conversion : SaturatingConversion)
    (p : ScalarBinding.IntegerPrimitives conversion.kind) (x : Word) : Int :=
  decodeStored conversion.kind ((conversion.sourceStore p x).cast (format_width conversion.kind))

/-- This identity is the explicitly chosen idealized conversion-macro
projection, not an equation between a Real clamp and its input. -/
def SaturatingConversion.idealized (_ : SaturatingConversion) (x : Math.Value) : Math.Value := x

theorem native_boolean (p : ScalarBinding.IntegerPrimitives .boolean) (x : Word) :
    (SaturatingConversion.mk .boolean).native p x=
      if ((SaturatingConversion.mk .boolean).sourceStore p x).toNat=0 then 0 else 1 := rfl

theorem native_integer (conversion : SaturatingConversion) (notBool : conversion.kind≠.boolean)
    (p : ScalarBinding.IntegerPrimitives conversion.kind) (x : Word) :
    conversion.native p x=
      if conversion.kind.signed then (conversion.sourceStore p x).toInt else Int.ofNat (conversion.sourceStore p x).toNat := by
  cases conversion with
  | mk kind => cases kind <;> simp_all [SaturatingConversion.native,decodeStored,
      IntegerKind.signed,Formats.format,IntegerKind.bits]

/-- One actual store/load pair refines the named native conversion. This is
a single memory primitive rule and does not assume a complete kernel frame. -/
theorem store_load (conversion : SaturatingConversion) (p : ScalarBinding.IntegerPrimitives conversion.kind)
    (zero : ZeroClamp conversion.kind p) (mem : Primitives M) (s : M) (point : Address) (x : Word) :
    decodeStored conversion.kind (mem.read (Formats.format conversion.kind).bytes
      (mem.write (Formats.format conversion.kind).bytes s point
        ((conversion.emittedStore p x).cast (format_width conversion.kind))) point)=conversion.native p x := by
  rw [mem.readSame,emitted_source conversion p zero]
  rfl

#axiomsClean self_equal_ordered_zero
#axiomsClean emitted_source
#axiomsClean native_boolean
#axiomsClean native_integer
#axiomsClean store_load
end VeriTile.Bench.Repairs.TritonArgmax.ConversionMacro
