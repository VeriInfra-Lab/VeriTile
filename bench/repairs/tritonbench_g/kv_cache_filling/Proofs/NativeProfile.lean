import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.PtxCast
import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.Source

namespace VeriTile.Bench.Repairs.KvCacheFilling.NativeProfile
open VeriTile.Triton
abbrev Word := FP.Value .fp32
abbrev Byte := BitVec 8

/-! The native proof keeps each emitted floating primitive as its own total
callback over FP32 words. In particular `div.full.f32` is **not** `FP.div
.ieeeValue`: PTX specifies a full-range approximation, not exact IEEE rounding.
The native minnum/maxnum callbacks retain their selected NaN/tie behavior, and
add/sub callbacks retain native NaN payload behavior. A structural equivalence
using the same callback on the same operands needs no arithmetic identity or
empirical primitive-conformance assumption.

Only literals, FP32 bit transport, and the documented native uint8 conversion
are fixed here. The source's integer-offset candidate changes none of the
floating primitive calls. This profile is a semantic interface to the selected
native lowering, not an experiment or a claim about unspecified LLVM poison.
PTX references and exact emitted artifacts are listed in README.md. -/

structure Arithmetic where
  (add sub divFull minNum maxNum : Word → Word → Word)
  equal : Word → Word → Bool

def literal (n : Nat) : Word := FP.round .fp32 .ieeeValue (Rat.ofInt n)
def half : Word := FP.Value.ofBits .fp32 0x3f000000

def operations (a : Arithmetic) : Operations Word Byte where
  minimum := a.minNum
  maximum := a.maxNum
  sub := a.sub
  div := a.divFull
  add := a.add
  neg := fun x => a.sub (literal 0) x
  equal := a.equal
  literal := literal
  half := half
  uint8 := PtxCast.byte
  pack := fun x y => x + y * 16

/-- Every literal used by this source is exactly representable in FP32. -/
theorem source_literals :
    literal 0 = FP.Value.ofBits .fp32 0 ∧
    literal 1 = FP.Value.ofBits .fp32 0x3f800000 ∧
    literal 15 = FP.Value.ofBits .fp32 0x41700000 ∧
    literal 255 = FP.Value.ofBits .fp32 0x437f0000 := by
  decide +kernel

/-- Ordered native scalar formula, independent of registers and memory. -/
def scale (a : Arithmetic) (levels : Nat) (lo hi : Word) : Word :=
  if a.equal hi lo then literal 1 else a.divFull (a.sub hi lo) (literal levels)

def zero (a : Arithmetic) (levels : Nat) (lo hi : Word) : Word :=
  a.divFull (a.sub (literal 0) lo) (scale a levels lo hi)

def code (a : Arithmetic) (levels : Nat) (lo hi x : Word) : Byte :=
  PtxCast.byte (a.add (a.add (a.divFull x (scale a levels lo hi)) (zero a levels lo hi)) half)

/-- Existing ordered scalar-source execution is reused with the actual
native operation profile; no old IEEE-equality or defined-cast premise occurs. -/
theorem quant_source (a : Arithmetic) (levels : Nat) (x : Nat → Word) (lo hi : Word) :
    sourceQuant (operations a) levels x lo hi =
      ⟨fun i => code a levels lo hi (x i), scale a levels lo hi, zero a levels lo hi⟩ := by
  simp [sourceQuant, runQuantSource, quantSource, QExpr.eval, Function.update,
    operations, code, scale, zero]

noncomputable def scalarBackend (a : Arithmetic) (rest : FP.Structural.Algebra Word) :
    FP.Structural.Algebra Word :=
  { PtxCast.backend rest with
    literal := fun p d x =>
      if FP.Software.fp32Context p d then
        if x = 0 then literal 0
        else if x = 1 then literal 1
        else if x = 15 then literal 15
        else if x = 255 then literal 255
        else if x = (1 / 2 : ℝ) then half
        else rest.literal p d x
      else rest.literal p d x
    binary := fun p d op x y =>
      if FP.Software.fp32Context p d then
        match op with
        | .add => a.add x y
        | .sub => a.sub x y
        | .div => a.divFull x y
        | .max => a.maxNum x y
        | _ => rest.binary p d op x y
      else rest.binary p d op x y
    unary := fun p op x => match op with
      | .neg d => if FP.Software.fp32Context p d then a.sub (literal 0) x else rest.unary p op x
      | _ => rest.unary p op x
    compareEq := fun p d =>
      if FP.Software.fp32Context p d then some a.equal else rest.compareEq p d
    cast := fun p src dst x =>
      if (src = .fp32 ∧ dst = .fp32) ∨
          (p = some .fp32 ∧ (src = .real ∨ src = .fp32) ∧ (dst = .real ∨ dst = .fp32)) then x
      else rest.cast p src dst x
    fp32Load := id }

theorem scalarBackend_div (a : Arithmetic) (rest : FP.Structural.Algebra Word)
    (x y : Word) :
    (scalarBackend a rest).binary (some .fp32) .real .div x y = a.divFull x y := by
  simp [scalarBackend, FP.Software.fp32Context]

theorem scalarBackend_cast (a : Arithmetic) (rest : FP.Structural.Algebra Word) :
    (scalarBackend a rest).floatToNat (some .fp32) .real 8 =
      some (fun x : Word => some (PtxCast.byte x).toNat) := by
  simp [scalarBackend, PtxCast.backend, FP.IntegerConversion.fp32Source]

#axiomsClean source_literals
#axiomsClean quant_source
#axiomsClean scalarBackend_div
#axiomsClean scalarBackend_cast
#specNonCircular code avoiding [sourceQuant, runQuantSource]
end VeriTile.Bench.Repairs.KvCacheFilling.NativeProfile
