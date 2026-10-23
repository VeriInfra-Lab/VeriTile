import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.PtxSigned
import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.KernelExecution
import bench.tritonbench_g.quantize_kv_transform.Proofs.MixedBits

namespace VeriTile.Bench.Repairs.QuantizeCopyKv.KernelNativeProfile
open VeriTile.Triton FP.Structural
abbrev Word := FP.Value .fp32
abbrev Half := VeriTile.Bench.TritonBenchG.QuantizeKvTransform.MixedBits.Half
open VeriTile.Bench.TritonBenchG.QuantizeKvTransform
noncomputable section

/-- Native arithmetic remains explicit, including complete reduction shape,
axis, row position and order. No IEEE-division replacement or reduction
permutation/associativity law is assumed. The half callbacks keep native
rounding and NaN handling; half storage itself has a proved exact embedding. -/
structure Arithmetic where
  absolute : Word → Word
  divFull : Word → Word → Word
  equal : Word → Word → Bool
  roundHalf : Word → Half
  promoteHalf : Half → Word
  reduceMax : Option ComputeDType → {shape : TileShape} →
    (ax : Fin shape.length) → (keepDims : Bool) → Values Word .real shape →
    Values Word .real (TileShape.reduceShape shape ax keepDims)

def zero : Word := FP.Value.ofBits .fp32 0
def one : Word := FP.Value.ofBits .fp32 0x3f800000
def levels : Word := FP.Value.ofBits .fp32 0x42fe0000

def algebra (a : Arithmetic) (rest : Algebra Word) : Algebra Word :=
  { rest with
    literal := fun p d x => if FP.Software.fp32Context p d then
      if x = 0 then zero else if x = 1 then one else if x = 127 then levels
      else rest.literal p d x
      else rest.literal p d x
    unary := fun p op x => match op with
      | .abs d => if FP.Software.fp32Context p d then a.absolute x else rest.unary p op x
      | _ => rest.unary p op x
    binary := fun p d op x y => match op with
      | .div => if FP.Software.fp32Context p d then a.divFull x y else rest.binary p d op x y
      | _ => rest.binary p d op x y
    cast := fun p src dst x => match src, dst with
      | .real, .fp16 | .fp32, .fp16 => MixedBits.embed (a.roundHalf x)
      | .fp16, .real | .fp16, .fp32 => a.promoteHalf (MixedBits.half x)
      | .fp16, .fp16 => MixedBits.embed (MixedBits.half x)
      | _, _ => rest.cast p src dst x
    fp32Load := id
    reduceMax := a.reduceMax }

def backend (a : Arithmetic) (rest : Algebra Word) : Algebra Word :=
  KernelExecution.backend (algebra a rest) PtxSigned.signedByte a.equal

theorem source_literals : zero.toRat? = some 0 ∧ one.toRat? = some 1 ∧
    levels.toRat? = some 127 := by decide +kernel

theorem division_callback (a : Arithmetic) (rest : Algebra Word) (x y : Word) :
    (backend a rest).binary (some .fp32) .real .div x y = a.divFull x y := by
  simp [backend, KernelExecution.backend, algebra, FP.Software.fp32Context]

theorem half_roundtrip (a : Arithmetic) (x : Word) :
    MixedBits.half (MixedBits.embed (a.roundHalf x)) = a.roundHalf x :=
  MixedBits.half_embedding _

theorem conversion_callback (a : Arithmetic) (rest : Algebra Word) :
    (backend a rest).floatToInt (some .fp32) .real 8 =
      some (fun x : Word => some (PtxSigned.signedByte x)) := by
  simp [backend, KernelExecution.backend]

#axiomsClean source_literals
#axiomsClean division_callback
#axiomsClean half_roundtrip
#axiomsClean conversion_callback
end
end VeriTile.Bench.Repairs.QuantizeCopyKv.KernelNativeProfile
