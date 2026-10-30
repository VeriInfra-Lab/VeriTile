import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.KernelNativeModel
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.Mathematics
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelScalarProfile
open VeriTile.Triton FP.Structural
noncomputable section
abbrev Tree := RowwiseQuantizationTriton.Tree

/-- Explicit structural max tree. Selection sees only launch configuration,
never payloads. Its leaves address the shared interpreter's actual axis. -/
def reduceTree (zero : α) (maximum : α → α → α) (tree : Tree) {shape : TileShape}
    (ax : Fin shape.length) : (keepDims : Bool) → Values α .real shape →
      Values α .real (TileShape.reduceShape shape ax keepDims)
  | Bool.false, values => fun out => tree.evaluate maximum (fun i =>
      if hi : i < TileShape.axisDim shape ax then
        values (TileShape.insertAxisIndex shape ax out ⟨i, hi⟩) else zero)
  | Bool.true, values => fun out => tree.evaluate maximum (fun i =>
      if hi : i < TileShape.axisDim shape ax then
        values (TileShape.replaceAxisIndex shape ax out ⟨i, hi⟩) else zero)

/-- The dtype tag distinguishes a packed binary16 value from FP32. No
floating conversion is performed merely to hold a half in structural memory. -/
def pack16 (x : Word16) : Word32 := ⟨BitVec.ofNat 32 x.bits.toNat⟩
def unpack16 (x : Word32) : Word16 := ⟨BitVec.ofNat 16 x.bits.toNat⟩

theorem half_roundtrip (x : Word16) : unpack16 (pack16 x) = x := by
  cases x with
  | mk bits =>
    unfold unpack16 pack16
    congr 1
    apply BitVec.eq_of_toNat_eq
    change bits.toNat % (2^32) % (2^16) = bits.toNat
    have hb := bits.isLt
    have h32 : bits.toNat < 2^32 := by omega
    rw [Nat.mod_eq_of_lt h32, Nat.mod_eq_of_lt hb]

/-- Native scalar profile with explicit packed-half storage. Division,
maximum, and atomic add retain the selected instruction policy. Multiplication,
FP16 narrowing, signed32 conversion, and signed16-to-low8 are concrete basic
semantics; no whole-row or output primitive is introduced. -/
def native (f : KernelNativeModel.FloatingInstructions) (tree : Tree)
    (rest : Algebra Word32) : Algebra Word32 :=
  { rest with
    literal := fun p d x => if x = 0 then FP.Value.zero .fp32
      else if x = 127 then FP.Value.ofBits .fp32 0x42fe0000 else rest.literal p d x
    binary := fun p d op x y => match op with
      | .mul => FP.mul .ieeeValue x y
      | .div => f.divide x y
      | .max => f.maximum x y
      | _ => rest.binary p d op x y
    unary := fun p op x => match op with
      | .abs _ => x.abs
      | _ => rest.unary p op x
    fromInt := fun _ z => FP.round .fp32 .ieeeValue (z : Rat)
    cast := fun p src dst x => match src, dst with
      | .real, .fp16 => pack16 (Int8DequantMatmul.narrow16 x)
      | .fp16, .real => KernelNativeModel.widen (unpack16 x)
      | .real, .real | .fp16, .fp16 => x
      | _, _ => rest.cast p src dst x
    fp32Load := id
    reduceMax := fun _ => reduceTree (FP.Value.zero .fp32) f.maximum tree
    atomicAdd := fun d => match d with
      | .real | .fp32 => some f.atomic32
      | .fp16 => some (fun x y => pack16 (f.atomic16 (unpack16 x) (unpack16 y)))
      | _ => rest.atomicAdd d
    floatToInt := fun pr d width => if width = 8 then
      some (fun x => some (KernelNativeModel.integer8 x).toInt)
      else rest.floatToInt pr d width }

/-- Mathematical interpretation of the source's scalar operations. Real
division is total; floating storage casts do not assert native exactness. -/
def mathematical (tree : Tree) (rest : Algebra ℝ) : Algebra ℝ :=
  { rest with
    literal := fun _ _ x => x
    binary := fun _ _ op x y => match op with
      | .add => x+y
      | .sub => x-y
      | .mul => x*y
      | .div => x/y
      | .max => max x y
      | .pow => Real.rpow x y
    unary := fun p op x => match op with
      | .abs _ => |x|
      | _ => rest.unary p op x
    fromInt := fun _ z => (z : ℝ)
    cast := fun _ _ _ x => x
    fp32Load := id
    reduceMax := fun _ => reduceTree 0 max tree
    atomicAdd := fun _ => some (· + ·) }

#axiomsClean half_roundtrip
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelScalarProfile
