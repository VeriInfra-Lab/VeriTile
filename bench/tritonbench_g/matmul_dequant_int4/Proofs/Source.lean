import bench.tritonbench_g.matmul_dequant_int4.Sources.Mathematical_FPContract
import bench.tritonbench_g.kcache_copy_triton.Sources.Mathematical_TypedCopy
import VeriTile.Triton.Float.FixedInteger
import VeriTile.Triton.Float.ScalarOps
namespace VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
attribute [local simp 10000] FP.Structural.eval_intShiftRight FP.Structural.eval_intBitAnd
structure Parameters where
  (weight scale zero output : RegionName)
  (rows columns groupSize weightRowStride weightColumnStride scaleRowStride scaleColumnStride
    zeroRowStride zeroColumnStride outputRowStride outputColumnStride blockRows blockColumns : Nat)

/-- Exactly the unchanged readable source quotation, parameterized by the two
admitted floating storage profiles. Packed channels remain signed INT32. -/
inductive Profile where | half | single deriving DecidableEq, Repr

def Profile.dtype : Profile → FloatDType
  | .half => .fp16 | .single => .fp32

def source (f : Profile) (p : Parameters) : ComputeKernel :=
  VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.typed f.dtype
    p.weight p.scale p.zero p.output p.rows p.columns p.groupSize
    p.weightRowStride p.weightColumnStride p.scaleRowStride p.scaleColumnStride
    p.zeroRowStride p.zeroColumnStride p.outputRowStride p.outputColumnStride p.blockRows p.blockColumns

abbrev Index (p : Parameters) := TileIndex [p.blockRows, p.blockColumns]
def row {α : Type} (p : Parameters) (s : State α) (i : Index p) : Nat := s.pids 0 * p.blockRows + i.1.val
def column {α : Type} (p : Parameters) (s : State α) (i : Index p) : Nat := s.pids 1 * p.blockColumns + i.2.1.val
def weightOffset {α : Type} (p : Parameters) (s : State α) (i : Index p) : Nat :=
  row p s i / 8 * p.weightRowStride + column p s i * p.weightColumnStride
def zeroOffset {α : Type} (p : Parameters) (s : State α) (i : Index p) : Nat :=
  row p s i / p.groupSize * p.zeroRowStride + column p s i / 8 * p.zeroColumnStride
def scaleOffset {α : Type} (p : Parameters) (s : State α) (i : Index p) : Nat :=
  row p s i / p.groupSize * p.scaleRowStride + column p s i * p.scaleColumnStride
def outputOffset {α : Type} (p : Parameters) (s : State α) (i : Index p) : Nat :=
  row p s i * p.outputRowStride + column p s i * p.outputColumnStride

def active {α : Type} (p : Parameters) (s : State α) (i : Index p) : Prop :=
  row p s i < p.rows ∧ column p s i < p.columns
instance {α : Type} (p : Parameters) (s : State α) : DecidablePred (active p s) :=
  fun _ => inferInstanceAs (Decidable (_ ∧ _))

/-- Arithmetic signed right shift followed by a real 32-bit mask. -/
def nibble (word : Int) (slot : Nat) : Int :=
  ((BitVec.ofInt 32 (word >>> (slot % 8 * 4))) &&& (15 : BitVec 32)).toInt

def difference {α : Type} [Inhabited α] (p : Parameters) (s : State α) (i : Index p) : Int :=
  nibble ((s.mem p.weight (weightOffset p s i)).read .int) (row p s i) -
    nibble ((s.mem p.zero (zeroOffset p s i)).read .int) (column p s i)

def value {α : Type} [Inhabited α] (f : Profile) (M : Algebra α) (p : Parameters) (s : State α) (i : Index p) : α :=
  M.binary none f.dtype .mul
    (M.cast (some .fp32) .real f.dtype (M.fromInt (some .fp32) (difference p s i)))
    (match f with
      | .half => ((s.mem p.scale (scaleOffset p s i)).read .fp16)
      | .single => ((s.mem p.scale (scaleOffset p s i)).read .fp32))

def cell {α : Type} (f : Profile) (x : α) : Cell α :=
  match f with
  | .half => .mk .fp16 x
  | .single => .mk .fp32 x

def written {α : Type} [Inhabited α] (f : Profile) (M : Algebra α) (p : Parameters) (s : State α) : State α :=
  (TileShape.allIndices [p.blockRows, p.blockColumns]).foldl (fun t i => if active p s i then
    t.write p.output (outputOffset p s i) (cell f (value f M p s i)) else t) s

def execution {α : Type} [Inhabited α] (f : Profile) (M : Algebra α) (p : Parameters) (s : State α) :
    {t : State α // FP.Structural.exec M (source f p) s = some t} := by
  cases f <;> (refine ⟨?_, ?_⟩; swap; simp [source, Profile.dtype, VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.typed, FP.Structural.exec, ComputeKernel.surfaceBody,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr,
    FP.Structural.evalComputeOp, FP.Structural.eval_intShiftRight, FP.Structural.eval_intBitAnd,
    FP.Structural.evalOp_unfold, FP.Structural.evalOffsets, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, FP.Structural.natLt, FP.Structural.numericEq, ComputeDType.eraseDType,
    FP.Structural.ofFloat, FP.Structural.toFloat, FloatDType.numericDType, FloatDType.toTileDType,
    Region.cast, Option.bind, integralFloorDiv, integralMod]; rfl)

theorem execution_memory {α : Type} [Inhabited α] (f : Profile) (M : Algebra α) (p : Parameters) (s : State α) :
    (execution f M p s).val.mem = (written f M p s).mem := by
  have hmask (n : Nat) : n &&& 15 < 4294967296 :=
    lt_of_le_of_lt Nat.and_le_right (by decide)
  have hmaskInt (n : Nat) : ((n &&& 15 : Nat) : Int) % 4294967296 = ((n &&& 15 : Nat) : Int) := by
    apply Int.emod_eq_of_lt (by omega)
    exact_mod_cast hmask n
  cases f <;> simp +contextual [execution, written, cell, Profile.dtype, masked_scatter_memory, value, difference, nibble,
    active, weightOffset, zeroOffset, scaleOffset, outputOffset, row, column,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    integralFloorDiv, integralMod, and_assoc, and_left_comm, and_comm,
    BitVec.toInt_eq_toNat_cond, BitVec.toNat_and, BitVec.toNat_ofInt,
    Int.shiftRight_eq_div_pow, Nat.shiftLeft_eq, Int.shiftLeft_eq,
    Nat.mod_eq_of_lt (hmask _), hmaskInt]

#axiomsClean execution
#axiomsClean execution_memory

end
end VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.Source
