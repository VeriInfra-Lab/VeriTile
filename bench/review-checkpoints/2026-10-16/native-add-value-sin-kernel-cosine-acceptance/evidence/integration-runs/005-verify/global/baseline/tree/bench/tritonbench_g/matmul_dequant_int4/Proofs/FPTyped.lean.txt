import bench.tritonbench_g.matmul_dequant_int4.MatmulDequantInt4
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy
import VeriTile.Triton.Float.FixedInteger
import VeriTile.Triton.Float.ScalarOps
namespace VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped
open VeriTile.Triton FP.Structural
open KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
attribute [local simp 10000] FP.Structural.eval_intShiftRight FP.Structural.eval_intBitAnd
structure Parameters where
  (weight scale zero output : RegionName)
  (rows columns groupSize weightRowStride weightColumnStride scaleRowStride scaleColumnStride
    zeroRowStride zeroColumnStride outputRowStride outputColumnStride blockRows blockColumns : Nat)

/-- Standalone dequantize, with native signed int32 packed loads and fp16
scale/output. The integer-to-fp16 operand promotion is materialized. -/
def typed (dtype : FloatDType)
    (b_ptr b_scale_ptr b_zp_ptr : RegionName)
    (fpb_ptr : RegionName)
    (K N group_size stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn stride_fpbk stride_fpbn : Nat)
    (BLOCK_SIZE_K BLOCK_SIZE_N : Nat) : ComputeKernel := triton {
  tl.region b_ptr = tl.int32, b_zp_ptr = tl.int32, b_scale_ptr = DTYPE, fpb_ptr = DTYPE
  k_block_idx = tl.program_id(axis=0)
  n_block_idx = tl.program_id(axis=1)
  offs_k = k_block_idx * $(BLOCK_SIZE_K) + tl.arange(0, $(BLOCK_SIZE_K))
  offs_n = n_block_idx * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  fpb_offs = offs_k[:, None] * $(stride_fpbk) + offs_n[None, :] * $(stride_fpbn)
  b_offs = (offs_k[:, None] // $(8)) * $(stride_bk) + offs_n[None, :] * $(stride_bn)
  bzp_offs = (offs_k[:, None] // $(group_size)) * $(stride_bzpk) + (offs_n[None, :] // $(8)) * $(stride_bzpn)
  bs_offs = (offs_k[:, None] // $(group_size)) * $(stride_bsk) + offs_n[None, :] * $(stride_bsn)
  n_mask = offs_n[None, :] < $(N)
  k_mask = offs_k[:, None] < $(K)
  mask = n_mask & k_mask
  int32_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
  zp_b = tl.load(b_zp_ptr + bzp_offs, mask=mask, other=0.0)
  scale_b = tl.load(b_scale_ptr + bs_offs, mask=mask, other=0.0)
  b_shift = (offs_k[:, None] % $(8)) * $(4)
  bzp_shift = (offs_n[None, :] % $(8)) * $(4)
  fp_weight = tl.toReal(((int32_b >> b_shift) & 15) - ((zp_b >> bzp_shift) & 15)).to(DTYPE) * scale_b
  tl.store(fpb_ptr + fpb_offs, fp_weight, mask=mask)
}

def source (p : Parameters) : ComputeKernel := typed .fp16 p.weight p.scale p.zero p.output
  p.rows p.columns p.groupSize p.weightRowStride p.weightColumnStride p.scaleRowStride p.scaleColumnStride
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

def value {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) (i : Index p) : α :=
  M.binary none .fp16 .mul (M.cast (some .fp32) .real .fp16 (M.fromInt (some .fp32) (difference p s i)))
    ((s.mem p.scale (scaleOffset p s i)).read .fp16)

def written {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) : State α :=
  (TileShape.allIndices [p.blockRows, p.blockColumns]).foldl (fun t i => if active p s i then
    t.write p.output (outputOffset p s i) (.mk .fp16 (value M p s i)) else t) s

def execution {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    {t : State α // FP.Structural.exec M (source p) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [source, typed, FP.Structural.exec, ComputeKernel.surfaceBody,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr,
    FP.Structural.evalComputeOp, FP.Structural.eval_intShiftRight, FP.Structural.eval_intBitAnd,
    FP.Structural.evalOp_unfold, FP.Structural.evalOffsets, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, FP.Structural.natLt, FP.Structural.numericEq, ComputeDType.eraseDType,
    FP.Structural.ofFloat, FP.Structural.toFloat, FloatDType.numericDType, FloatDType.toTileDType,
    Region.cast, Option.bind, integralFloorDiv, integralMod]
  rfl

theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    (execution M p s).val.mem = (written M p s).mem := by
  have hmask (n : Nat) : n &&& 15 < 4294967296 :=
    lt_of_le_of_lt Nat.and_le_right (by decide)
  have hmaskInt (n : Nat) : ((n &&& 15 : Nat) : Int) % 4294967296 = ((n &&& 15 : Nat) : Int) := by
    apply Int.emod_eq_of_lt (by omega)
    exact_mod_cast hmask n
  simp +contextual [execution, written, masked_scatter_memory, value, difference, nibble,
    active, weightOffset, zeroOffset, scaleOffset, outputOffset, row, column,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    integralFloorDiv, integralMod, and_assoc, and_left_comm, and_comm,
    BitVec.toInt_eq_toNat_cond, BitVec.toNat_and, BitVec.toNat_ofInt,
    Int.shiftRight_eq_div_pow, Nat.shiftLeft_eq, Int.shiftLeft_eq,
    Nat.mod_eq_of_lt (hmask _), hmaskInt]

#axiomsClean execution
#axiomsClean execution_memory

end
end VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped
