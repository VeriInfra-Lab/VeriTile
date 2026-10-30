import bench.tritonbench_g.matmul_dequantize.Sources.Mathematical_MatmulDequantize
import bench.tritonbench_g.matmul_dequant_int4.Sources.Mathematical_FPCorrect
namespace VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTyped
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
def dequantTyped (dtype : FloatDType)
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

namespace Standalone
abbrev Parameters := MatmulDequantInt4.FPTyped.Parameters
abbrev Half := MatmulDequantInt4.Binary16.Value

def source (p : Parameters) : ComputeKernel := dequantTyped .fp16 p.weight p.scale p.zero p.output
  p.rows p.columns p.groupSize p.weightRowStride p.weightColumnStride p.scaleRowStride p.scaleColumnStride
  p.zeroRowStride p.zeroColumnStride p.outputRowStride p.outputColumnStride p.blockRows p.blockColumns

/-- Both independent Python sources contain this complete standalone stage;
this bridge reuses its signed-word and concrete binary16 result proof. -/
theorem source_bridge (p : Parameters) :
    source p = MatmulDequantInt4.FPTyped.source p := rfl

theorem correctness [Inhabited Half] (rest : Algebra Half) (p : Parameters) (s : State Half)
    (hInputs : MatmulDequantInt4.FPTyped.PackedInputs p s)
    (hw : ∀ i j, MatmulDequantInt4.FPTyped.active p s i → MatmulDequantInt4.FPTyped.active p s j →
      MatmulDequantInt4.FPTyped.outputOffset p s i = MatmulDequantInt4.FPTyped.outputOffset p s j → i = j) :
    ∃ final, FP.Structural.exec (MatmulDequantInt4.FPTyped.software rest) (source p) s = some final ∧
      MatmulDequantInt4.FPTyped.Post p s final (MatmulDequantInt4.FPTyped.roundedValue p s) ∧
      ∀ i, MatmulDequantInt4.FPTyped.active p s i → ∃ word zero : BitVec 32,
        s.mem p.weight (MatmulDequantInt4.FPTyped.weightOffset p s i) = .mk .int word.toInt ∧
        s.mem p.zero (MatmulDequantInt4.FPTyped.zeroOffset p s i) = .mk .int zero.toInt ∧
        final.mem p.output (MatmulDequantInt4.FPTyped.outputOffset p s i) =
          .mk .fp16 (MatmulDequantInt4.FPTyped.packedResult p s i word zero) := by
  rw [source_bridge]
  exact MatmulDequantInt4.matmul_dequant_int4_fp16_correctness rest p s hInputs hw

theorem bit_success [Inhabited Half] (rest : Algebra Half) :
    ∃ final, FP.Structural.exec (MatmulDequantInt4.FPTyped.software rest)
      (source MatmulDequantInt4.FPTyped.witnessParameters) MatmulDequantInt4.FPTyped.witnessInitial = some final ∧
      final.mem "output" 119 = .mk .fp16 MatmulDequantInt4.FPTyped.resultValue ∧
      final.mem "output" 120 = MatmulDequantInt4.FPTyped.witnessInitial.mem "output" 120 ∧
      final.mem "weight" 7 = .mk .int MatmulDequantInt4.FPTyped.weightWord.toInt ∧
      (-7 * (1365 / 4096) : Rat) ≠ -597 / 256 := by
  rw [source_bridge]
  exact MatmulDequantInt4.matmul_dequant_int4_bit_success rest
#axiomsClean source_bridge
#axiomsClean correctness
#axiomsClean bit_success
end Standalone
end
end VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTyped
