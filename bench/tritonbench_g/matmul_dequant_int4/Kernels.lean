import bench.tritonbench_g.matmul_dequant_int4.Contract
import bench.tritonbench_g.matmul_dequant_int4.Proofs.Grid

namespace VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked
open VeriTile.Triton
open MatmulDequantInt4.Direct
noncomputable section

/-- The unchanged source configurations, including duplicate autotune entries. -/
def tuning (i : Fin 8) : MatmulDequantInt4.Source.DequantConfig :=
  MatmulDequantInt4.Source.dequantConfigs[i.val]'(by simpa [MatmulDequantInt4.Source.dequantConfigs] using i.isLt)
theorem tuning_mem (i : Fin 8) : tuning i∈MatmulDequantInt4.Source.dequantConfigs := List.getElem_mem _

def params (p : PackedInfo) (i : Fin 8) : Source.Parameters where
  weight := "Q"
  scale := "S"
  zero := "Z"
  output := "O"
  rows := p.weight.tensor.rows*8
  columns := p.weight.tensor.columns
  groupSize := p.group
  weightRowStride := p.weight.tensor.strides.row
  weightColumnStride := p.weight.tensor.strides.column
  scaleRowStride := p.scale.tensor.strides.row
  scaleColumnStride := p.scale.tensor.strides.column
  zeroRowStride := p.zero.tensor.strides.row
  zeroColumnStride := p.zero.tensor.strides.column
  outputRowStride := p.weight.tensor.columns
  outputColumnStride := 1
  blockRows := (tuning i).bk
  blockColumns := (tuning i).bn

def Source.Profile.hostDtype : Source.Profile → MatmulDequantInt4.Host.DType
  | .half => .fp16 | .single => .fp32

/-- Complete readable shared quotation: loads, unpacking, signed subtraction,
selected-width conversion/multiply, mask and store. No numerical rewrite. -/
abbrev dequantize (f : Source.Profile) (p : PackedInfo) (i : Fin 8) := Source.source f (params p i)

theorem valid (p : PackedInfo) (i : Fin 8) : KernelGrid.Valid (params p i) := by
  have h := MatmulDequantInt4.Source.dequant_valid _ (tuning_mem i)
  exact ⟨h.1,h.2.1,rfl,rfl,by simp [params],by simp [params],by simp [params]⟩

/-- Existing output-buffer ABI for the kernel, independent of allocator or
returned-object implementation. The Python wrapper supplies canonical strides. -/
def OutputABI (p : PackedInfo) (f : Source.Profile) (o : Info) : Prop :=
  Ordinary o f.hostDtype (p.weight.tensor.rows*8) p.weight.tensor.columns p.weight.tensor.device ∧
  o.tensor.strides=MatmulDequantInt4.Strides.canonical p.weight.tensor.columns ∧
  ∀ t∈[p.weight.tensor,p.scale.tensor,p.zero.tensor],MatmulDequantInt4.Host.Disjoint o.tensor t

/-- Specialize the existing contiguous output ABI in the complete source.
This removes a vector multiplication by the unit column stride. All loads,
selected-width arithmetic, masks and stores remain in source order. -/
def unitTyped (dtype : FloatDType) (p : Source.Parameters) : ComputeKernel :=
  let b_ptr := p.weight
  let b_zp_ptr := p.zero
  let b_scale_ptr := p.scale
  let fpb_ptr := p.output
  triton {
  tl.region b_ptr = tl.int32, b_zp_ptr = tl.int32, b_scale_ptr = DTYPE, fpb_ptr = DTYPE
  k_block_idx = tl.program_id(axis=0)
  n_block_idx = tl.program_id(axis=1)
  offs_k = k_block_idx * $(p.blockRows) + tl.arange(0, $(p.blockRows))
  offs_n = n_block_idx * $(p.blockColumns) + tl.arange(0, $(p.blockColumns))
  fpb_offs = offs_k[:, None] * $(p.outputRowStride) + offs_n[None, :]
  b_offs = (offs_k[:, None] // $(8)) * $(p.weightRowStride) + offs_n[None, :] * $(p.weightColumnStride)
  bzp_offs = (offs_k[:, None] // $(p.groupSize)) * $(p.zeroRowStride) + (offs_n[None, :] // $(8)) * $(p.zeroColumnStride)
  bs_offs = (offs_k[:, None] // $(p.groupSize)) * $(p.scaleRowStride) + offs_n[None, :] * $(p.scaleColumnStride)
  n_mask = offs_n[None, :] < $(p.columns)
  k_mask = offs_k[:, None] < $(p.rows)
  mask = n_mask & k_mask
  int32_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
  zp_b = tl.load(b_zp_ptr + bzp_offs, mask=mask, other=0.0)
  scale_b = tl.load(b_scale_ptr + bs_offs, mask=mask, other=0.0)
  b_shift = (offs_k[:, None] % $(8)) * $(4)
  bzp_shift = (offs_n[None, :] % $(8)) * $(4)
  fp_weight = tl.toReal(((int32_b >> b_shift) & 15) - ((zp_b >> bzp_shift) & 15)).to(DTYPE) * scale_b
  tl.store(fpb_ptr + fpb_offs, fp_weight, mask=mask)
}

def unitSource (f : Source.Profile) (p : Source.Parameters) : ComputeKernel := unitTyped f.dtype p

#axiomsClean valid
end
end VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked
