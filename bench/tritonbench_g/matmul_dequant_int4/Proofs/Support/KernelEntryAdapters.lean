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
  VeriTile.Bench.TritonBenchG.MatmulDequantInt4.Body.unitDequantizeKernel dtype b_ptr b_scale_ptr b_zp_ptr fpb_ptr p.rows p.columns p.groupSize p.weightRowStride p.weightColumnStride p.scaleRowStride p.scaleColumnStride p.zeroRowStride p.zeroColumnStride p.outputRowStride p.outputColumnStride p.blockRows p.blockColumns

def unitSource (f : Source.Profile) (p : Source.Parameters) : ComputeKernel := unitTyped f.dtype p

#axiomsClean valid
end
end VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked
