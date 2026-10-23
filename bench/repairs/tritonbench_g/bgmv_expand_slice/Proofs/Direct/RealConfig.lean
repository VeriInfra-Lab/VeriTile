import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.Direct.Kernels
import bench.tritonbench_g.bgmv_expand_slice.Proofs.RealSetupSource

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.RealConfig
open VeriTile.Triton Kernels
abbrev Parameters := VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealSetupSource.Parameters

def parameters (h : Call) : Parameters where
  input := "X"
  weights := "W"
  output := "Y"
  indices := ⟨"Indices"⟩
  N := h.N
  K := h.K
  inputBatchStride := h.K
  inputStride := 1
  slotStride := h.N*h.K
  rowStride := h.K
  colStride := 1
  outputBatchStride := h.columns
  outputStride := 1
  slice := h.slice
  BN := 256
  BK := h.BK
  splits := h.N

theorem source_parameters (h : Call) : source h =
    (parameters h).source h.inputType h.weightType h.outputType
      (hostCast h.inputType) Bool.false h.even h.add := by rfl

#axiomsClean source_parameters
end VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.RealConfig
