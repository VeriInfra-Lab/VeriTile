import bench.repairs.tritonbench_g.bgmv_expand_slice.Kernels
import bench.tritonbench_g.bgmv_expand_slice.Proofs.RealSetupSource

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.RealConfig
open VeriTile.Triton Kernels
abbrev Parameters := VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealSetupSource.Parameters

def parameters (h : Call) : Parameters where
  input := "X"
  weights := "W"
  output := "Y"
  indices := ⟨"Indices"⟩
  N := h.N
  K := h.K
  inputBatchStride := h.BK
  inputStride := 1
  slotStride := (h.N+255)*h.BK
  rowStride := h.BK
  colStride := 1
  outputBatchStride := h.columns+255
  outputStride := 1
  slice := h.slice
  BN := 256
  BK := h.BK
  splits := h.N

theorem source_parameters (h : Call) : source h =
    (parameters h).source h.inputType h.weightType h.outputType
      (hostCast h.inputType) Bool.false h.even h.add := by rfl

#axiomsClean source_parameters
end VeriTile.Bench.Repairs.BgmvExpandSlice.RealConfig
