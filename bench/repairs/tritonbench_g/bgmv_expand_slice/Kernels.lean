import bench.tritonbench_g.bgmv_expand_slice.Kernels

/-! The checked wrapper launches `directSource` on original contiguous storage.
The historical padded `source` is retained for archived proof credit only.
Both quote the same numerical kernel. Explicit i64 widening is justified by
the direct wrapper's derived bounds. Selected precision: FP32 input cast to
FP16; FP16 products/reduction; FP32 output addition and store.
-/
namespace VeriTile.Bench.Repairs.BgmvExpandSlice
open VeriTile.Triton
namespace Original
abbrev InputStorage := VeriTile.Bench.TritonBenchG.BgmvExpandSlice.TypedSource.InputStorage
abbrev WeightStorage := VeriTile.Bench.TritonBenchG.BgmvExpandSlice.TypedSource.WeightStorage
end Original

/-- Shared mathematical call descriptor; `Contract.call` derives the checked
direct call from the actual Python conditions and next_power_of_2. -/
structure Call where
  batches : Nat
  adapters : Nat
  N : Nat
  K : Nat
  columns : Nat
  slice : Nat
  BK : Nat
  blockPositive : 0 < BK
  rankCovered : K ≤ BK
  sliceInside : slice + N ≤ columns
  even : Bool
  evenExact : even = Bool.true → K = BK ∨ K = 0
  add : Bool
  inputType : Original.InputStorage := .fp32
  weightType : Original.WeightStorage := .fp16
  outputType : FloatDType := .fp32

namespace Kernels
abbrev original := VeriTile.Bench.TritonBenchG.BgmvExpandSlice.TypedSource.original
abbrev hostCast := VeriTile.Bench.TritonBenchG.BgmvExpandSlice.TypedSource.hostCast


def source (h : Call) : ComputeKernel :=
  original h.inputType h.weightType (hostCast h.inputType) h.outputType
    "X" "W" "Y" h.N h.K ⟨"Indices"⟩ h.BK 1 ((h.N+255)*h.BK) h.BK 1
    (h.columns+255) 1 h.slice 256 h.BK h.N h.even h.add


/-- Checked direct contiguous source: original backing storage and canonical
active row strides, with no snapshots, padding allocations or copyback. -/
def directSource (h : Call) : ComputeKernel :=
  original h.inputType h.weightType (hostCast h.inputType) h.outputType
    "X" "W" "Y" h.N h.K ⟨"Indices"⟩ h.K 1 (h.N*h.K) h.K 1
    h.columns 1 h.slice 256 h.BK h.N h.even h.add

def Primary (h : Call) : Prop :=
  h.inputType=.fp32 ∧ h.weightType=.fp16 ∧ h.outputType=.fp32

theorem split_length (h : Call) (positive : 0<h.N) : (h.N+h.N-1)/h.N=1 := by
  change (h.N+h.N-1)/h.N=1
  apply Nat.div_eq_of_lt_le <;> omega

#axiomsClean split_length
end Kernels
end VeriTile.Bench.Repairs.BgmvExpandSlice
