import bench.tritonbench_g.bgmv_expand_slice.Sources.Mathematical_Kernels

/-! Pure call geometry and storage tags for the checked direct wrapper.
The complete signed source is shared through SurfaceSource. -/
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

abbrev hostCast := VeriTile.Bench.TritonBenchG.BgmvExpandSlice.TypedSource.hostCast

def Primary (h : Call) : Prop :=
  h.inputType=.fp32 ∧ h.weightType=.fp16 ∧ h.outputType=.fp32

theorem split_length (h : Call) (positive : 0<h.N) : (h.N+h.N-1)/h.N=1 := by
  change (h.N+h.N-1)/h.N=1
  apply Nat.div_eq_of_lt_le <;> omega

#axiomsClean split_length
end Kernels
end VeriTile.Bench.Repairs.BgmvExpandSlice
