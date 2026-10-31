import VeriTile.Triton

/-! Port-local signed-mask IO skin, assembled through the existing unified flat-memory bridge. -/
namespace VeriTile.Bench.TritonBenchG.MaskedAddCuda
open VeriTile.Triton

structure IntMasked2DKernelIO₂ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
  /-- First input buffer (ℝ channel). -/
  in1 : RegionName
  /-- Second input buffer (ℝ channel). -/
  in2 : RegionName
  /-- Integer input buffer (`.int` channel). -/
  mbuf : RegionName
  /-- Output buffer. -/
  out : RegionName
  /-- Tile length: each program instance owns `B`-lane windows. -/
  B : Nat
  /-- Lane `j`'s `in1` read address for program `(pid₀, pid₁)`. -/
  read1 : Nat → Nat → Fin B → Nat
  /-- Lane `j`'s `in2` read address. -/
  read2 : Nat → Nat → Fin B → Nat
  /-- Lane `j`'s `mbuf` read address (the integer tile's window). -/
  readm : Nat → Nat → Fin B → Nat
  /-- Lane `j`'s write address. -/
  write : Nat → Nat → Fin B → Nat
  /-- Program `(pid₀, pid₁)`'s **static read-active** lanes — the
  bounds/trace-safety superset. -/
  mask : Nat → Nat → Fin B → Prop
  /-- Program `(pid₀, pid₁)`'s **write-active** lanes given the loaded integer
  tile; defaults to the static `mask`. -/
  writeMask : Nat → Nat → (Fin B → Int) → Fin B → Prop :=
    fun p₀ p₁ _ j => mask p₀ p₁ j

namespace IntMasked2DKernelIO₂

end IntMasked2DKernelIO₂
end VeriTile.Bench.TritonBenchG.MaskedAddCuda
