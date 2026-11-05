import VeriTile.Triton.DSL
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit

/-! Shared floating storage enumeration and the former abstract source-function
interface used by mathematical adapters. Complete executable original/candidate
quotations live in the task-level `Kernels.lean`; this module defines no kernel.
The literal source retains signed indexing, early return, scaling and atomics.
-/
open Lean
open VeriTile.Triton
namespace VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.TypedSource
inductive Storage where
  | fp16 | bf16
  deriving DecidableEq

abbrev KernelSource := FloatDType → RegionName → RegionName → RegionName → Nat → Nat →
  Region .int → ℝ → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → ComputeKernel

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.TypedSource
