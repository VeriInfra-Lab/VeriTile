import VeriTile.Triton.DSL
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit

/-! Complete source of `bgmv_shrink_kernel.py`'s `_bgmv_shrink_kernel`.
Both versions retain the signed sentinel, every split-K iteration and mask,
the exact SPLIT_K == 1 choice between store and atomic addition, and explicit
output conversion. The candidate commutes only the final fp32 scaling.

Input and weights share fp16 or bf16 storage, as required by the host. The
unused xk_stride parameter remains in the interface: the source reads its
input contiguously. Program IDs, dimensions, other strides and scaling stay
symbolic. Atomic execution is per program; no cross-program interleaving or
launch-wide reduction is claimed here.

Translation-surface blocker: this macro specializes storage dtypes and makes
Triton's implicit store/atomic operand conversion explicit. The early return
becomes a non-sentinel active guard. `tl.max_contiguous` is a value-preserving
layout hint. The original Python source remains in this directory.

Correct.lean proves both complete real flat-IO contracts; FPEquiv.lean proves
their equivalence from the admitted scaling atom. Proofs/RealFoundation keeps
older active surfaces solely as a source of reusable mathematical lemmas.
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
