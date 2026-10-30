import bench.tritonbench_g.square_matrix.Proofs.FP
import bench.tritonbench_g.square_matrix.Proofs.Mathematical_Source

namespace VeriTile.Bench.Repairs.SquareMatrix.KernelFP
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SquareMatrix
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- The complete original typed source succeeds even for arbitrary floating
primitive implementations. This adds no numerical law or finite-value domain. -/
theorem successful {α : Type} [Inhabited α] (M : Algebra α)
    (n B : Nat) (s : State α) :
    ∃ t, FP.Structural.exec M (fp32Kernel n B) s = some t := by
  simp [fp32Kernel, typedKernel, FP.Structural.exec, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    numericLt, natLt, bop, store, ComputeDType.eraseDType, numeric]

/-- Actual full-source execution equality for the existing index rewrite.
The floating multiplication, even in inactive lanes, is unchanged. -/
theorem variants_equal {α : Type} [Inhabited α] (M : Algebra α)
    (n B : Nat) (s : State α) :
    FP.Structural.exec M (fp32Kernel n B) s =
      FP.Structural.exec M (indexCommutedKernel n B) s := by
  simp [fp32Kernel, indexCommutedKernel, typedKernel, FP.Structural.exec,
    ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, numericLt, natLt, bop, store, ComputeDType.eraseDType, numeric,
    Nat.mul_comm]
  have swapped :
      bop (fun a b : Nat => a*b) Broadcast.nil (fun _ : TileIndex [] => s.pids 0) (fun _ => n) =
        bop (fun a b : Nat => a*b) Broadcast.nil (fun _ : TileIndex [] => n) (fun _ => s.pids 0) := by
    funext i
    exact Nat.mul_comm _ _
  rw [swapped]

/-- Per-row native FP32 square with complete typed memory observations and
frame. Same-format loads use the established bit-preserving load model; the
mul operation retains its explicit FP32 profile. -/
theorem coordinates {α : Type} [Inhabited α] (M : Algebra α)
    (n : Nat) (s : State α) :
    ∃ t, FP.Structural.exec (FPProof.backend M) (fp32Kernel n (block n)) s = some t ∧
      (∀ col, col < n → t.mem "out" (s.pids 0*n+col) =
        .mk .real (FPProof.reference M ((s.mem "x" (s.pids 0*n+col)).read .real))) ∧
      (∀ (r : RegionName) o,
        r ≠ "out" ∨ o < s.pids 0*n ∨ s.pids 0*n+n ≤ o → t.mem r o = s.mem r o) := by
  obtain ⟨t, he, hm⟩ := FPProof.complete_correct M n (block n) s
  refine ⟨t, he, ?_, ?_⟩
  · intro col active
    rw [hm, FPProof.expected_cell M n (block n) s (block_covers n)]
    simp [active, FPProof.resultCell]
  · intro r o outside
    rw [hm, FPProof.expected_cell M n (block n) s (block_covers n)]
    apply if_neg
    rintro ⟨region, lower, upper⟩
    rcases outside with different | before | after
    · exact different region
    · omega
    · omega

theorem precision (n B : Nat) :
    (fp32Kernel n B).surfaceBody[5]? = some (expectedProduct .fp32 B) ∧
    (fp32Kernel n B).surfaceBody.getLast? = some (expectedStore .fp32 n B) :=
  ⟨multiplication_precision .fp32 Bool.false _ _ _ _ _ _,
   store_precision .fp32 Bool.false _ _ _ _ _ _⟩

#axiomsClean successful
#axiomsClean variants_equal
#axiomsClean coordinates
#axiomsClean precision
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SquareMatrix.KernelFP
