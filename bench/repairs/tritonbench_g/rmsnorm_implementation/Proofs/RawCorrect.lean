import bench.repairs.tritonbench_g.rmsnorm_implementation.Contract
import bench.repairs.tritonbench_g.rmsnorm_implementation.Proofs.Source
import bench.repairs.tritonbench_g.rmsnorm_implementation.Proofs.ActiveOutput

/-! Kernel-only correctness of the displayed two-pass source. Input values are
independent logical arrays. Only active output coordinates must be injective;
zero strides on singleton output dimensions are therefore supported. -/
namespace VeriTile.Bench.Repairs.RmsnormImplementation.RawCorrect
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.RmsnormImplementation
noncomputable section

def KernelCorrect (source : ComputeKernel) (sxb sxm sxk sw sob som sok n : Nat) (eps : ℝ) : Prop :=
  ∀ (before : BlockState) (xs ws : Nat → ℝ),
    (∀ k, k < n → before.readMem "x" (xColOffset before sxb sxm sxk k) = xs k) →
    (∀ k, k < n → before.readMem "weights" (k*sw) = ws k) →
    Function.Injective (fun k : Fin n => outColOffset before sob som sok k.val) →
    ∃ final, exec source.eraseDType.toAlgKernel before = some final ∧
      (∀ k : Fin n, final.readMem "out" (outColOffset before sob som sok k.val) =
        mathematical n eps xs ws k.val) ∧
      (∀ (r : RegionName) address,
        (r ≠ "out" ∨ ∀ k : Fin n, address ≠ outColOffset before sob som sok k.val) →
        final.mem r address = before.mem r address)

theorem rmsnorm_kernel_correctness (sxb sxm sxk sw sob som sok n : Nat) (eps : ℝ) :
    Spec.Real (KernelCorrect
      (kernel "x" "weights" "out" sxb sxm sxk sw sob som sok n blockSize eps)
      sxb sxm sxk sw sob som sok n eps) := by
  intro before xs ws hx hw unique
  obtain ⟨final, he, hv, hf⟩ := ActiveOutput.run .triv "x" "weights" "out"
    sxb sxm sxk sw sob som sok n blockSize eps (by decide) (by decide) (by decide) before unique
  rw [execR_triv] at he
  refine ⟨final, (Source.original_run _ _ _ _ _ _ _ _ _ _ _ _ _ _).trans he, ?_, hf⟩
  intro k
  rw [hv]
  have sum : rmsVarFullNCarrier before "x" sxb sxm sxk n blockSize =
      ∑ j : Fin n, (xs j.val)^2 := by
    unfold rmsVarFullNCarrier
    apply Finset.sum_congr rfl
    intro j _
    rw [hx j.val j.isLt]
  simp only [xColOffset] at hx
  simp only [rmsnormWeightedYFullNSpec, rmsnormYFullNSpec, rmsInvVarFullN, sum,
    hx k.val k.isLt, hw k.val k.isLt, mathematical, div_eq_mul_inv,
    one_mul]

theorem rmsnorm_index_commuted_correctness (sxb sxm sxk sw sob som sok n : Nat) (eps : ℝ) :
    Spec.Real (KernelCorrect
      (indexCommutedKernel "x" "weights" "out" sxb sxm sxk sw sob som sok n blockSize eps)
      sxb sxm sxk sw sob som sok n eps) := by
  intro before xs ws hx hw unique
  obtain ⟨final, he, hv, hf⟩ := rmsnorm_kernel_correctness sxb sxm sxk sw sob som sok n eps
    before xs ws hx hw unique
  exact ⟨final, (Source.commuted_run _ _ _ _ _ _ _ _ _ _ _ _ _ _).trans he, hv, hf⟩

#specNonCircular mathematical avoiding [kernel, indexCommutedKernel, exec]
#axiomsClean rmsnorm_kernel_correctness
#axiomsClean rmsnorm_index_commuted_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormImplementation.RawCorrect
