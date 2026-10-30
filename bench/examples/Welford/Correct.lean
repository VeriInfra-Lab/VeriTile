import bench.examples.Welford.Kernels
import bench.examples.Welford.Proofs.Real
import VeriTile.Triton
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit

/-! Real correctness of the sources in Kernels.lean.
IO contracts and mathematical formulas are stated below; execution and memory
lemmas are in Proofs/Real.lean. Floating casts are interpreted over the reals. The uint64 count conversion remains;
the explicit length bound excludes wrapping counts from the population-statistics claim. -/

namespace VeriTile.Bench.Examples.WelfordCorrect
open VeriTile.Bench.Examples.Welford.Kernels
open VeriTile Triton
open VeriTile.Triton.TiledReduction.WelfordRec
open VeriTile.Examples (InputRowLoadedAt onlineWelfordLoopBody)
open scoped VeriTile.Triton.KernelIO₁ₓ₂

/-- Use the checked projection without normalizing the entire source AST. -/
private theorem algorithm_of_projection (source : ComputeKernel) (target : AlgKernel)
    (h : source.toAlgorithm? = Except.ok target) : source.toAlgKernel = target := by
  unfold ComputeKernel.toAlgKernel
  rw [h]

private theorem checked_projection (source : ComputeKernel) (target : AlgKernel)
    (h : source.toAlgorithm? = Except.ok target) :
    source.toAlgorithm? = Except.ok source.toAlgKernel := by
  rw [algorithm_of_projection source target h]
  exact h

def twopassIO (N stride : Nat) : KernelIO₁ₓ₂ where
  kernel := (twopassWelfordKernel "x" "mean" "var" N stride).eraseDType
  projection := checked_projection _ _ (twopass_projection "x" "mean" "var" N stride)
  inp := "x"
  out1 := "mean"
  out2 := "var"
  Bin := N
  Bout1 := 1
  Bout2 := 1
  read := fun pid => pid * stride
  write1 := fun _ => 0
  write2 := fun _ => 0

specification twopass_welford_correct (N stride : Nat) (h64 : N < 18446744073709551616) :
    Spec.Real (twopassIO N stride ⊨ fun xs =>
      ((fun _ => (∑ i, xs i) / N),
       (fun _ => (∑ i, (xs i - (∑ j, xs j) / N) ^ 2) / N))) := by
  have halg : (twopassIO N stride).kernel.toAlgKernel =
      (twopassMathKernel "x" "mean" "var" N stride).toAlgKernel := by
    exact algorithm_of_projection _ _ (twopass_projection "x" "mean" "var" N stride)
  refine KernelIO₁ₓ₂.Implements.intro _ ?_ ?_ ?_
  · rw [halg]; exact twopass_flattenOk "x" "mean" "var" N stride
  · intro bounds s hx hm hv _
    rw [halg]; exact twopass_traceSafe "x" "mean" "var" N stride s bounds hx hm hv
  · intro s xs hx
    rw [halg]
    obtain ⟨t, he, hm, hv, hf⟩ := twopass_region_run N stride s xs hx h64
    refine ⟨t, he, ?_, ?_, ?_⟩
    · intro i
      have hi : i.val = 0 := Nat.lt_one_iff.mp i.isLt
      simpa [hi, twoPassMean] using hm
    · intro i
      have hi : i.val = 0 := Nat.lt_one_iff.mp i.isLt
      simpa [hi, twoPassS, twoPassMean] using hv
    · intro r o hm hv _
      apply hf r o
      · rcases hm with h | h
        · exact Or.inl h
        · exact Or.inr (h ⟨0, Nat.zero_lt_one⟩)
      · rcases hv with h | h
        · exact Or.inl h
        · exact Or.inr (h ⟨0, Nat.zero_lt_one⟩)

def onlineIO (N stride : Nat) : KernelIO₁ₓ₂ where
  kernel := (onlineWelfordKernel "x" "mean" "var" N stride).eraseDType
  projection := checked_projection _ _ (online_projection "x" "mean" "var" N stride)
  inp := "x"
  out1 := "mean"
  out2 := "var"
  Bin := N
  Bout1 := 1
  Bout2 := 1
  read := fun pid => pid * stride
  write1 := fun _ => 0
  write2 := fun _ => 0

specification online_welford_correct (N stride : Nat) (h64 : N < 18446744073709551616) :
    Spec.Real (onlineIO N stride ⊨ fun xs =>
      ((fun _ => (∑ i, xs i) / N),
       (fun _ => (∑ i, (xs i - (∑ j, xs j) / N) ^ 2) / N))) := by
  have halg : (onlineIO N stride).kernel.toAlgKernel =
      (onlineMathKernel "x" "mean" "var" N stride).toAlgKernel := by
    exact algorithm_of_projection _ _ (online_projection "x" "mean" "var" N stride)
  refine KernelIO₁ₓ₂.Implements.intro _ ?_ ?_ ?_
  · rw [halg]; exact online_flattenOk "x" "mean" "var" N stride
  · intro bounds s hx hm hv _
    rw [halg]; exact online_traceSafe "x" "mean" "var" N stride s bounds hx hm hv
  · intro s xs hx
    rw [halg]
    obtain ⟨t, he, hm, hv, hf⟩ := online_region_run N stride s xs hx h64
    refine ⟨t, he, ?_, ?_, ?_⟩
    · intro i
      have hi : i.val = 0 := Nat.lt_one_iff.mp i.isLt
      simpa [hi, twoPassMean] using hm
    · intro i
      have hi : i.val = 0 := Nat.lt_one_iff.mp i.isLt
      simpa [hi, twoPassS, twoPassMean] using hv
    · intro r o hm hv _
      apply hf r o
      · rcases hm with h | h
        · exact Or.inl h
        · exact Or.inr (h ⟨0, Nat.zero_lt_one⟩)
      · rcases hv with h | h
        · exact Or.inl h
        · exact Or.inr (h ⟨0, Nat.zero_lt_one⟩)

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.WelfordCorrect
