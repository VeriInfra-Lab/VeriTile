import bench.repairs.tritonbench_g.masked_add_cuda.Proofs.Source
import bench.repairs.tritonbench_g.masked_add_cuda.Proofs.MemoryContract
import bench.tritonbench_g.masked_add_cuda.Proofs.IntegerReal

/-! Kernel-only Real specifications. Each in-bounds lane has the independent
veto formula, and every cell outside the actual write set is unchanged.
Read-only data/mask regions may observe overlapping physical storage. -/
namespace VeriTile.Bench.Repairs.MaskedAddCuda.KernelRealResults
open VeriTile Triton
open VeriTile.Bench.Repairs.MaskedAddCuda.Correct
open VeriTile.Bench.TritonBenchG.MaskedAddCuda
noncomputable section

/-- Successful execution with independently bound input values and a complete
write-set frame. There is no allocation or input-separation premise. -/
def KernelCorrect (source : ComputeKernel) (n : Nat) (alpha : ℝ) : Prop :=
  ∀ (before : BlockState) (grad data : Nat → ℝ) (mask : Nat → Int),
    (∀ j : Fin blockSize, integerAddOffset before blockSize j < n →
      before.readMem "grad" (integerAddOffset before blockSize j) = grad (integerAddOffset before blockSize j)) →
    (∀ j : Fin blockSize, integerAddOffset before blockSize j < n →
      before.readMem "data" (integerAddOffset before blockSize j) = data (integerAddOffset before blockSize j)) →
    (∀ j : Fin blockSize, integerAddOffset before blockSize j < n →
      before.readMemValue .int "mask" (integerAddOffset before blockSize j) = mask (integerAddOffset before blockSize j)) →
    ∃ final, exec source.eraseDType.toAlgKernel before = some final ∧
      (∀ j : Fin blockSize, integerAddOffset before blockSize j < n →
        final.readMem "grad" (integerAddOffset before blockSize j) =
          formula (grad (integerAddOffset before blockSize j))
            (data (integerAddOffset before blockSize j)) (mask (integerAddOffset before blockSize j)) alpha) ∧
      (∀ (r : RegionName) i, (r ≠ "grad" ∨ ∀ j : Fin blockSize,
        integerAddOffset before blockSize j < n ∧ mask (integerAddOffset before blockSize j) = 0 →
          i ≠ integerAddOffset before blockSize j) → final.mem r i = before.mem r i)

theorem masked_add_kernel_correctness (storage : MaskStorage) (n : Nat) (alpha : ℝ) :
    Spec.Real (KernelCorrect (kernel storage "grad" "data" "mask" n alpha) n alpha) := by
  intro before grad data mask hg hp hm
  obtain ⟨final, he⟩ := integer_math_kernel_exec_isSome "grad" "data" "mask" n alpha blockSize before
  refine ⟨final, ?_, ?_, ?_⟩
  · rw [kernel_eq_original, integer_exec]
    exact he
  · intro j hj
    have hv := integer_math_kernel_correct "grad" "data" "mask" n alpha blockSize before final he j
    simpa only [integerAddActive, integerAddSpec, hj, true_and, hg j hj, hp j hj,
      hm j hj, formula] using hv
  · intro r i outside
    apply integer_math_kernel_frame "grad" "data" "mask" n alpha blockSize before final he
    intro j hj hz hit
    rcases outside with hr | ho
    · exact hr hit.1.symm
    · apply ho j ⟨hj, (hm j hj).symm.trans hz⟩
      exact hit.2.symm

theorem masked_add_commuted_correctness (storage : MaskStorage) (n : Nat) (alpha : ℝ) :
    Spec.Real (KernelCorrect (commutedKernel storage "grad" "data" "mask" n alpha) n alpha) := by
  intro before grad data mask hg hp hm
  obtain ⟨final, he, hv, hf⟩ := masked_add_kernel_correctness storage n alpha before grad data mask hg hp hm
  refine ⟨final, ?_, hv, hf⟩
  rw [kernel_eq_original, integer_exec] at he
  rw [commutedKernel_eq_original, integer_exec]
  exact he

/-- Bounds concern each accessed region separately; read-only aliases are legal. -/
theorem masked_add_access_safe (storage : MaskStorage) (n : Nat) (alpha : ℝ)
    (before : BlockState) (bounds : RegionBounds)
    (hg : n ≤ bounds "grad") (hp : n ≤ bounds "data") (hm : n ≤ bounds "mask") :
    Kernel.TraceSafe bounds (kernel storage "grad" "data" "mask" n alpha).eraseDType.toAlgKernel before ∧
    Kernel.TraceSafe bounds (commutedKernel storage "grad" "data" "mask" n alpha).eraseDType.toAlgKernel before := by
  rw [kernel_eq_original, commutedKernel_eq_original]
  exact ⟨integer_traceSafe storage Bool.false _ _ _ n alpha blockSize bounds before
      (fun _ h => Nat.lt_of_lt_of_le h hp) (fun _ h => Nat.lt_of_lt_of_le h hg)
      (fun _ h => Nat.lt_of_lt_of_le h hm),
    integer_traceSafe storage Bool.true _ _ _ n alpha blockSize bounds before
      (fun _ h => Nat.lt_of_lt_of_le h hp) (fun _ h => Nat.lt_of_lt_of_le h hg)
      (fun _ h => Nat.lt_of_lt_of_le h hm)⟩


#specNonCircular formula avoiding [kernel, commutedKernel, exec]
#axiomsClean masked_add_kernel_correctness
#axiomsClean masked_add_commuted_correctness
#axiomsClean masked_add_access_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MaskedAddCuda.KernelRealResults
