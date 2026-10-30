import bench.repairs.tritonbench_g.dropout_triton.Proofs.Source
import bench.repairs.tritonbench_g.dropout_triton.Proofs.Region
import bench.repairs.tritonbench_g.dropout_triton.Contract

/-! Kernel-only mathematical specifications. The total Real interpretation and
ordinary inverted-dropout domain are named separately; neither changes Python
admission or adds a host execution obligation. -/
namespace VeriTile.Bench.Repairs.DropoutTriton.RealEndpoints
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.DropoutTriton
open Correct
noncomputable section

/-- Optional flat-memory contract. Its pairwise-disjoint allocation premise
is stronger than Python admission and is not used by the primary region theorem. -/
def dropoutIO (source : ComputeKernel) (n : Nat) (p : ℝ) : Prop :=
  RealIO.Implements source.eraseDType .int32 "x" "keep" "out" n blockSize p

/-- Successful per-program execution, active output values, finite access
bounds and full tagged-memory frame, using the explicitly total Real division. -/
theorem dropout_kernel_flat_correctness (n : Nat) (p : ℝ) :
    Spec.Real (dropoutIO (kernel n p) n p) :=
  RealIO.original_correct .fp32 .int32 "x" "keep" "out" n blockSize p

theorem dropout_index_commuted_flat_correctness (n : Nat) (p : ℝ) :
    Spec.Real (dropoutIO (indexCommutedKernel n p) n p) :=
  RealIO.candidate_correct .fp32 .int32 "x" "keep" "out" n blockSize p

/-- Canonical-source Real correctness, independent of physical input separation.
The named input regions may be two observations of the same backing bytes.
Only actual active input reads are bound; the output formula and frame follow
from successful execution of the displayed kernel. -/
theorem dropout_kernel_correctness (n : Nat) (p : ℝ) (before : BlockState)
    (xs : Nat → ℝ) (keep : Nat → Int)
    (hx : ∀ j : Fin blockSize, RealProof.offset before blockSize j < n →
      before.readMem "x" (RealProof.offset before blockSize j) = xs (RealProof.offset before blockSize j))
    (hk : ∀ j : Fin blockSize, RealProof.offset before blockSize j < n →
      before.readMemValue .int "keep" (RealProof.offset before blockSize j) = keep (RealProof.offset before blockSize j)) :
    Spec.Real (∃ final, exec (kernel n p).eraseDType.toAlgKernel before = some final ∧
      (∀ j : Fin blockSize, RealProof.offset before blockSize j < n →
        final.readMem "out" (RealProof.offset before blockSize j) =
          expected xs keep p (RealProof.offset before blockSize j)) ∧
      (∀ (r : RegionName) i, (r ≠ "out" ∨ ∀ j : Fin blockSize,
        RealProof.offset before blockSize j < n → i ≠ RealProof.offset before blockSize j) →
        final.mem r i = before.mem r i)) := by
  obtain ⟨final, he, hv, hf⟩ := RealProof.run .int32 "x" "keep" "out" n blockSize p before
  refine ⟨final, ?_, ?_, hf⟩
  · simpa only [exec, RegionProof.original_body] using he
  · intro j hj
    rw [hv j hj]
    simp only [RealProof.expected, RealProof.readKeep, hx j hj, hk j hj,
      decide_eq_true_eq, expected]

theorem dropout_index_commuted_correctness (n : Nat) (p : ℝ) (before : BlockState)
    (xs : Nat → ℝ) (keep : Nat → Int)
    (hx : ∀ j : Fin blockSize, RealProof.offset before blockSize j < n →
      before.readMem "x" (RealProof.offset before blockSize j) = xs (RealProof.offset before blockSize j))
    (hk : ∀ j : Fin blockSize, RealProof.offset before blockSize j < n →
      before.readMemValue .int "keep" (RealProof.offset before blockSize j) = keep (RealProof.offset before blockSize j)) :
    Spec.Real (∃ final, exec (indexCommutedKernel n p).eraseDType.toAlgKernel before = some final ∧
      (∀ j : Fin blockSize, RealProof.offset before blockSize j < n →
        final.readMem "out" (RealProof.offset before blockSize j) =
          expected xs keep p (RealProof.offset before blockSize j)) ∧
      (∀ (r : RegionName) i, (r ≠ "out" ∨ ∀ j : Fin blockSize,
        RealProof.offset before blockSize j < n → i ≠ RealProof.offset before blockSize j) →
        final.mem r i = before.mem r i)) := by
  obtain ⟨final, he, hv, hf⟩ := dropout_kernel_correctness n p before xs keep hx hk
  exact ⟨final, (RegionProof.candidate_exec n p before).symm.trans he, hv, hf⟩

/-- Access bounds concern each input separately and do not require the two
read-only physical views to be disjoint. -/
theorem dropout_kernel_access_safe (n : Nat) (p : ℝ) (before : BlockState)
    (bounds : RegionBounds) (hx : n ≤ bounds "x") (hk : n ≤ bounds "keep")
    (ho : n ≤ bounds "out") :
    Kernel.TraceSafe bounds (kernel n p).eraseDType.toAlgKernel before := by
  have safe := RealProof.trace_safe .int32 "x" "keep" "out" n blockSize p bounds before
    (fun _ active => Nat.lt_of_lt_of_le active hx)
    (fun _ active => Nat.lt_of_lt_of_le active hk)
    (fun _ active => Nat.lt_of_lt_of_le active ho)
  simpa only [Kernel.TraceSafe, RegionProof.original_body] using safe

/-- The conventional partial formula is meaningful exactly on its declared
mathematical domain. This condition is not a Python input guard. -/
theorem mathematical_projection (n : Nat) (xs : Nat → ℝ) (keep : Nat → Int) (p : ℝ)
    (domain : MathematicalDomain n keep p) (i : Nat) (hi : i < n) :
    mathematicalValue (xs i) (keep i) p = some (expected xs keep p i) := by
  by_cases hk : keep i = 0
  · simp [mathematicalValue, expected, hk]
  · simp [mathematicalValue, expected, hk, domain i hi hk]

/-- Conventional mathematical formula, using the same alias-permitting
regional execution as the primary theorem. The only extra condition is the
explicit kept-coordinate singularity domain, not input separation. -/
theorem dropout_mathematical_correct (n : Nat) (p : ℝ) (before : BlockState)
    (xs : Nat → ℝ) (keep : Nat → Int)
    (hx : ∀ j : Fin blockSize, RealProof.offset before blockSize j < n →
      before.readMem "x" (RealProof.offset before blockSize j) = xs (RealProof.offset before blockSize j))
    (hk : ∀ j : Fin blockSize, RealProof.offset before blockSize j < n →
      before.readMemValue .int "keep" (RealProof.offset before blockSize j) = keep (RealProof.offset before blockSize j))
    (domain : MathematicalDomain n keep p) :
    Spec.Real (∃ final, exec (kernel n p).eraseDType.toAlgKernel before = some final ∧
      (∀ j : Fin blockSize, RealProof.offset before blockSize j < n →
        mathematicalValue (xs (RealProof.offset before blockSize j))
          (keep (RealProof.offset before blockSize j)) p =
          some (final.readMem "out" (RealProof.offset before blockSize j))) ∧
      (∀ (r : RegionName) i, (r ≠ "out" ∨ ∀ j : Fin blockSize,
        RealProof.offset before blockSize j < n → i ≠ RealProof.offset before blockSize j) →
        final.mem r i = before.mem r i)) := by
  obtain ⟨final, he, hv, hf⟩ := dropout_kernel_correctness n p before xs keep hx hk
  refine ⟨final, he, ?_, hf⟩
  intro j hj
  rw [hv j hj]
  exact mathematical_projection n xs keep p domain _ hj


/-- Logical tile inputs for the shared mixed-channel IO assembly. -/
theorem masked_run (n : Nat) (p : ℝ) (s : BlockState)
    (xs : Fin blockSize → ℝ) (keep : Fin blockSize → Int)
    (hx : ∀ j, RealProof.offset s blockSize j<n →
      s.readMem "x" (RealProof.offset s blockSize j)=xs j)
    (hk : ∀ j, RealProof.offset s blockSize j<n →
      s.readMemValue .int "keep" (RealProof.offset s blockSize j)=keep j) :
    ∃ t,exec (kernel n p).eraseDType.toAlgKernel s=some t ∧
      (∀ j,RealProof.offset s blockSize j<n →
        t.readMem "out" (RealProof.offset s blockSize j)=
          if keep j≠0 then xs j/(1-p) else 0) ∧
      (∀ (r : RegionName) address,(r≠"out" ∨ ∀ j,RealProof.offset s blockSize j<n →
        address≠RealProof.offset s blockSize j) → t.mem r address=s.mem r address) := by
  obtain ⟨t,ran,values,frame⟩ := RealProof.run .int32 "x" "keep" "out" n blockSize p s
  refine ⟨t,?_,?_,frame⟩
  · simpa only [exec,RegionProof.original_body] using ran
  · intro j hj
    rw [values j hj]
    simp only [RealProof.expected,RealProof.readKeep,hx j hj,hk j hj,decide_eq_true_eq]

theorem masked_safe (n : Nat) (p : ℝ) (bounds : RegionBounds) (s : BlockState)
    (hx : ∀ j : Fin blockSize,RealProof.offset s blockSize j<n →
      RealProof.offset s blockSize j<bounds "x")
    (hk : ∀ j : Fin blockSize,RealProof.offset s blockSize j<n →
      RealProof.offset s blockSize j<bounds "keep")
    (ho : ∀ j : Fin blockSize,RealProof.offset s blockSize j<n →
      RealProof.offset s blockSize j<bounds "out") :
    Kernel.TraceSafe bounds (kernel n p).eraseDType.toAlgKernel s := by
  have safe := RealProof.trace_safe .int32 "x" "keep" "out" n blockSize p bounds s hx hk ho
  simpa only [Kernel.TraceSafe,RegionProof.original_body] using safe

theorem candidate_masked_run (n : Nat) (p : ℝ) (s : BlockState)
    (xs : Fin blockSize → ℝ) (keep : Fin blockSize → Int)
    (hx : ∀ j, RealProof.offset s blockSize j<n →
      s.readMem "x" (RealProof.offset s blockSize j)=xs j)
    (hk : ∀ j, RealProof.offset s blockSize j<n →
      s.readMemValue .int "keep" (RealProof.offset s blockSize j)=keep j) :
    ∃ t,exec (indexCommutedKernel n p).eraseDType.toAlgKernel s=some t ∧
      (∀ j,RealProof.offset s blockSize j<n →
        t.readMem "out" (RealProof.offset s blockSize j)=
          if keep j≠0 then xs j/(1-p) else 0) ∧
      (∀ (r : RegionName) address,(r≠"out" ∨ ∀ j,RealProof.offset s blockSize j<n →
        address≠RealProof.offset s blockSize j) → t.mem r address=s.mem r address) := by
  obtain ⟨t,ran,values,frame⟩ := masked_run n p s xs keep hx hk
  exact ⟨t,(RegionProof.candidate_exec n p s).symm.trans ran,values,frame⟩

#specNonCircular mathematicalValue avoiding [kernel, indexCommutedKernel]
#specNonCircular expected avoiding [kernel, indexCommutedKernel]
#axiomsClean dropout_kernel_flat_correctness
#axiomsClean dropout_index_commuted_flat_correctness
#axiomsClean dropout_kernel_access_safe
#axiomsClean dropout_kernel_correctness
#axiomsClean dropout_index_commuted_correctness
#axiomsClean dropout_mathematical_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DropoutTriton.RealEndpoints
