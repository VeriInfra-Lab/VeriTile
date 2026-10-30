import bench.tritonbench_g.rmsnorm_implementation.IO
import bench.tritonbench_g.rmsnorm_implementation.Proofs.RawCorrect
import bench.tritonbench_g.rmsnorm_implementation.Proofs.IOSafety

/-! Composition of the established two-pass value/frame proof and actual source
safety into the shared finite regional IO judgment. No runtime or numerical
assumption is added by this adapter. -/
namespace VeriTile.Bench.Repairs.RmsnormImplementation.Interface
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.RmsnormImplementation
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def extend {n : Nat} (xs : Fin n → ℝ) (k : Nat) : ℝ :=
  if h : k<n then xs ⟨k,h⟩ else 0

@[simp] theorem extend_fin {n : Nat} (xs : Fin n → ℝ) (k : Fin n) :
    extend xs k.val = xs k := by simp [extend,k.isLt]

theorem mathematical_formula {n : Nat} (eps : ℝ) (x w : Fin n → ℝ) (k : Fin n) :
    mathematical n eps (extend x) (extend w) k.val = formula eps x w k := by
  simp only [mathematical,formula,extend_fin]

theorem unique_address (n sob som sok : Nat) (h : OutputUnique n sok) (s : BlockState) :
    Function.Injective (fun k : Fin n => outColOffset s sob som sok k.val) := by
  intro a b eq
  apply h
  exact Nat.add_left_cancel eq

/-- The public active-only uniqueness condition is exactly the previous
state-dependent address condition; the common row offset cancels. -/
theorem unique_iff (n sob som sok : Nat) (s : BlockState) :
    OutputUnique n sok ↔ Function.Injective (fun k : Fin n => outColOffset s sob som sok k.val) := by
  refine ⟨fun h => unique_address n sob som sok h s, ?_⟩
  intro h a b eq
  apply h
  simp only [outColOffset,eq]

theorem source_correct (candidate : Bool) (sxb sxm sxk sw sob som sok n : Nat) (eps : ℝ) :
    RawCorrect.KernelCorrect (source candidate sxb sxm sxk sw sob som sok n eps)
      sxb sxm sxk sw sob som sok n eps := by
  cases candidate
  · exact RawCorrect.rmsnorm_kernel_correctness sxb sxm sxk sw sob som sok n eps
  · exact RawCorrect.rmsnorm_index_commuted_correctness sxb sxm sxk sw sob som sok n eps

theorem implements (candidate : Bool) (sxb sxm sxk sw sob som sok n : Nat) (eps : ℝ)
    (unique : OutputUnique n sok) :
    (rowIO candidate sxb sxm sxk sw sob som sok n eps).RegionalScheduledImplements
      oneProgram (fun v _ k => formula eps (v (0 : Fin 2)) (v (1 : Fin 2)) k) := by
  apply UKernelIO.RegionalScheduledImplements.intro
  · intro s vals _ bounds hb
    have hx : ∀ k, k<n → s.pids 0*sxb+s.pids 1*sxm+k*sxk < bounds "x" := by
      intro k hk
      exact hb.1 (0 : Fin 2) ⟨k,hk⟩ trivial
    have hw : ∀ k, k<n → k*sw < bounds "weights" := by
      intro k hk
      exact hb.1 (1 : Fin 2) ⟨k,hk⟩ trivial
    have ho : ∀ k, k<n → s.pids 0*sob+s.pids 1*som+k*sok < bounds "out" := by
      intro k hk
      exact hb.2.1 (0 : Fin 1) ⟨k,hk⟩ trivial
    simp only [oneProgram,algorithmScheduleSafe_singleton]
    change (source candidate sxb sxm sxk sw sob som sok n eps).eraseDType.toAlgKernel.TraceSafeR .triv bounds s
    cases candidate
    · exact IOSafety.original bounds sxb sxm sxk sw sob som sok n eps s
        (unique_address n sob som sok unique s) hx hw ho
    · exact IOSafety.commuted bounds sxb sxm sxk sw sob som sok n eps s
        (unique_address n sob som sok unique s) hx hw ho
  · intro s vals inputs
    have hx : ∀ k, k<n → s.readMem "x" (xColOffset s sxb sxm sxk k) = extend (vals (0 : Fin 2)) k := by
      intro k hk
      simpa [extend,rowIO,ChanTy.read,xColOffset,hk] using inputs (0 : Fin 2) ⟨k,hk⟩ trivial
    have hw : ∀ k, k<n → s.readMem "weights" (k*sw) = extend (vals (1 : Fin 2)) k := by
      intro k hk
      simpa [extend,rowIO,ChanTy.read,xColOffset,hk] using inputs (1 : Fin 2) ⟨k,hk⟩ trivial
    obtain ⟨final,ran,values,frame⟩ := source_correct candidate sxb sxm sxk sw sob som sok n eps
      s (extend (vals (0 : Fin 2))) (extend (vals (1 : Fin 2))) hx hw (unique_address n sob som sok unique s)
    refine ⟨final, ?_, ?_, ?_⟩
    · simpa only [oneProgram,runAlgorithmSchedule_singleton] using ran
    · intro o k _
      exact (values k).trans (mathematical_formula eps (vals (0 : Fin 2)) (vals (1 : Fin 2)) k)
    · intro r address outside _
      apply frame
      by_cases hr : r="out"
      · right
        intro k
        exact (outside (0 : Fin 1) k trivial).resolve_left (by simpa only [rowIO] using not_not_intro hr)
      · exact Or.inl hr

#axiomsClean implements
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormImplementation.Interface
