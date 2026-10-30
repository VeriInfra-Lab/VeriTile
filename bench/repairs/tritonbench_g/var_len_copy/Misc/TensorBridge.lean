import bench.repairs.tritonbench_g.var_len_copy.Proofs.KernelRealExecution
import bench.repairs.tritonbench_g.var_len_copy.Proofs.Contract
import bench.repairs.tritonbench_g.var_len_copy.Proofs.KernelRealSafetySource

/-! Optional correspondence between Python checks and the kernel input contract. -/
namespace VeriTile.Bench.Repairs.VarLenCopy.TensorBridge
open VeriTile.Triton
noncomputable section

/-- The complete shared kernel copies each requested active element from the
initial source and preserves every cell outside this program's destination
segment. Nonpositive lengths admit arbitrary signed starts and perform no writes. -/
theorem var_len_copy_real (m : Inputs) (segments : Nat → Segment)
    (h : InputContract m segments) (s : BlockState)
    (binding : KernelMetadataBinding m segments s)
    (hp : s.pids 0 < (m .oldStarts).numel) :
    Spec.Real (∃ final,
      exec Source.originalKernel.eraseDType.toAlgKernel s = some final ∧
      KernelCopy (segments (s.pids 0)) (s.readMem "source") (final.readMem "output") ∧
      (∀ (r : RegionName) i, r ≠ "output" ∨ i < (segments (s.pids 0)).dest ∨
        (segments (s.pids 0)).dest + (segments (s.pids 0)).len ≤ i →
        final.mem r i = s.mem r i)) := by
  apply RealExecution.var_len_copy_kernel_real (segments (s.pids 0)) s (binding.lengthCells _ hp)
  intro positive
  constructor
  · rw [binding.oldCells _ hp]
    exact (Int.toNat_of_nonneg (h.sourceBound _ hp positive).1).symm
  · rw [binding.newCells _ hp]
    exact (Int.toNat_of_nonneg (h.outputBound _ hp positive).1).symm

/-- Every actual metadata, source and destination access is within the
corresponding admitted buffer extent, including tail and nonpositive branches. -/
theorem kernel_trace_safe (m : Inputs) (segments : Nat → Segment)
    (h : InputContract m segments) (s : BlockState)
    (binding : KernelMetadataBinding m segments s)
    (hp : s.pids 0 < (m .oldStarts).numel) :
    Source.originalKernel.eraseDType.toAlgKernel.TraceSafe (kernelBounds m) s := by
  let a := segments (s.pids 0)
  have hl := binding.lengthCells _ hp
  by_cases positive : 0 < a.len
  · have old := h.sourceBound _ hp positive
    have dest := h.outputBound _ hp positive
    have ho : s.readMemValue .int "old" (s.pids 0) = (a.old : Int) := by
      rw [binding.oldCells _ hp]
      exact (Int.toNat_of_nonneg old.1).symm
    have hn : s.readMemValue .int "new" (s.pids 0) = (a.dest : Int) := by
      rw [binding.newCells _ hp]
      exact (Int.toNat_of_nonneg dest.1).symm
    exact KernelReal.source_safe (kernelBounds m) s ⟨"old"⟩ ⟨"length"⟩ ⟨"new"⟩
      "source" "output" 256 a.length.toInt a.old a.dest (by decide) (by decide) hl ho hn
      (by simpa [kernelBounds] using hp) (by simpa [kernelBounds] using hp)
      (by simpa [kernelBounds] using hp) (by simpa [kernelBounds,a,Segment.len] using old.2)
      (by simpa [kernelBounds,a,Segment.len] using dest.2)
  · have nonpositive : a.length.toInt ≤ 0 := by
      change ¬0 < a.length.toInt.toNat at positive
      omega
    exact KernelReal.source_empty_safe (kernelBounds m) s ⟨"old"⟩ ⟨"length"⟩ ⟨"new"⟩
      "source" "output" 256 a.length.toInt (by decide) nonpositive hl
      (by simpa [kernelBounds] using hp) (by simpa [kernelBounds] using hp)
      (by simpa [kernelBounds] using hp)


/-- Exact guarded admission; there is no payload copy or numerical guard. -/
theorem python_checks_iff (m : Option Inputs) (segments : Nat → Segment) :
    checkArguments m segments = .ok () ↔ AdmittedArguments m segments :=
  checkArguments_iff m segments

#axiomsClean var_len_copy_real
#axiomsClean kernel_trace_safe
#axiomsClean python_checks_iff
end
end VeriTile.Bench.Repairs.VarLenCopy.TensorBridge
