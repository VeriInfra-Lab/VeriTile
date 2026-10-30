import bench.repairs.tritonbench_g.var_len_copy.Contract
import bench.repairs.tritonbench_g.var_len_copy.Proofs.KernelRealIO
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.VarLenCopy.RealExecution
open VeriTile.Triton
noncomputable section

/-- Complete source execution on raw signed metadata and raw regional
memory. Active reads name the requested spans; nonpositive lengths do not
constrain unused signed starts and preserve every cell. -/
theorem var_len_copy_kernel_real (a : Segment) (s : BlockState)
    (hl : s.readMemValue .int "length" (s.pids 0)=a.length.toInt)
    (addresses : 0<a.len →
      s.readMemValue .int "old" (s.pids 0)=(a.old : Int) ∧
      s.readMemValue .int "new" (s.pids 0)=(a.dest : Int)) :
    Spec.Real (∃ final,
      exec Source.originalKernel.eraseDType.toAlgKernel s = some final ∧
      KernelCopy a (s.readMem "source") (final.readMem "output") ∧
      (∀ (r : RegionName) i, r ≠ "output" ∨ i < a.dest ∨
        a.dest+a.len ≤ i → final.mem r i = s.mem r i)) := by
  by_cases positive : 0 < a.len
  · have ho := (addresses positive).1
    have hn := (addresses positive).2
    obtain ⟨final,he,hf,hv,_,_,_⟩ := KernelReal.source_io s ⟨"old"⟩ ⟨"length"⟩ ⟨"new"⟩
      "source" "output" 256 a.length.toInt a.old a.dest (by decide) (by decide) hl ho hn
    refine ⟨final,he,hv,?_⟩
    intro r i outside
    change r ≠ "output" ∨ i < a.dest ∨ a.dest+a.len ≤ i at outside
    symm
    apply hf r i
    rintro ⟨rfl,n,hnb,rfl⟩
    change n < a.len at hnb
    rcases outside with outside | outside | outside
    · exact outside rfl
    · omega
    · omega
  · have nonpositive : a.length.toInt ≤ 0 := by
      change ¬0 < a.length.toInt.toNat at positive
      omega
    obtain ⟨final,he,hf⟩ := KernelReal.source_empty s ⟨"old"⟩ ⟨"length"⟩ ⟨"new"⟩
      "source" "output" 256 a.length.toInt (by decide) nonpositive hl
    refine ⟨final,he,?_,?_⟩
    · intro i hi
      exact False.elim (positive (Nat.zero_lt_of_lt hi))
    · intro r i _
      exact congrFun (congrFun hf r) i

#specNonCircular KernelCopy avoiding [Source.originalKernel,Source.kernel,exec]
#axiomsClean var_len_copy_kernel_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.VarLenCopy.RealExecution
