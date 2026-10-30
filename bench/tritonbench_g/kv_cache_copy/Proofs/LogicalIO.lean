import bench.tritonbench_g.kv_cache_copy.IO
import bench.tritonbench_g.kv_cache_copy.Proofs.IOCorrect

namespace VeriTile.Bench.Repairs.KvCacheCopy.LogicalIO
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

theorem judgment_iff (rest : Algebra ℝ) (p : Source.Parameters) (seqPid headPid : Nat) :
    (io rest p seqPid headPid ⊨{fun a => CopyIO.OutputUnique p headPid a.length a.block}
      fun a => ⟨a.key,a.value⟩) ↔
    (CopyIO.io rest p seqPid headPid).ImplementsOn (CopyIO.domain rest p seqPid headPid)
      (CopyIO.reference rest p seqPid headPid) := by
  change (CopyIO.io rest p seqPid headPid).ImplementsOn _ _ ↔ _
  apply Iff.of_eq
  congr 1
  funext a o j
  fin_cases o <;> rfl

theorem implements (rest : Algebra ℝ) (p : Source.Parameters) (seqPid headPid : Nat) :
    io rest p seqPid headPid ⊨{fun a => CopyIO.OutputUnique p headPid a.length a.block}
      fun a => ⟨a.key,a.value⟩ :=
  (judgment_iff rest p seqPid headPid).mpr (IOCorrect.correct rest p seqPid headPid)
end
end VeriTile.Bench.Repairs.KvCacheCopy.LogicalIO
