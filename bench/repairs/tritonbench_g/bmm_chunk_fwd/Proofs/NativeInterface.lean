import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.NativeMemoryConnection
import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.NativeScheduleSafety

namespace VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO.Native
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open KernelInterface NativeGraph
noncomputable section
open Classical
set_option maxHeartbeats 2000000

theorem frame_of_patch (rest : Algebra Expr) (p : Parameters) (c : Host.Config) (v : Values p)
    (initial final : State Expr) (memory : final.mem=patch p initial.mem (complete p c v)) :
    (io rest p c).Frame v initial final := by
  intro r off outside
  rw [memory]
  by_cases same : r=p.Out
  · have noWrite : ¬writeMask p c off := by
      intro written
      rcases outside.1 (0 : Fin 1) (⟨off,write_bound p c off written⟩,PUnit.unit) written with region|offset
      · exact region same
      · exact offset rfl
    simp only [patch,same,↓reduceIte,complete_frame p c v off noWrite]
  · simp [patch,same]

theorem readback_of_patch (rest : Algebra Expr) (p : Parameters) (c : Host.Config) (v : Values p)
    (initial final : State Expr) (reads : Reads p c v initial.mem)
    (memory : final.mem=patch p initial.mem (complete p c v)) :
    (io rest p c).OutputReadback v (formula p c v) final := by
  intro o i _
  fin_cases o
  change (final.mem p.Out i.1.val).read .fp32=
    (complete p c v i.1.val).getD (oldOutput p v i.1.val)
  rw [memory]
  simp only [patch,↓reduceIte]
  cases value : complete p c v i.1.val
  · exact reads.out i.1.val i.1.isLt
  · rfl

/-- Existing complete execution and independent memory-reference proofs provide
the shared source IO output and exact exterior frame. All configurations,
arbitrary tagged memories and allowed aliases remain admitted. -/
theorem results (rest : Algebra Expr) (p : Parameters) (c : Host.Config) (v : Values p)
    (initial : State Expr) (binding : (io rest p c).InputBinding v initial) :
    ∃ final, runSchedule (schedule rest p c Bool.false) initial=some final ∧
      runSchedule (schedule rest p c Bool.true) initial=some final ∧
      (io rest p c).OutputReadback v (formula p c v) final ∧
      (io rest p c).Frame v initial final := by
  obtain ⟨final,source,candidate,memory⟩ := successful rest p c initial
  have reads := reads_from_binding rest p c v initial binding
  have patchMemory := memory.trans (kernel_memory p c v initial.mem reads)
  exact ⟨final,source,candidate,readback_of_patch rest p c v initial final reads patchMemory,
    frame_of_patch rest p c v initial final patchMemory⟩

theorem native_implements (rest : Algebra Expr) (p : Parameters) (c : Host.Config) :
    (io rest p c).Implements (formula p c) := by
  intro vals _ bounds bounded initial binding
  obtain ⟨final,source,_,readback,frame⟩ := results rest p c vals initial binding
  exact ⟨all_safe rest p c vals bounds bounded.1 Bool.false initial,final,source,readback,frame⟩

theorem native_equivalent (rest : Algebra Expr) (p : Parameters) (c : Host.Config) :
    (io rest p c).toProgramIO.Equivalent (candidate rest p c) := by
  intro vals _ bounds bounded initial binding
  obtain ⟨final,source,candidateRun,_,frame⟩ := results rest p c vals initial binding
  refine ⟨all_safe rest p c vals bounds bounded Bool.false initial,
    all_safe rest p c vals bounds bounded Bool.true initial,final,final,source,candidateRun,?_,frame,frame⟩
  intro o i active
  rfl

end
end VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO.Native
