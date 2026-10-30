import bench.tritonbench_g.flash_attn.Proofs.FlatReal
import VeriTile.Triton.Memory.TraceRewrite
import VeriTile.Triton.Memory.FlattenFrame

namespace VeriTile.Bench.TritonBenchG.FlashAttn.Contracts
open VeriTile.Triton Formula RealExecution
noncomputable section

/-- Exact-real region IO with the independent two-output attention formula.
The reference scatters L then O in source order, retaining output aliasing. -/
def RegionIO (kernel : Kernel) (p : Layout) (s : BlockState)
    (Q K V L O : RegionName) : Prop :=
  ∃ t, exec kernel s = some t ∧
    t.mem = (referenceState p s Q K V L O).mem ∧
    ∀ r o, ¬writeWindow p L O (s.pids 0) (s.pids 1) r o → t.mem r o = s.mem r o

/-- The historical full-source flat result, retaining its region-state
witness and region frame. `FlatIO` additionally exposes actual source safety
and the complete flat-memory frame. -/
def RegionFlatIO (kernel : Kernel) (A : FlatAlloc) (p : Layout) (s : BlockState)
    (Q K V L O : RegionName) : Prop :=
  ∃ t, exec (A.flattenKernel kernel) (A.flattenState s) = some (A.flattenState t) ∧
    t.mem = (referenceState p s Q K V L O).mem ∧
    ∀ r o, ¬writeWindow p L O (s.pids 0) (s.pids 1) r o → t.mem r o = s.mem r o

/-- Complete source-real IO for both attention outputs: actual trace safety,
successful finite-flat execution, an independent initial-memory formula and
the raw-cell frame, including unallocated addresses and unrelated dtypes. -/
def FlatIO (kernel : Kernel) (A : FlatAlloc) (p : Layout) (s : BlockState)
    (Q K V L O : RegionName) : Prop :=
  kernel.TraceSafe A.extent s ∧
  ∃ t, exec (A.flattenKernel kernel) (A.flattenState s) = some t ∧
    t.mem = (A.flattenState (referenceState p s Q K V L O)).mem ∧
    ∀ r o, (r ≠ A.flat ∨ ∀ region offset,
      writeWindow p L O (s.pids 0) (s.pids 1) region offset → o ≠ A.addr region offset) →
      t.mem r o = (A.flattenState s).mem r o

theorem index_safe (swapped : Bool) (stride : Nat) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds [Candidate.indexStatement swapped stride] s := by
  cases swapped <;> simp [Candidate.indexStatement, Stmt.TraceSafeList,
    Stmt.TraceSafe, Op.SafeAt.eq_def]
  all_goals split <;> trivial

/-- The natural index normalization preserves each reached access, as well
as the complete execution result used to reach subsequent statements. -/
theorem source_safe_iff (Q K V L O : RegionName) (p : Layout)
    (bounds : RegionBounds) (s : BlockState) :
    (Candidate.source Bool.false Q K V L O p).TraceSafe bounds s ↔
      (Candidate.source Bool.true Q K V L O p).TraceSafe bounds s := by
  change Stmt.TraceSafeList bounds (Candidate.source Bool.false Q K V L O p).body s ↔
    Stmt.TraceSafeList bounds (Candidate.source Bool.true Q K V L O p).body s
  rw [Candidate.decomposition Bool.false, Candidate.decomposition Bool.true]
  simp only [TraceRewrite.append_iff, ProgramRewrite.append,
    Candidate.index_exec, index_safe]

theorem source_safe (swapped : Bool) (Q K V L O : RegionName) (p : Layout)
    (s : BlockState) (bounds : RegionBounds)
    (hb : AddressBounds p s Q K V L O bounds) (hBN : 0 < p.BLOCK_N) :
    (Candidate.source swapped Q K V L O p).TraceSafe bounds s := by
  have h : (Candidate.source Bool.false Q K V L O p).TraceSafe bounds s := by
    change (NormalizedSource.original Q K V L O p).TraceSafe bounds s
    rw [NormalizedSource.original_eq_algorithm]
    exact algorithm_safe Q K V L O p s bounds hb hBN
  cases swapped
  · exact h
  · exact (source_safe_iff Q K V L O p bounds s).mp h

theorem full_flat (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (swapped : Bool) (Q K V L O : RegionName) (p : Layout) (s : BlockState)
    (hb : AddressBounds p s Q K V L O A.extent)
    (hundef : s.undef = fun _ _ => 0) (hBN : 0 < p.BLOCK_N)
    (hhi : 0 < hi p (s.pids 0)) :
    FlatIO (Candidate.source swapped Q K V L O p) A p s Q K V L O := by
  refine ⟨source_safe swapped Q K V L O p s A.extent hb hBN, ?_⟩
  obtain ⟨t, he, hm, hf⟩ := source_flat_correct A hd hcov swapped Q K V L O p s hb hundef hBN hhi
  refine ⟨A.flattenState t, he, ?_, ?_⟩
  · funext r o
    exact A.flattenState_mem_congr (fun r o => congrFun (congrFun hm r) o) r o
  · exact A.flattenState_frame s t (writeWindow p L O (s.pids 0) (s.pids 1)) hf

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FlashAttn.Contracts
