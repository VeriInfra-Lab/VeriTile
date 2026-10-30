import bench.tritonbench_g.chunk_delta_fwd.Sources.Model
namespace VeriTile.Bench.Repairs.ChunkDeltaFwd
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 6000000

theorem foldRange_list {β : Type} (f : Nat → β → β) (start count : Nat) (a : β) :
    foldRange f start (start+count) 1 a =
      (List.range' start count).foldl (fun acc n => f n acc) a := by
  induction count generalizing start a with
  | zero => simp [foldRange]
  | succ n ih =>
    rw [foldRange]
    simp only [show (1 : Nat)≠0 from by decide,if_false,
      show start<start+(n+1) by omega,if_true,List.range'_succ,List.foldl_cons]
    simpa only [Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using
      ih (start+1) (f start a)

theorem foldRange_zero {β : Type} (f : Nat → β → β) (a : β) : foldRange f 0 0 1 a=a := by
  simp [foldRange]

theorem foldRange_succ {β : Type} (f : Nat → β → β) (n : Nat) (a : β) :
    foldRange f 0 (n+1) 1 a=f n (foldRange f 0 n 1 a) := by
  have h0 := foldRange_list f 0 n a
  have h1 := foldRange_list f 0 (n+1) a
  simp only [Nat.zero_add] at h0 h1
  rw [h0,h1,List.range'_1_concat,List.foldl_append]
  simp

/-- Index-aware source-loop induction. Execution of each actual body is proved
by its caller; the conclusion fixes the terminal index, including zero loops. -/
theorem range_invariant {α : Type} [Inhabited α] (M : Algebra α)
    (idx : RegName) (body : List ComputeStmt) (stop : Nat)
    (I : Nat → State α → Prop)
    (hstep : ∀ n, n<stop → ∀ s, I n s →
      ∃ t, run M body (s.setReg idx .nat [] (fun _ => n))=some t ∧ I (n+1) t)
    (cur : Nat) (hc : cur≤stop) (s : State α) (hi : I cur s) :
    ∃ t, FP.Structural.range M idx cur stop 1 body s=some t ∧ I stop t := by
  by_cases h : cur<stop
  · obtain ⟨next,he,hn⟩ := hstep cur h s hi
    obtain ⟨final,hf,hd⟩ := range_invariant M idx body stop I hstep (cur+1) (by omega) next hn
    refine ⟨final,?_,hd⟩
    rw [FP.Structural.range]
    simpa only [show (1 : Nat)≠0 from by decide,if_false,h,if_true,he,Option.bind_some] using hf
  · have eq : cur=stop := by omega
    subst cur
    exact ⟨s,by simp [FP.Structural.range],hi⟩
termination_by stop-cur
decreasing_by omega

#axiomsClean foldRange_succ
#axiomsClean range_invariant
end
end VeriTile.Bench.Repairs.ChunkDeltaFwd
