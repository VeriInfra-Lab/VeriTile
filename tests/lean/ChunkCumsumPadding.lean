import bench.tritonbench_g.chunk_cumsum_kernel.Proofs.RealComplete

/-! Actual repair and current candidate: three supported BT=16 chunks with a
partial final chunk, nonzero PID and arbitrary unspecified values. The full
exterior memory frame also preserves neighboring non-real cells. -/
namespace ChunkCumsumPaddingTests
open VeriTile.Triton
open VeriTile.Bench.Repairs.ChunkCumsumKernel
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 1000000

private def initial (padding : ℝ) : BlockState where
  mem := fun r _ => if r="o" then .of .nat 99 else .real 1
  regs := fun _ _ _ => none
  pids := fun _ => 1
  undef := fun _ _ => padding

theorem complete_outputs (swapped : Bool) (padding : ℝ) :
    ∃ v,exec (RealIndexBridge.sourceFor swapped "s" "o" 33 16).eraseDType.toAlgKernel
      (initial padding)=some v ∧
      (∀ i : Fin 33,v.readMem "o" (33+i.val)=(i.val+1:Nat)) ∧
      (∀ (r : RegionName) (off : Nat),r≠"o" ∨ off<33 ∨ 66≤off →
        v.mem r off=(initial padding).mem r off) := by
  obtain ⟨v,ran,values,frame⟩ := RealIndexBridge.complete_blocks swapped "s" "o" 33 16
    (initial padding) (by decide) (by decide)
  refine ⟨v,ran,?_,?_⟩
  · intro i
    have filter : (Finset.range 33).filter (fun j => j ≤ i.val)=Finset.range (i.val+1) := by
      ext j
      simp only [Finset.mem_filter,Finset.mem_range]
      have inside := i.isLt
      omega
    have value := values i.val i.isLt
    simpa [initial,filter,BlockState.readMem,Region.mk.injEq] using value
  · intro r off outside
    apply frame r off
    rcases outside with different|before|after
    · exact Or.inl different
    · right;intro i hi;change off≠33+i;omega
    · right;intro i hi;change off≠33+i;omega

theorem nonzero_padding_outputs (swapped : Bool) :
    ∃ v,exec (RealIndexBridge.sourceFor swapped "s" "o" 33 16).eraseDType.toAlgKernel
      (initial 7)=some v ∧
      (∀ i : Fin 33,v.readMem "o" (33+i.val)=(i.val+1:Nat)) ∧
      v.mem "o" 32=.of .nat 99 := by
  obtain ⟨v,ran,values,frame⟩ := complete_outputs swapped 7
  exact ⟨v,ran,values,(frame "o" 32 (Or.inr (Or.inl (by decide)))).trans rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end ChunkCumsumPaddingTests
