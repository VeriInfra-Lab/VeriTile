import bench.tritonbench_g.batched_vecmat_mult.IO

namespace VeriTile.Bench.Repairs.BatchedVecmatMult.IOSchedule
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
open SourceGraph (Expr)
open VecmatIO
noncomputable section
set_option maxHeartbeats 2000000

theorem accumulated_congr (plans : SourceGraph.PlanSelector) (p : Parameters)
    (s t : State Expr) (hp : s.pids=t.pids) (hm : s.mem=t.mem)
    (c : Nat) (i : TileIndex [p.BM,p.BN]) :
    SourceGraph.accumulated plans p s c i=SourceGraph.accumulated plans p t c i := by
  induction c with
  | zero => rfl
  | succ c ih =>
    simp only [SourceGraph.accumulated,ih,SourceGraph.blockSum,SourceGraph.aOffset,
      SourceGraph.bOffset,SourceGraph.row,SourceGraph.col,hp,hm]

theorem memory_congr (plans : SourceGraph.PlanSelector) (p : Parameters)
    (s t : State Expr) (hp : s.pids=t.pids) (hm : s.mem=t.mem) :
    SourceGraph.expectedMemory plans p s=SourceGraph.expectedMemory plans p t := by
  simp only [SourceGraph.expectedMemory,SourceGraph.outputOffset,
    SourceGraph.row,SourceGraph.col,hp,hm,accumulated_congr plans p s t hp hm]
  rfl

/-- Run the actual standard invocations; reuse the existing complete source
program result, identifying its independent memory recurrence at each launch. -/
theorem run_prefix (rest : Algebra Expr) (profile : KernelProfile.Selector)
    (p : Parameters) (initial : State Expr) (k : Nat) :
    ∃ final, FP.Structural.runSchedule ((List.range k).map (launch rest profile p)) initial=some final ∧
      final.mem=SourceGraph.referenceGridPrefix (KernelProfile.toPlans profile) p initial k := by
  induction k with
  | zero => exact ⟨initial,rfl,rfl⟩
  | succ k ih =>
    obtain ⟨previous,ran,hm⟩ := ih
    obtain ⟨final,executed,result,_⟩ := SourceGraph.complete_correct rest (KernelProfile.toPlans profile) p
      ((launch rest profile p k).initial previous)
    refine ⟨final,?_,?_⟩
    · rw [List.range_succ,List.map_append,runSchedule_append,ran]
      simpa only [Option.bind_some,List.map_singleton,runSchedule_singleton,Invocation.run] using executed
    · rw [result]
      apply memory_congr
      · rfl
      · exact hm

theorem grid (rest : Algebra Expr) (profile : KernelProfile.Selector)
    (p : Parameters) (initial : State Expr) :
    ∃ final, FP.Structural.runSchedule (schedule rest profile p) initial=some final ∧
      final.mem=SourceGraph.gridExpectedMemory (KernelProfile.toPlans profile) p initial :=
  run_prefix rest profile p initial (SourceGraph.gridCount p)

#axiomsClean grid
end
end VeriTile.Bench.Repairs.BatchedVecmatMult.IOSchedule
