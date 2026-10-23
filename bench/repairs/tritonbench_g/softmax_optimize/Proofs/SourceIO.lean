import bench.repairs.tritonbench_g.softmax_optimize.Proofs.Reduction
import bench.repairs.tritonbench_g.softmax_optimize.Proofs.Assignments
import bench.repairs.tritonbench_g.softmax_optimize.Proofs.SecondLoops

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.SourceIO
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SoftmaxOptimize
open scoped BigOperators
set_option maxHeartbeats 5000000
noncomputable section

/-- Complete successful execution of the actual repaired typed source under
its independent Real interpretation. Both input loops, the max/sum reduction,
both output loops and every store are constructed. No execution, output or
numerical-correctness premise is used. The frame preserves raw memory cells. -/
theorem kernel_io (Y X : RegionName) (M N B : Nat) (hYX : Y ≠ X)
    (hN : 0 < N) (hB : 0 < B) (initial : BlockState) :
    ∃ finalState, exec (kernel Y X M N B).eraseDType.toAlgKernel initial = some finalState ∧
      (∀ j : Fin N, finalState.readMem Y (initial.pid * N + j.val) =
        softmaxOptimizeFullSpec initial X N j) ∧
      RowFrame Y N initial finalState ∧ finalState.pids = initial.pids ∧
      finalState.numPids = initial.numPids ∧ finalState.undef = initial.undef := by
  obtain ⟨streamed, hstreamed, hfirst⟩ := Reduction.first_pass initial X N B hN hB
  obtain ⟨shift, reduced, hreduce, hpids, hmem, hundef, hm, hz⟩ :=
    Reduction.reduction_run initial streamed X N B hN hB hfirst
  have hpidReg : reduced.regs .int [] "pid_m" = some (Tile.scalar (initial.pid : Int)) := by
    rw [Assignments.reduction_register B hreduce .int [] "pid_m" (by simp)]
    exact hfirst.pidReg
  have hcols : reduced.regs .int [] "cols" = some (Tile.scalar (N : Int)) := by
    rw [Assignments.reduction_register B hreduce .int [] "cols" (by simp)]
    exact hfirst.colsReg
  let total : ℝ := ∑ j ∈ Finset.range N, Real.exp (initial.readMem X (initial.pid * N + j))
  have hpid : streamed.pid = initial.pid := by unfold BlockState.pid; rw [hfirst.pids]
  have htotal : (∑ j ∈ Finset.range N, Real.exp (streamed.readMem X (streamed.pid * N + j))) = total := by
    unfold total
    apply Finset.sum_congr rfl
    intro j hj
    rw [hpid, hfirst.toStable.readMem]
  rw [htotal] at hz
  let prepared := reduced.setReg "prev_multiple" .int [] (Tile.scalar (Config.previous N B : Int))
  have hprepare : stepStmt (.assign .int [] "prev_multiple" (Source.previousExpression B)) reduced =
      some prepared := stepStmt_assign_eq_some (Reduction.previous_eval N B hN hB reduced hcols)
  have hsecond : SecondPass.Invariant Y X N B shift total initial 0 prepared := by
    refine ⟨⟨hpids.trans hfirst.pids,
      (Assignments.reduction_numPids B hreduce).trans hfirst.numPids,
      ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, ?_⟩
    · funext r o
      exact (hundef r o).trans (congrFun (congrFun hfirst.undef r) o)
    · simpa only [prepared, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
        not_false_eq_true] using hpidReg
    · simpa only [prepared, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
        not_false_eq_true] using hcols
    · simp only [prepared, BlockState.setReg_same]
    · simpa only [prepared, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
        not_false_eq_true] using hm
    · simpa only [prepared, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
        not_false_eq_true] using hz
    · intro r o outside
      change reduced.mem r o = initial.mem r o
      rw [hmem, hfirst.mem]
    · intro j hj
      omega
  obtain ⟨finalState, houtputs, hfinal⟩ :=
    SecondLoops.outputs_run Y X N B hYX hN hB shift total initial prepared hsecond
  refine ⟨finalState, ?_, ?_, hfinal.frame, hfinal.pids, hfinal.numPids, hfinal.undef⟩
  · unfold exec
    rw [Source.source_body]
    change stepStmts
      ((Reduction.seedStatements N B ++ [FirstLoops.fullStatement X N B, FirstLoops.tailStatement X N B]) ++
        (reductionStmts B ++ .assign .int [] "prev_multiple" (Source.previousExpression B) ::
          [SecondLoops.fullStatement Y X N B, SecondLoops.tailStatement Y X N B])) initial = some finalState
    rw [ProgramRewrite.append, hstreamed]
    simp only [Option.bind_some]
    rw [ProgramRewrite.append, hreduce]
    simp only [Option.bind_some]
    rw [stepStmts.cons_some hprepare]
    exact houtputs
  · intro j
    rw [softmaxOptimizeFullSpec_eq_div]
    exact hfinal.written j (lt_of_lt_of_le j.isLt (Config.tail_covers hN hB))

theorem selected_tile_io (Y X : RegionName) (M N : Nat) (hYX : Y ≠ X)
    (hN : 0 < N) (initial : BlockState) :
    ∃ finalState, exec (kernel Y X M N (Config.tile N)).eraseDType.toAlgKernel initial = some finalState ∧
      (∀ j : Fin N, finalState.readMem Y (initial.pid * N + j.val) =
        softmaxOptimizeFullSpec initial X N j) ∧
      RowFrame Y N initial finalState ∧ finalState.pids = initial.pids ∧
      finalState.numPids = initial.numPids ∧ finalState.undef = initial.undef :=
  kernel_io Y X M N (Config.tile N) hYX hN (Config.tile_positive N) initial

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxOptimize.SourceIO
