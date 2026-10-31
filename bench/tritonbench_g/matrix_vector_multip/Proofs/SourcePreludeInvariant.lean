import bench.tritonbench_g.matrix_vector_multip.Proofs.SourceRuntimeOnly
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatrixVectorMultip.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

structure RuntimeInvariant (p : Parameters) (initial : State α) (off : Nat)
    (acc : WholeSource.Acc α p) (current : State α) : Prop where
  context : WholeSource.Context p (initial.pids 0) off initial.mem acc current
  pids : current.pids=initial.pids
  metadata : ∀ name n, (name,n) ∈ scalarParameters p → current.regs .nat [] name=some (fun _ => n)
  counter : AffineWhile.NormalizedCounter current "m" off

theorem preludeInvariant (A : Algebra α) (p : Parameters) (s : State α)
    (small : ∀ name n, (name,n) ∈ scalarParameters p → n<2^64)
    (pidSmall : s.pids 0<2^64) (rowsSmall : p.blockRows≤2^64) (colsSmall : p.blockColumns≤2^64) :
    RuntimeInvariant p s 0 (fun _ => A.literal (some .fp32) .real 0) (preludeExecution A p s).val := by
  have modN := Nat.mod_eq_of_lt (small "N" p.rows (by simp [scalarParameters]))
  have modM := Nat.mod_eq_of_lt (small "M" p.columns (by simp [scalarParameters]))
  have modAN := Nat.mod_eq_of_lt (small "stride_an" p.strideRow (by simp [scalarParameters]))
  have modAM := Nat.mod_eq_of_lt (small "stride_am" p.strideColumn (by simp [scalarParameters]))
  have modBM := Nat.mod_eq_of_lt (small "stride_bm" p.strideVector (by simp [scalarParameters]))
  have modCN := Nat.mod_eq_of_lt (small "stride_cn" p.strideOut (by simp [scalarParameters]))
  norm_num only [Nat.reducePow] at modN modM modAN modAM modBM modCN
  have modPid : s.pids 0 % 18446744073709551616=s.pids 0 := Nat.mod_eq_of_lt pidSmall
  have laneN (i : Fin p.blockRows) : i.val % 18446744073709551616=i.val :=
    Nat.mod_eq_of_lt (lt_of_lt_of_le i.isLt rowsSmall)
  have laneM (i : Fin p.blockColumns) : i.val % 18446744073709551616=i.val :=
    Nat.mod_eq_of_lt (lt_of_lt_of_le i.isLt colsSmall)
  refine ⟨?_,prelude_pids A p s,?_,prelude_counter A p s⟩
  · refine ⟨?_,?_,?_,?_,?_,?_,prelude_memory A p s⟩
    all_goals simp [preludeExecution,WholeSource.row,WholeSource.aOffset,WholeSource.bOffset,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Int.toNat_emod,
      modN,modM,modAN,modAM,modBM,modCN,modPid,laneN,laneM,Nat.add_comm]
    all_goals funext i
    all_goals simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,modPid,laneN,laneM,Nat.add_comm]
  · intro name n member
    simpa only [Nat.mod_eq_of_lt (small name n member)] using prelude_metadata A p s name n member

#axiomsClean preludeInvariant
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.Source
