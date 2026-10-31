import bench.tritonbench_g.attention_score.Proofs.Reused.NativeMath

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.NativeColumn
open VeriTile Triton FP.Structural NativeInstructions NativeStaging NativeIdeal Geometry
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def column (decode : BitVec 32 → ℝ) (B D : Nat) (p : RealSource.Parameters) (s : State Expr)
    (blocks : Nat) (j : Fin B) : ℝ :=
  RealFormula.column (NativeMath.inputs decode D p s) p.scale p.queryContext p.keyContext B blocks
    p.window p.complement p.evenM p.evenN p.offset p.size (s.pids 0*B+j.val)

theorem fold_column (decode : BitVec 32 → ℝ) (hz : decode 0=0) (B D : Nat) (hB : Supported B)
    (p : RealSource.Parameters) (s : State Expr) (keyRange : s.pids 0*B<2147483648) (remaining done stop : Nat) (acc : Fin B → Graph)
    (hacc : ∀ j, (acc j).eval (isa decode)=column decode B D p s done j)
    (hupper : stop≤(done+remaining)*B)
    (hlower : done+remaining=0 ∨ (done+remaining-1)*B<stop) :
    ∀ j, (foldRange (NativeFormula.update B D p s) (done*B) stop B acc j).eval (isa decode)=
      column decode B D p s (done+remaining) j := by
  have hpositive := supported_positive B hB
  induction remaining generalizing done acc with
  | zero =>
      intro j
      have hnot : ¬done*B<stop := Nat.not_lt.mpr (by simpa using hupper)
      simp [foldRange,Nat.ne_of_gt hpositive,hnot,hacc]
  | succ remaining ih =>
      have hlast : (done+remaining)*B<stop := by
        rcases hlower with hz|hl
        · omega
        · simpa only [Nat.add_succ,Nat.add_sub_cancel] using hl
      have hlt : done*B<stop := lt_of_le_of_lt (Nat.mul_le_mul_right B (Nat.le_add_right done remaining)) hlast
      have hnext : ∀ j, (NativeFormula.update B D p s (done*B) acc j).eval (isa decode)=
          column decode B D p s (done+1) j := by
        intro j
        change (acc j).eval (isa decode)+(NativeFormula.contribution B D p s (done*B) j).eval (isa decode)=_
        rw [hacc,NativeMath.contribution_eval decode hz _ _ hB p s _ j keyRange]
        exact (RealFormula.column_succ _ _ _ _ _ _ _ _ _ _ _ _ _).symm
      have next := ih (done+1) (NativeFormula.update B D p s (done*B) acc) hnext
        (by simpa [Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using hupper)
        (by simpa [Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using hlower)
      have hf : foldRange (NativeFormula.update B D p s) (done*B) stop B acc=
          foldRange (NativeFormula.update B D p s) (done*B+B) stop B (NativeFormula.update B D p s (done*B) acc) := by
        conv_lhs => rw [foldRange]
        simp [Nat.ne_of_gt hpositive,hlt]
      intro j
      rw [hf]
      simpa [Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using next j

theorem accumulated_eval (decode : BitVec 32 → ℝ) (hz : decode 0=0) (B D : Nat) (hB : Supported B)
    (p : RealSource.Parameters) (s : State Expr) (keyRange : s.pids 0*B<2147483648) (j : Fin B) :
    (NativeFormula.accumulated B D p s j).eval (isa decode)=
      column decode B D p s (RealLoop.tripCount p.queryContext B) j := by
  obtain ⟨hu,hl⟩ := RealLoop.tripCount_bounds p.queryContext B (supported_positive B hB)
  simpa [NativeFormula.accumulated] using fold_column decode hz B D hB p s keyRange
    (RealLoop.tripCount p.queryContext B) 0 p.queryContext (fun _ => NativeFormula.zero32)
    (by intro j; simp [column,RealFormula.column,NativeFormula.zero32,Graph.eval,isa])
    (by simpa using hu) (by simpa using hl) j

theorem logical_column (decode : BitVec 32 → ℝ) (hz : decode 0=0) (B D : Nat) (hB : Supported B)
    (p : RealSource.Parameters) (s : State Expr) (keyRange : s.pids 0*B<2147483648) (j : Fin B)
    (hj : s.pids 0*B+j.val<p.keyContext) (he : p.evenM=Bool.true → p.queryContext%B=0) :
    (NativeFormula.accumulated B D p s j).eval (isa decode)=
      LogicalFormula.column (NativeMath.inputs decode D p s) p.scale p.queryContext
        p.window p.complement p.offset p.size (s.pids 0*B+j.val) := by
  rw [accumulated_eval decode hz _ _ hB p s keyRange]
  exact LogicalFormula.repaired_column _ _ _ _ _ _ _ _ _ _ _ _ (supported_positive B hB) hj he

#axiomsClean logical_column
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.NativeColumn
