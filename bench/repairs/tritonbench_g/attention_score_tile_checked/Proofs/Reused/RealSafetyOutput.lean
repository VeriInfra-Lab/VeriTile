import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealSafetyParts
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealProgram

/-! Safety of the exact masked output store. Only active columns require
allocated addresses; numerical output values do not occur in these premises. -/
namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealSafetyOutput
open VeriTile.Triton
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false
noncomputable section

theorem correct (bounds : RegionBounds) (s : BlockState) (Out : RegionName)
    (B NK soz soh batch head startN : Nat)
    (hz : s.regs .int [] "off_z" = some (Tile.scalar (batch : Int)))
    (hh : s.regs .int [] "off_h" = some (Tile.scalar (head : Int)))
    (hn : s.regs .int [] "start_n" = some (Tile.scalar (startN : Int)))
    (hb : ∀ i : Fin B, startN*B+i.val < NK → batch*soz+head*soh+startN*B+i.val < bounds Out) :
    Stmt.TraceSafeList bounds (RealProgram.postlude Out B NK soz soh) s := by
  let base := batch*soz+head*soh
  let so := s.setReg "o_offset" .int [] (Tile.scalar (base : Int))
  let sr := so.setReg "o_range" .int [B] (Tile.vec (fun i => ((i.val+startN*B : Nat) : Int)))
  let sp := sr.setReg "o_ptrs" .ptr [B] (Tile.vec (fun i => (Out, base+startN*B+i.val)))
  have eo : evalOp (.add .int .nil
      (.mul .int .nil (.ref .int [] "off_z") (.constInt soz))
      (.mul .int .nil (.ref .int [] "off_h") (.constInt soh))) s = some (Tile.scalar (base : Int)) := by
    simp only [evalOp.eq_def, hz, hh, bind, Option.bind]
    simp [Tile.bop, Tile.scalar, NumericDType.add, NumericDType.mul, Broadcast.leftIndex, Broadcast.rightIndex, base, -Tile.scalar_eta]
  have er : evalOp (.add .int .scalarR (.castNatToInt (.arange B))
      (.mul .int .nil (.ref .int [] "start_n") (.constInt B))) so =
      some (Tile.vec (fun i => ((i.val+startN*B : Nat) : Int))) := by
    simp [evalOp.eq_def, so, hn]
    simp [Tile.bop, Tile.scalar, Tile.vec, NumericDType.add, NumericDType.mul, Broadcast.leftIndex, Broadcast.rightIndex, -Tile.scalar_eta]
  have ep : evalOp (.ptrAdd .scalarL (.ptrBase Out)
      (.castIntToNat (.add .int .scalarL (.ref .int [] "o_offset") (.ref .int [B] "o_range")))) sr =
      some (Tile.vec (fun i => (Out, base+startN*B+i.val))) := by
    simp [evalOp.eq_def, sr, so, Tile.bop, Tile.ptrAdd, Tile.scalar, Tile.vec,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, Nat.add_assoc,
      Nat.add_left_comm, Nat.add_comm, -Tile.scalar_eta, ← Int.natCast_add, ← Int.natCast_mul, -Nat.cast_add, -Nat.cast_mul]
  have em : evalOp (.lt .int .scalarR (.ref .int [B] "o_range") (.castNatToInt (.constNat NK))) sp =
      some (Tile.vec (fun i => decide (startN*B+i.val < NK))) := by
    simp [evalOp.eq_def, sp, sr, Tile.cop, Tile.scalar, Tile.vec, ComparableDType.lt,
      Broadcast.leftIndex, Broadcast.rightIndex, Nat.add_comm, -Tile.scalar_eta,
      ← Int.natCast_add, ← Int.natCast_mul, -Nat.cast_add, -Nat.cast_mul]
  rw [RealProgram.postlude, TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some eo)]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some er)]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some ep)]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
  apply Stmt.TraceSafeList.cons_intro
  · change Stmt.TraceSafe bounds (.store .real [B] (.ptr (.ref .ptr [B] "o_ptrs"))
      (.ref .real [B] "o") (.mask (.lt .int .scalarR (.ref .int [B] "o_range") (.castNatToInt (.constNat NK))))) sp
    simp only [Stmt.TraceSafe, MemAccess.SafeAt, Op.SafeAt.eq_def, MaskOpt.SafeAt, true_and]
    unfold MemAccess.ActiveAddressSafe memAccessActiveAddressSafe Op.PointerAddressesSafeOn
    refine ⟨by simp [Op.MemorySafe], ?_⟩
    intro ptrs hp i active
    have hp' : evalOp (.ref .ptr [B] "o_ptrs") sp =
        some (Tile.vec (fun i => (Out, base+startN*B+i.val))) := by simp [sp]
    rw [hp'] at hp
    cases hp
    obtain ⟨mask,hm,hi⟩ := active
    rw [em] at hm
    cases hm
    simp only [Tile.vec_data, decide_eq_true_eq] at hi
    exact hb i.1 hi
  · intro _ _; exact Stmt.TraceSafeList.nil_intro

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealSafetyOutput
