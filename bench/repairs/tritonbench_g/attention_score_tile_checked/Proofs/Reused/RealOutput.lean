import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealProgram

/-! The actual masked output store: active lanes receive their accumulated
column, inactive lanes and every cell outside the write window are preserved.
The source uses a unit inner output stride even though its signature also
accepts stride_on. -/
namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealOutput
open VeriTile.Triton
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false
noncomputable section

def writeOutput (s : BlockState) (Out : RegionName) (B NK base startN : Nat)
    (values : Fin B → ℝ) : BlockState :=
  (TileShape.allIndices [B]).foldl (fun st i =>
    if startN*B+i.1.val < NK then st.writeMem Out (base+startN*B+i.1.val) (values i.1) else st) s

theorem readback (s : BlockState) (Out : RegionName) (B NK base startN : Nat)
    (values : Fin B → ℝ) (i : Fin B) :
    (writeOutput s Out B NK base startN values).readMem Out (base+startN*B+i.val) =
      if startN*B+i.val < NK then values i else s.readMem Out (base+startN*B+i.val) := by
  exact BlockState.scatter_readback_prop_masked_nd (region := Out) (shape := [B]) s
    (fun j => base+startN*B+j.1.val) (fun j => values j.1)
    (fun j => startN*B+j.1.val < NK) (by
      intro a b hab
      dsimp only at hab
      exact Prod.ext (Fin.ext (by omega)) (by cases a.2; cases b.2; rfl)) (i,PUnit.unit)

theorem frame (s : BlockState) (Out : RegionName) (B NK base startN : Nat)
    (values : Fin B → ℝ) (r : RegionName) (a : Nat)
    (h : r ≠ Out ∨ ∀ i : Fin B, startN*B+i.val < NK → a ≠ base+startN*B+i.val) :
    (writeOutput s Out B NK base startN values).mem r a = s.mem r a := by
  unfold writeOutput
  generalize TileShape.allIndices [B] = indices
  induction indices generalizing s with
  | nil => rfl
  | cons i rest ih =>
      rw [List.foldl_cons]
      split_ifs with active
      · rw [ih]
        have hn : ¬(r = Out ∧ a = base+startN*B+i.1.val) := by
          rintro ⟨hr,ha⟩
          rcases h with hr' | hi
          · exact hr' hr
          · exact hi i.1 active ha
        simp [BlockState.writeMem, hn]
      · exact ih s

theorem executes (s : BlockState) (Out : RegionName) (B NK soz soh batch head startN : Nat)
    (values : Fin B → ℝ)
    (hz : s.regs .int [] "off_z" = some (Tile.scalar (batch : Int)))
    (hh : s.regs .int [] "off_h" = some (Tile.scalar (head : Int)))
    (hn : s.regs .int [] "start_n" = some (Tile.scalar (startN : Int)))
    (ho : s.regs .real [B] "o" = some (Tile.vec (fun i => some (values i)))) :
    ∃ t, stepStmts (RealProgram.postlude Out B NK soz soh) s = some t ∧
      (∀ i : Fin B, t.readMem Out (batch*soz+head*soh+startN*B+i.val) =
        if startN*B+i.val < NK then values i
        else s.readMem Out (batch*soz+head*soh+startN*B+i.val)) ∧
      (∀ r a, (r ≠ Out ∨ ∀ i : Fin B, startN*B+i.val < NK →
        a ≠ batch*soz+head*soh+startN*B+i.val) → t.mem r a = s.mem r a) := by
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
  have es : stepStmt (.store .real [B] (.ptr (.ref .ptr [B] "o_ptrs"))
      (.ref .real [B] "o") (.mask (.lt .int .scalarR (.ref .int [B] "o_range") (.castNatToInt (.constNat NK))))) sp =
      some (writeOutput sp Out B NK base startN values) := by
    simp [stepStmt, evalOp.eq_def, sp, sr, so, ho, writeOutput,
      Tile.cop, Tile.scalar, Tile.vec, Broadcast.leftIndex, Broadcast.rightIndex,
      ComparableDType.lt, BlockState.writeMemTyped_real, FloatDType.storeValue,
      FloatDType.real_toWithBot, WithBot.unbotD_some, Nat.add_comm, -Tile.scalar_eta, ← Int.natCast_add, ← Int.natCast_mul, -Nat.cast_add, -Nat.cast_mul]
  have he : stepStmts (RealProgram.postlude Out B NK soz soh) s =
      some (writeOutput sp Out B NK base startN values) := by
    rw [RealProgram.postlude, stepStmts.cons_some (stepStmt_assign_eq_some eo),
      stepStmts.cons_some (stepStmt_assign_eq_some er), stepStmts.cons_some (stepStmt_assign_eq_some ep),
      stepStmts.cons_some es, stepStmts.nil]
  refine ⟨_, he, ?_, ?_⟩
  · intro i
    simpa [sp,sr,so,base] using readback sp Out B NK base startN values i
  · intro r a ha
    exact frame sp Out B NK base startN values r a ha

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealOutput
