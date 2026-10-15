import bench.tritonbench_g.pow_scalar_tensor.Proofs.IntegerSource
import bench.tritonbench_g.pow_scalar_tensor.Proofs.Transport

namespace VeriTile.Bench.TritonBenchG.PowScalarTensor.IntegerSteps
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

/-- Independent scalar-base integer power from the initial typed input cell.
Negative exponents use real zpow; no output or execution is consulted. -/
def powSpec (s : BlockState) (X : RegionName) (sx : Nat) (base : ℝ) (task : Nat) : ℝ :=
  base ^ s.readMemValue .int X (task * sx)

/-- The exact integer exponent tile computed by a source load when undef is zero. -/
def exponents (s : BlockState) (X : RegionName) (sx N B T : Nat) : Tile .int [B] :=
  ⟨fun i => if T * B + i.1.val < N then s.readMemValue .int X ((T*B+i.1.val)*sx) else 0⟩

/-- All computed lanes satisfy the native signed32 domain; masked lanes really
contain integer zero, and are checked even though they will not be stored. -/
theorem exponents_valid (s : BlockState) (X : RegionName) (sx N B T : Nat)
    (h : ∀ i : Fin B, T*B+i.val < N →
      Int32Bits.inRange (s.readMemValue .int X ((T*B+i.val)*sx)) = Bool.true) :
    Int32Bits.valuesValid (exponents s X sx N B T).data = Bool.true := by
  simp only [Int32Bits.valuesValid, List.all_eq_true]
  intro i hi
  dsimp [exponents]
  split_ifs with ha
  · exact h i.1 ha
  · rfl

/-- A complete int32 load, including the oracle-to-integer conversion on its
unspecified masked lanes. -/
theorem load_int (R : RegionName) (len stride BT off : Nat)
    (bpName : RegName) (t : BlockState) (hu : t.undef = fun _ _ => 0)
    (hbp : t.regs .blockPtr [BT] bpName = some (⟨fun _ : TileIndex [BT] =>
        { region := R, baseOffset := 0, parentShape := [len],
          blockShape := [BT], strides := [stride], offsets := [off] }⟩ :
        Tile .blockPtr [BT])) :
    evalOp (Op.load .int
        (MemAccess.blockPtr (Op.ref .blockPtr [BT] bpName) [0]) .none .unspecified) t
      = some (⟨fun idx : TileIndex [BT] =>
          if off + idx.1.val < len then t.readMemValue .int R ((off+idx.1.val)*stride)
          else 0⟩ : Tile .int [BT]) := by
  simp only [evalOp, evalOp_ref, hbp, Option.bind_some, Option.bind]
  refine congrArg some ?_
  congr 1
  funext idx
  obtain ⟨i,u⟩ := idx
  simp only [TileShape.indexToList, BlockPtr.inBounds_1d, BlockPtr.address_1d,
    Nat.zero_add, decide_eq_true_eq]
  by_cases ha : off+i.val < len
  · simp [ha]
  · simp [ha, t.paddingValue_of_undef_zero _ _ _ _ hu, BlockState.defaultCarrier]

private theorem outAddr_tileIndex_injective (T : Nat)
    {out0_stride0 tile_size0 : Nat} (hStride : 0 < out0_stride0) :
    Function.Injective (fun idx : TileIndex [tile_size0] =>
      (T * tile_size0 + idx.1.val) * out0_stride0) := by
  rintro ⟨a,u⟩ ⟨b,v⟩ h
  have hab : a = b := Fin.ext (Nat.add_left_cancel (Nat.eq_of_mul_eq_mul_right hStride h))
  cases hab; cases u; cases v; rfl

theorem perTile_steps (candidate : Bool)
    (val0 : ℝ) (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 tile_size0 : Nat)
    (hStride : 0 < out0_stride0)
    (T : Nat) (t : BlockState) (V : Tile .nat [])
    (hT : t.regs .nat [] "tile_id" = some V)
    (hV : V.data PUnit.unit = T)
    (hu : t.undef = fun _ _ => 0)
    (hvalid : ∀ i : Fin tile_size0, T*tile_size0+i.val < s0 →
      Int32Bits.inRange (t.readMemValue .int in0_ptr ((T*tile_size0+i.val)*in0_stride0)) = Bool.true) :
    ∃ t', stepStmts (IntegerSource.perTile candidate val0 in0_ptr out0_ptr in0_stride0
        out0_stride0 s0 tile_size0) t = some t'
      ∧ (∀ i : Fin tile_size0, taskIndex T tile_size0 i < s0 →
          t'.readMem out0_ptr (taskIndex T tile_size0 i * out0_stride0)
            = powSpec t in0_ptr in0_stride0 val0 (taskIndex T tile_size0 i))
      ∧ (∀ o : Nat, (∀ i : Fin tile_size0, taskIndex T tile_size0 i < s0 →
            o ≠ taskIndex T tile_size0 i * out0_stride0) →
          t'.readMem out0_ptr o = t.readMem out0_ptr o)
      ∧ (∀ (R : RegionName) (o : Nat), R ≠ out0_ptr →
          t'.readMem R o = t.readMem R o)
      ∧ (∀ (dtype : TileDType) (shape : TileShape) (name : RegName),
          name ≠ "tile_id0" → name ≠ "offset0" → name ≠ "in0_bptr" →
          name ≠ "in0" → name ≠ "out0" → name ≠ "out0_bptr" →
          t'.regs dtype shape name = t.regs dtype shape name)
      ∧ (∀ (R : RegionName) (o : Nat),
          (R ≠ out0_ptr ∨ ∀ i : Fin tile_size0, taskIndex T tile_size0 i < s0 →
            o ≠ taskIndex T tile_size0 i * out0_stride0) →
          t'.mem R o = t.mem R o) := by
  -- the intermediate register values (raw evaluation results)
  set offV : Tile .nat [] :=
    Tile.bop (NumericDType.mul .nat) Broadcast.nil V (Tile.scalar tile_size0)
    with hoffV
  set inBP : Tile .blockPtr [tile_size0] := ⟨fun _ =>
    { region := in0_ptr, baseOffset := 0, parentShape := [s0],
      blockShape := [tile_size0], strides := [in0_stride0],
      offsets := [T * tile_size0] }⟩ with hinBP
  set outBP : Tile .blockPtr [tile_size0] := ⟨fun _ =>
    { region := out0_ptr, baseOffset := 0, parentShape := [s0],
      blockShape := [tile_size0], strides := [out0_stride0],
      offsets := [T * tile_size0] }⟩ with houtBP
  set inT : Tile .int [tile_size0] :=
    exponents t in0_ptr in0_stride0 s0 tile_size0 T with hinT
  set outT : Tile .real [tile_size0] :=
    Tile.libdevicePowi Broadcast.scalarL (Tile.scalar (some val0)) inT with houtT
  have hvalues : Int32Bits.valuesValid inT.data = Bool.true :=
    exponents_valid t in0_ptr in0_stride0 s0 tile_size0 T hvalid
  -- the six register-writing statements
  set t1 := t.setReg "tile_id0" .nat [] V with ht1
  set t2 := t1.setReg "offset0" .nat [] offV with ht2
  set t3 := t2.setReg "in0_bptr" .blockPtr [tile_size0] inBP with ht3
  set t4 := t3.setReg "in0" .int [tile_size0] inT with ht4
  set t5 := t4.setReg "out0" .real [tile_size0] outT with ht5
  set t6 := t5.setReg "out0_bptr" .blockPtr [tile_size0] outBP with ht6
  have hoffData : offV.data PUnit.unit = T * tile_size0 := by
    rw [hoffV, Tile.bop_data]
    show NumericDType.mul .nat (V.data PUnit.unit) tile_size0 = T * tile_size0
    rw [hV]
    rfl
  have hstep : stepStmts (IntegerSource.perTile candidate val0 in0_ptr out0_ptr in0_stride0
      out0_stride0 s0 tile_size0) t
      = some ((TileShape.allIndices [tile_size0]).foldl
          (fun acc idx =>
            if T * tile_size0 + idx.1.val < s0 then
              acc.writeMem out0_ptr ((T * tile_size0 + idx.1.val) * out0_stride0)
                ((outT.data idx).unbotD 0)
            else acc) t6) := by
    simp only [IntegerSource.perTile]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.ref .nat [] "tile_id") t = some V by simp [hT]))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (IntegerSource.offset candidate tile_size0) t1
        = some offV by
          change evalOp (RealSource.offset candidate tile_size0) t1 = some offV
          rw [RealTransport.offset_eval]
          simp [ht1, hoffV]))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (makeBlockPtr_1d_eval in0_ptr s0 in0_stride0 tile_size0 "offset0" t2
        offV (T * tile_size0) (by simp [ht2]) hoffData))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.load .int (MemAccess.blockPtr
          (Op.ref .blockPtr [tile_size0] "in0_bptr") [0]) .none .unspecified) t3
        = some inT from
        load_int in0_ptr s0 in0_stride0 tile_size0
          (T * tile_size0) "in0_bptr" t3 hu (by simp [ht3, hinBP])))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.libdevicePowi Broadcast.scalarL
          (Op.const val0) (Op.ref .int [tile_size0] "in0")) t4
        = some outT by
        rw [evalOp_libdevicePowi, evalOp_const]
        simp only [evalOp_ref,
          show t4.regs .int [tile_size0] "in0" = some inT by simp [ht4],
          bind, Option.bind, hvalues, if_true, houtT]))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (makeBlockPtr_1d_eval out0_ptr s0 out0_stride0 tile_size0 "offset0" t5
        offV (T * tile_size0) (by simp [ht5, ht4, ht3, ht2]) hoffData))]
    rw [stepStmts.cons_some (store_blockPtr_1d_checked_step out0_ptr s0
      out0_stride0 tile_size0 (T * tile_size0) "out0_bptr" "out0" outT t6
      (by simp [ht6, houtBP]) (by simp [ht6, ht5]))]
    exact stepStmts.nil
  refine ⟨_, hstep, ?_, ?_, ?_, ?_, ?_⟩
  · -- active lanes: genuine scalar-base power value
    intro i hi
    simp only [taskIndex] at hi ⊢
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
      (outAddr_tileIndex_injective T hStride) (i, PUnit.unit)]
    rw [if_pos hi]
    simp only [houtT, hinT, exponents, Tile.libdevicePowi_data, Broadcast.leftIndex_scalarL,
      Broadcast.rightIndex_scalarL, Tile.scalar, if_pos hi, powSpec]
    rfl
  · -- untouched `out0` cells
    intro o ho
    have hpres := BlockState.scatter_prop_masked_preserves_other_offset
      (region := out0_ptr)
      (fun idx : TileIndex [tile_size0] =>
        (T * tile_size0 + idx.1.val) * out0_stride0)
      (fun idx : TileIndex [tile_size0] => (outT.data idx).unbotD 0)
      (fun idx : TileIndex [tile_size0] => T * tile_size0 + idx.1.val < s0)
      o (fun idx hidx => by
        have := ho idx.1 (by simpa [taskIndex] using hidx)
        simpa [taskIndex] using Ne.symm this)
      (TileShape.allIndices [tile_size0]) t6
    rw [hpres]
    simp [ht6, ht5, ht4, ht3, ht2, ht1]
  · -- other regions untouched
    intro R o hR
    have hpres := BlockState.scatter_prop_masked_preserves_other_region
      (region := out0_ptr)
      (fun idx : TileIndex [tile_size0] =>
        (T * tile_size0 + idx.1.val) * out0_stride0)
      (fun idx : TileIndex [tile_size0] => (outT.data idx).unbotD 0)
      (fun idx : TileIndex [tile_size0] => T * tile_size0 + idx.1.val < s0)
      R hR o (TileShape.allIndices [tile_size0]) t6
    rw [hpres]
    simp [ht6, ht5, ht4, ht3, ht2, ht1]
  · -- registers other than the six per-tile scratch registers untouched
    intro dtype shape name h1 h2 h3 h4 h5 h6
    rw [BlockState.foldl_writeMem_prop_masked_regs]
    simp [ht6, ht5, ht4, ht3, ht2, ht1, h1, h2, h3, h4, h5, h6]
  · -- cell-level frame off the written window (the flat-memory IO face)
    intro R o hcond
    rw [foldl_writeMem_frame
      (region := out0_ptr)
      (fun idx : TileIndex [tile_size0] =>
        (T * tile_size0 + idx.1.val) * out0_stride0)
      (fun idx : TileIndex [tile_size0] => (outT.data idx).unbotD 0)
      (fun idx : TileIndex [tile_size0] => T * tile_size0 + idx.1.val < s0)
      R o (TileShape.allIndices [tile_size0]) ?_]
    · simp [ht6, ht5, ht4, ht3, ht2, ht1]
    · rcases hcond with h | h
      · exact Or.inl h
      · exact Or.inr fun idx _ hidx =>
          Ne.symm (h idx.1 (by simpa [taskIndex] using hidx))


theorem perTile_flat (candidate : Bool) (base : ℝ) (X Y : RegionName) (sx sy N B : Nat) :
    StmtList.FlattenOk (IntegerSource.perTile candidate base X Y sx sy N B) := by
  cases candidate <;> simp [IntegerSource.perTile, IntegerSource.offset,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.PowScalarTensor.IntegerSteps
