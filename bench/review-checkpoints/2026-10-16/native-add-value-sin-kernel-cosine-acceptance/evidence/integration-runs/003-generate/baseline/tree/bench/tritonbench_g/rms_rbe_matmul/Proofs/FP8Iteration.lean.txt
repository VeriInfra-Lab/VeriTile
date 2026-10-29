import bench.tritonbench_g.rms_rbe_matmul.Proofs.FP8Loads

namespace VeriTile.Bench.TritonBenchG.RmsRbeMatmul.ByteExecution
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

private theorem eval_int_two (s : BlockState) :
    evalOp (Op.constInt 2) s = some (Tile.scalar (2 : Int)) := by
  simp only [evalOp]

/-- Finite decoding is required at each actual unmasked weight address,
including padding and wrapped inactive M/N lanes. -/
def Bytes (s : BlockState) (W RMS : RegionName)
    (N BN BK swk swn srms blocks : Nat) : Prop :=
  (∀ t < blocks, ∀ (e : Fin BK) (j : Fin BN),
    ByteLoads.FiniteAt s W (e.val * swk + colIndex s N BN j * swn + t * BK * swk)) ∧
  (∀ t < blocks, ∀ e : Fin BK,
    ByteLoads.FiniteAt s RMS (e.val * srms + t * BK * srms))

/-- The numerical registers follow independent sums of the decoded input
view. Actual execution retains the original raw cells and byte encodings. -/
noncomputable def Invariant (X W RMS : RegionName) (initial : BlockState)
    (M N BM BN BK sxb sxm sxk swk swn srms blocks t : Nat) (s : BlockState) : Prop :=
  rrmInvariant X W RMS (ByteLoads.view W RMS initial)
    M N BM BN BK sxb sxm sxk swk swn srms blocks t (ByteLoads.view W RMS s) ∧
    s.mem = initial.mem

/-- A complete raw-byte source iteration updates the RMS moment and matrix
product while preserving every original raw memory cell. -/
theorem step (X W RMS : RegionName) (raw0 : BlockState)
    (M N BM BN BK sxb sxm sxk swk swn srms numKBlocks : Nat)
    (hXW : X ≠ W) (hXR : X ≠ RMS)
    (hbytes : Bytes raw0 W RMS N BN BK swk swn srms numKBlocks)
    (t : Nat) (s : BlockState) (hclt : t < numKBlocks)
    (hinv : Invariant X W RMS raw0 M N BM BN BK sxb sxm sxk swk swn srms numKBlocks t s) :
    ∃ s', stepStmts (RealSource.loop Bool.true BM BN BK sxk swk srms)
        (s.setReg "_i" .nat [] (Tile.scalar t)) = some s' ∧
      Invariant X W RMS raw0 M N BM BN BK sxb sxm sxk swk swn srms numKBlocks (t + 1) s' := by
  rw [ByteLoads.source_iteration]
  obtain ⟨hinv, hrawmem⟩ := hinv
  let s0 := ByteLoads.view W RMS raw0
  simp only [rrmInvariant] at hinv
  obtain ⟨hpids, hcle, hpb, hpm, hpn, hacc, hxsum, hxp, hwp, hrmsp, hundef, hmem⟩ := hinv
  change s.pids = s0.pids at hpids
  change s.regs _ _ _ = _ at hpb hpm hpn hacc hxsum hxp hwp hrmsp
  change ∀ rg o, s.undef rg o = 0 at hundef
  set xpT : Tile .ptr [BM, BK] :=
    ⟨fun idx : TileIndex [BM, BK] =>
      (X.cast, s0.pids 0 * sxb + rowIndex s0 M N BM BN idx.1 * sxm + idx.2.1.val * sxk
        + t * BK * sxk)⟩ with hxpT
  set wpT : Tile .ptr [BK, BN] :=
    ⟨fun idx : TileIndex [BK, BN] =>
      (W.cast, idx.1.val * swk + colIndex s0 N BN idx.2.1 * swn + t * BK * swk)⟩ with hwpT
  set rmspT : Tile .ptr [1, BK] :=
    ⟨fun idx : TileIndex [1, BK] => (RMS.cast, idx.2.1.val * srms + t * BK * srms)⟩ with hrmspT
  set accT : Tile .real [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] =>
      some (accPartial s0 X W RMS M N BM BN sxb sxm sxk swk swn srms BK idx.1 idx.2.1 t)⟩ with haccT
  set sk := s.setReg "_i" .nat [] (Tile.scalar t) with hsk
  have hrmem : ∀ o : Nat, sk.readMem X o = s0.readMem X o := by
    intro o
    simp only [hsk, BlockState.setReg_readMem, s0, BlockState.readMem]
    rw [ByteLoads.view_outside W RMS raw0 X o hXW hXR, BlockState.setReg_mem, hrawmem]
  have hvread : ∀ (R : RegionName) (o : Nat), (ByteLoads.view W RMS sk).readMem R o = s0.readMem R o := by
    intro R o
    unfold BlockState.readMem
    rw [ByteLoads.view_memory W RMS sk raw0 (by simpa [hsk] using hrawmem)]
  have hxpk : sk.regs .ptr [BM, BK] "x_ptrs" = some xpT := by simp [hsk, hxp, hxpT]
  have hwpk : sk.regs .ptr [BK, BN] "w_ptrs" = some wpT := by simp [hsk, hwp, hwpT]
  have hrmspk : sk.regs .ptr [1, BK] "rms_w_ptrs" = some rmspT := by simp [hsk, hrmsp, hrmspT]
  -- canonical loaded / derived tiles
  set xT0 : Tile .real [BM, BK] :=
    ⟨fun idx : TileIndex [BM, BK] =>
      some (xElem s0 X M N BM BN sxb sxm sxk idx.1 (t * BK + idx.2.1.val))⟩ with hxT0
  set rmsT : Tile .real [1, BK] :=
    ⟨fun idx : TileIndex [1, BK] =>
      some (rmsElem s0 RMS srms (t * BK + idx.2.1.val))⟩ with hrmsT
  set xT1 : Tile .real [BM, BK] :=
    ⟨fun idx : TileIndex [BM, BK] =>
      some (xElem s0 X M N BM BN sxb sxm sxk idx.1 (t * BK + idx.2.1.val)
        * rmsElem s0 RMS srms (t * BK + idx.2.1.val))⟩ with hxT1
  set wT : Tile .real [BK, BN] :=
    ⟨fun idx : TileIndex [BK, BN] =>
      some (wElem s0 W N BN swk swn idx.2.1 (t * BK + idx.1.val))⟩ with hwT
  set xsumT1 : Tile .real [BM, BK] :=
    ⟨fun idx : TileIndex [BM, BK] =>
      some (xSumPartial s0 X M N BM BN sxb sxm sxk BK idx.1 idx.2.1 (t + 1))⟩ with hxsumT1
  simp only [ByteLoads.loop, rrmLoopBody, List.map_cons, List.map_nil, ByteLoads.lowered,
    Bool.or_eq_true, beq_iff_eq, String.reduceEq, false_or, or_false, ↓reduceIte]
  -- 1: x = tl.load(x_ptrs)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.load .real (.ptr (Op.ref .ptr [BM, BK] "x_ptrs")) MaskOpt.none) sk
          = some xT0 from by
        rw [load_ptr_none_real "x_ptrs" sk xpT hxpk]
        refine congrArg some ?_
        ext idx
        simp only [hxpT, hrmem, Region.cast_id, hxT0, xElem]
        rw [show s0.pids 0 * sxb + rowIndex s0 M N BM BN idx.1 * sxm + idx.2.1.val * sxk
                + t * BK * sxk
              = s0.pids 0 * sxb + rowIndex s0 M N BM BN idx.1 * sxm
                + (t * BK + idx.2.1.val) * sxk from by ring]))]
  -- 2: x_sum += pow(x.to(fp32), 2)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Op.ref .real [BM, BK] "x_sum")
            (Op.libdevicePowi Broadcast.scalarR
              (Op.castFloat FloatDType.real FloatDType.real (Op.ref .real [BM, BK] "x"))
              (Op.constInt 2)))
          (sk.setReg "x" .real [BM, BK] xT0) = some xsumT1 from by
        rw [evalOp_add]
        simp only [evalOp_libdevicePowi, evalOp_ref, eval_int_two, Option.bind_eq_bind]
        rw [cast_real_real_eval "x" _ xT0 (by simp)]
        rw [show (sk.setReg "x" .real [BM, BK] xT0).regs .real [BM, BK] "x_sum"
              = some (⟨fun idx : TileIndex [BM, BK] =>
                  some (xSumPartial s0 X M N BM BN sxb sxm sxk BK idx.1 idx.2.1 t)⟩ : Tile .real [BM, BK]) from by
            simp [hsk, hxsum, s0]]
        simp only [eval_int_two, Option.bind_some]
        refine congrArg some ?_
        ext idx
        simp only [hxsumT1, hxT0, Tile.bop_data, Tile.scalar_data,
          Broadcast.leftIndex_consSame, Broadcast.rightIndex_consSame,
          Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
          Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
          Tile.libdevicePowi_data, WithBot.realPowi_two,
          NumericDType.add, WithBot.realAdd, WithBot.realMul,
          Option.map₂, Option.bind, Option.map]
        rw [xSumPartial_succ, pow_two]))]
  -- 3: rms_w = tl.load(rms_w_ptrs)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show @evalOp .real [1, BK] (Op.decodeFloat8E5 .real
        (Op.load .int (.ptr (Op.ref .ptr [1, BK] "rms_w_ptrs")) MaskOpt.none))
          ((sk.setReg "x" .real [BM, BK] xT0).setReg "x_sum" .real [BM, BK] xsumT1)
          = some rmsT from by
        rw [ByteLoads.load W RMS _ "rms_w_ptrs" rmspT (by simp [hrmspk])
          (by intro i; simp [hrmspT])
          (by
            intro i
            apply (ByteLoads.finite_memory _ raw0 (by simpa only [BlockState.setReg_mem, hsk] using hrawmem) _ _).mpr
            simpa [hrmspT] using hbytes.2 t hclt i.2.1)]
        refine congrArg some ?_
        ext idx
        simp only [hrmspT, ByteLoads.view_setReg, BlockState.setReg_readMem, hvread, Region.cast_id, hrmsT, rmsElem]
        rw [show idx.2.1.val * srms + t * BK * srms
              = (t * BK + idx.2.1.val) * srms from by ring]))]
  -- 4: x = x * rms_w
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.mul .real (Broadcast.consR (Broadcast.consSame Broadcast.nil))
            (Op.ref .real [BM, BK] "x") (Op.ref .real [1, BK] "rms_w")) _
          = some xT1 from by
        rw [evalOp_mul]
        simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
          String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
        refine congrArg some ?_
        ext idx
        simp only [hxT0, hrmsT, hxT1, Tile.bop_data,
          Broadcast.leftIndex_consR, Broadcast.leftIndex_consSame, Broadcast.leftIndex_nil,
          Broadcast.rightIndex_consR, Broadcast.rightIndex_consSame, Broadcast.rightIndex_nil,
          NumericDType.mul, WithBot.realMul, Option.map₂, Option.bind, Option.map]))]
  -- 5: w = tl.load(w_ptrs)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show @evalOp .real [BK, BN] (Op.decodeFloat8E5 .real
        (Op.load .int (.ptr (Op.ref .ptr [BK, BN] "w_ptrs")) MaskOpt.none)) _
          = some wT from by
        rw [ByteLoads.load W RMS _ "w_ptrs" wpT (by simp [hwpk])
          (by intro i; simp [hwpT])
          (by
            intro i
            apply (ByteLoads.finite_memory _ raw0 (by simpa only [BlockState.setReg_mem, hsk] using hrawmem) _ _).mpr
            simpa [hwpT, s0, ByteLoads.view, colIndex, colGlobal] using hbytes.1 t hclt i.1 i.2.1)]
        refine congrArg some ?_
        ext idx
        simp only [hwpT, ByteLoads.view_setReg, BlockState.setReg_readMem, hvread, Region.cast_id, hwT, wElem]
        rw [show idx.1.val * swk + colIndex s0 N BN idx.2.1 * swn + t * BK * swk
              = (t * BK + idx.1.val) * swk + colIndex s0 N BN idx.2.1 * swn from by ring]))]
  -- 6: accumulator += tl.dot(x, w)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Op.ref .real [BM, BN] "accumulator")
            (Op.dot (batch := []) (Op.ref .real [BM, BK] "x") (Op.ref .real [BK, BN] "w"))) _
          = some (⟨fun idx : TileIndex [BM, BN] =>
              some (accPartial s0 X W RMS M N BM BN sxb sxm sxk swk swn srms BK idx.1 idx.2.1 (t + 1))⟩
              : Tile .real [BM, BN]) from by
        rw [accdot_eval BM BK BN _ "accumulator" "w" accT xT1 wT
            (by simp [hsk, hacc, haccT]) (by simp) (by simp)]
        refine congrArg some ?_
        ext idx
        rw [adddot_lane BM BN accT (Tile.dot [] xT1 wT) idx.1 idx.2.1
            (accPartial s0 X W RMS M N BM BN sxb sxm sxk swk swn srms BK idx.1 idx.2.1 t)
            (Finset.univ.sum fun e : Fin BK =>
              (xElem s0 X M N BM BN sxb sxm sxk idx.1 (t * BK + e.val)
                * rmsElem s0 RMS srms (t * BK + e.val))
                * wElem s0 W N BN swk swn idx.2.1 (t * BK + e.val))
            (by rw [haccT])
            (tile_dot_data BM BK BN xT1 wT idx.1 idx.2.1 _ _
              (fun e => by rw [hxT1]) (fun e => by rw [hwT]))]
        show _ = some (accPartial s0 X W RMS M N BM BN sxb sxm sxk swk swn srms BK idx.1 idx.2.1 (t + 1))
        rw [accPartial_succ]))]
  -- 7–9: the three pointer advances
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (ptr_adv_eval _ "x_ptrs" BK sxk xpT (by simp [hxpk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (ptr_adv_eval _ "w_ptrs" BK swk wpT (by simp [hwpk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (ptr_adv_eval _ "rms_w_ptrs" BK srms rmspT (by simp [hrmspk])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  change rrmInvariant X W RMS s0 M N BM BN BK sxb sxm sxk swk swn srms numKBlocks (t + 1)
    (ByteLoads.view W RMS _) ∧ _
  constructor
  swap
  · simpa only [BlockState.setReg_mem, hsk] using hrawmem
  simp only [rrmInvariant, ByteLoads.view_setReg]
  refine ⟨by simpa [hsk, ByteLoads.view] using hpids, by omega, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simpa [hsk, ByteLoads.view, s0] using hpb
  · simpa [hsk, ByteLoads.view, s0] using hpm
  · simpa [hsk, ByteLoads.view, s0] using hpn
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    rw [hxsumT1]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.scalar, hxpT, NumericDType.add]
    ring
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.scalar, hwpT, NumericDType.add]
    ring
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.scalar, hrmspT, NumericDType.add]
    ring
  · intro rg o; simpa [hsk, ByteLoads.view] using hundef rg o
  · show _ = s0.mem
    rw [← hmem, hsk]; rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsRbeMatmul.ByteExecution
