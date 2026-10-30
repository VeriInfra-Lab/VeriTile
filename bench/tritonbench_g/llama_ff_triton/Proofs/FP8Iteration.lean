import bench.tritonbench_g.llama_ff_triton.Proofs.FP8Loads

namespace VeriTile.Bench.TritonBenchG.LlamaFfTriton.ByteExecution
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- Finite-byte conditions cover every unmasked weight address reached by
all actual K iterations. The FP equivalence has no such restriction. -/
def Bytes (s : BlockState) (W1 W3 RMS : RegionName)
    (N BN BK sw1k sw1n sw3k sw3n srms blocks : Nat) : Prop :=
  (∀ t < blocks, ∀ (e : Fin BK) (j : Fin BN),
    ByteLoads.FiniteAt s W1 (e.val * sw1k + colIndex s N BN j * sw1n + t * BK * sw1k)) ∧
  (∀ t < blocks, ∀ (e : Fin BK) (j : Fin BN),
    ByteLoads.FiniteAt s W3 (e.val * sw3k + colIndex s N BN j * sw3n + t * BK * sw3k)) ∧
  (∀ t < blocks, ∀ e : Fin BK,
    ByteLoads.FiniteAt s RMS (e.val * srms + t * BK * srms))

/-- Numerical registers follow the real recurrence of the finite decoded
input view; the source state retains every original typed memory cell. -/
def Invariant (A W1 W3 RMS : RegionName) (initial : BlockState)
    (M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms blocks t : Nat) (s : BlockState) : Prop :=
  ffInvariant A W1 W3 RMS (ByteLoads.view W1 W3 RMS initial)
    M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms blocks t
    (ByteLoads.view W1 W3 RMS s) ∧ s.mem = initial.mem

/-- A complete raw-byte source iteration updates both matrix products and
RMS moment by the actual loaded values, preserving the original raw memory. -/
theorem step (A W1 W3 RMS : RegionName) (raw0 : BlockState)
    (M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms numKBlocks : Nat)
    (hA1 : A ≠ W1) (hA3 : A ≠ W3) (hAR : A ≠ RMS)
    (hbytes : Bytes raw0 W1 W3 RMS N BN BK sw1k sw1n sw3k sw3n srms numKBlocks)
    (t : Nat) (s : BlockState) (hclt : t < numKBlocks)
    (hinv : Invariant A W1 W3 RMS raw0 M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms numKBlocks t s) :
    ∃ s', stepStmts (RealSource.loop Bool.true BM BN BK sak sw1k sw3k srms)
        (s.setReg "_i" .nat [] (Tile.scalar t)) = some s' ∧
      Invariant A W1 W3 RMS raw0 M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms numKBlocks (t + 1) s' := by
  rw [ByteLoads.source_iteration]
  obtain ⟨hinv, hrawmem⟩ := hinv
  let s0 := ByteLoads.view W1 W3 RMS raw0
  simp only [ffInvariant] at hinv
  obtain ⟨hpids, hcle, hpm, hpn, hacc1, hacc2, hasum, hap, hw1p, hw3p, hrmsp, hundef, hmem⟩ := hinv
  change s.pids = s0.pids at hpids
  change s.regs _ _ _ = _ at hpm hpn hacc1 hacc2 hasum hap hw1p hw3p hrmsp
  change ∀ rg o, s.undef rg o = 0 at hundef
  set apT : Tile .ptr [BM, BK] :=
    ⟨fun idx : TileIndex [BM, BK] =>
      (A.cast, rowIndex s0 M N BM BN idx.1 * sam + idx.2.1.val * sak + t * BK * sak)⟩ with hapT
  set w1pT : Tile .ptr [BK, BN] :=
    ⟨fun idx : TileIndex [BK, BN] =>
      (W1.cast, idx.1.val * sw1k + colIndex s0 N BN idx.2.1 * sw1n + t * BK * sw1k)⟩ with hw1pT
  set w3pT : Tile .ptr [BK, BN] :=
    ⟨fun idx : TileIndex [BK, BN] =>
      (W3.cast, idx.1.val * sw3k + colIndex s0 N BN idx.2.1 * sw3n + t * BK * sw3k)⟩ with hw3pT
  set rmspT : Tile .ptr [1, BK] :=
    ⟨fun idx : TileIndex [1, BK] => (RMS.cast, idx.2.1.val * srms + t * BK * srms)⟩ with hrmspT
  set acc1T : Tile .real [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] =>
      some (accPartial s0 A W1 RMS M N BM BN sam sak sw1k sw1n srms BK idx.1 idx.2.1 t)⟩ with hacc1T
  set acc2T : Tile .real [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] =>
      some (accPartial s0 A W3 RMS M N BM BN sam sak sw3k sw3n srms BK idx.1 idx.2.1 t)⟩ with hacc2T
  set sk := s.setReg "_i" .nat [] (Tile.scalar t) with hsk
  have hrmem : ∀ o : Nat, sk.readMem A o = s0.readMem A o := by
    intro o
    simp only [hsk, BlockState.setReg_readMem, s0, BlockState.readMem]
    rw [ByteLoads.view_outside W1 W3 RMS raw0 A o hA1 hA3 hAR, BlockState.setReg_mem, hrawmem]
  have hvread : ∀ (R : RegionName) (o : Nat), (ByteLoads.view W1 W3 RMS sk).readMem R o = s0.readMem R o := by
    intro R o
    unfold BlockState.readMem
    rw [ByteLoads.view_memory W1 W3 RMS sk raw0 (by simpa [hsk] using hrawmem)]
  have hapk : sk.regs .ptr [BM, BK] "a_ptrs" = some apT := by simp [hsk, hap, hapT]
  have hw1pk : sk.regs .ptr [BK, BN] "w1_ptrs" = some w1pT := by simp [hsk, hw1p, hw1pT]
  have hw3pk : sk.regs .ptr [BK, BN] "w3_ptrs" = some w3pT := by simp [hsk, hw3p, hw3pT]
  have hrmspk : sk.regs .ptr [1, BK] "rms_w_ptrs" = some rmspT := by simp [hsk, hrmsp, hrmspT]
  -- canonical loaded / derived tiles
  set aT0 : Tile .real [BM, BK] :=
    ⟨fun idx : TileIndex [BM, BK] =>
      some (aElem s0 A M N BM BN sam sak idx.1 (t * BK + idx.2.1.val))⟩ with haT0
  set rmsT : Tile .real [1, BK] :=
    ⟨fun idx : TileIndex [1, BK] =>
      some (rmsElem s0 RMS srms (t * BK + idx.2.1.val))⟩ with hrmsT
  set aT1 : Tile .real [BM, BK] :=
    ⟨fun idx : TileIndex [BM, BK] =>
      some (aElem s0 A M N BM BN sam sak idx.1 (t * BK + idx.2.1.val)
        * rmsElem s0 RMS srms (t * BK + idx.2.1.val))⟩ with haT1
  set w1T : Tile .real [BK, BN] :=
    ⟨fun idx : TileIndex [BK, BN] =>
      some (wElem s0 W1 N BN sw1k sw1n idx.2.1 (t * BK + idx.1.val))⟩ with hw1T
  set w3T : Tile .real [BK, BN] :=
    ⟨fun idx : TileIndex [BK, BN] =>
      some (wElem s0 W3 N BN sw3k sw3n idx.2.1 (t * BK + idx.1.val))⟩ with hw3T
  set asumT1 : Tile .real [BM, BK] :=
    ⟨fun idx : TileIndex [BM, BK] =>
      some (aSumPartial s0 A M N BM BN sam sak BK idx.1 idx.2.1 (t + 1))⟩ with hasumT1
  simp only [ByteLoads.loop, ffLoopBody, List.map_cons, List.map_nil, ByteLoads.lowered,
    Bool.or_eq_true, beq_iff_eq, String.reduceEq, false_or, or_false, ↓reduceIte]
  -- 1: a = tl.load(a_ptrs)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.load .real (.ptr (Op.ref .ptr [BM, BK] "a_ptrs")) MaskOpt.none) sk
          = some aT0 from by
        rw [load_ptr_none_real "a_ptrs" sk apT hapk]
        refine congrArg some ?_
        ext idx
        simp only [hapT, hrmem, Region.cast_id, haT0, aElem]
        rw [show rowIndex s0 M N BM BN idx.1 * sam + idx.2.1.val * sak + t * BK * sak
              = rowIndex s0 M N BM BN idx.1 * sam + (t * BK + idx.2.1.val) * sak from by ring]))]
  -- 2: a_sum += pow(a.to(fp32), 2)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Op.ref .real [BM, BK] "a_sum")
            (Op.pow Broadcast.scalarR
              (Op.castFloat FloatDType.real FloatDType.real (Op.ref .real [BM, BK] "a"))
              (Op.const 2)))
          (sk.setReg "a" .real [BM, BK] aT0) = some asumT1 from by
        rw [evalOp_add]
        simp only [evalOp_pow, evalOp_ref, evalOp_const, Option.bind_eq_bind]
        rw [cast_real_real_eval "a" _ aT0 (by simp)]
        rw [show (sk.setReg "a" .real [BM, BK] aT0).regs .real [BM, BK] "a_sum"
              = some (⟨fun idx : TileIndex [BM, BK] =>
                  some (aSumPartial s0 A M N BM BN sam sak BK idx.1 idx.2.1 t)⟩ : Tile .real [BM, BK]) from by
            simp [hsk, hasum, s0]]
        simp only [Option.bind_some]
        refine congrArg some ?_
        ext idx
        simp only [hasumT1, haT0, Tile.bop_data, Tile.scalar_data,
          Broadcast.leftIndex_consSame, Broadcast.rightIndex_consSame,
          Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
          Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
          NumericDType.add, WithBot.realAdd, WithBot.realPow_some,
          Option.map₂, Option.bind, Option.map]
        rw [aSumPartial_succ, rpow_two_eq]))]
  -- 3: rms_w = tl.load(rms_w_ptrs)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show @evalOp .real [1, BK] (Op.decodeFloat8E5 .real (Op.load .int (.ptr (Op.ref .ptr [1, BK] "rms_w_ptrs")) MaskOpt.none))
          ((sk.setReg "a" .real [BM, BK] aT0).setReg "a_sum" .real [BM, BK] asumT1)
          = some rmsT from by
        rw [ByteLoads.load W1 W3 RMS _ "rms_w_ptrs" rmspT (by simp [hrmspk])
          (by intro i; simp [hrmspT])
          (by
            intro i
            apply (ByteLoads.finite_memory _ raw0 (by simpa only [BlockState.setReg_mem, hsk] using hrawmem) _ _).mpr
            simpa [hrmspT, s0, ByteLoads.view, colIndex, colGlobal] using hbytes.2.2 t hclt i.2.1)]
        refine congrArg some ?_
        ext idx
        simp only [hrmspT, ByteLoads.view_setReg, BlockState.setReg_readMem, hvread, Region.cast_id, hrmsT, rmsElem]
        rw [show idx.2.1.val * srms + t * BK * srms
              = (t * BK + idx.2.1.val) * srms from by ring]))]
  -- 4: a = a * rms_w
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.mul .real (Broadcast.consR (Broadcast.consSame Broadcast.nil))
            (Op.ref .real [BM, BK] "a") (Op.ref .real [1, BK] "rms_w")) _
          = some aT1 from by
        rw [evalOp_mul]
        simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
          String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
        refine congrArg some ?_
        ext idx
        simp only [haT0, hrmsT, haT1, Tile.bop_data,
          Broadcast.leftIndex_consR, Broadcast.leftIndex_consSame, Broadcast.leftIndex_nil,
          Broadcast.rightIndex_consR, Broadcast.rightIndex_consSame, Broadcast.rightIndex_nil,
          NumericDType.mul, WithBot.realMul, Option.map₂, Option.bind, Option.map]))]
  -- 5: b = tl.load(w1_ptrs)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show @evalOp .real [BK, BN] (Op.decodeFloat8E5 .real (Op.load .int (.ptr (Op.ref .ptr [BK, BN] "w1_ptrs")) MaskOpt.none)) _
          = some w1T from by
        rw [ByteLoads.load W1 W3 RMS _ "w1_ptrs" w1pT (by simp [hw1pk])
          (by intro i; simp [hw1pT])
          (by
            intro i
            apply (ByteLoads.finite_memory _ raw0 (by simpa only [BlockState.setReg_mem, hsk] using hrawmem) _ _).mpr
            simpa [hw1pT, s0, ByteLoads.view, colIndex, colGlobal] using hbytes.1 t hclt i.1 i.2.1)]
        refine congrArg some ?_
        ext idx
        simp only [hw1pT, ByteLoads.view_setReg, BlockState.setReg_readMem, hvread, Region.cast_id, hw1T, wElem]
        rw [show idx.1.val * sw1k + colIndex s0 N BN idx.2.1 * sw1n + t * BK * sw1k
              = (t * BK + idx.1.val) * sw1k + colIndex s0 N BN idx.2.1 * sw1n from by ring]))]
  -- 6: acc1 += tl.dot(a, b)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Op.ref .real [BM, BN] "acc1")
            (Op.dot (batch := []) (Op.ref .real [BM, BK] "a") (Op.ref .real [BK, BN] "b"))) _
          = some (⟨fun idx : TileIndex [BM, BN] =>
              some (accPartial s0 A W1 RMS M N BM BN sam sak sw1k sw1n srms BK idx.1 idx.2.1 (t + 1))⟩
              : Tile .real [BM, BN]) from by
        rw [accdot_eval BM BK BN _ "acc1" "b" acc1T aT1 w1T
            (by simp [hsk, hacc1, hacc1T]) (by simp) (by simp)]
        refine congrArg some ?_
        ext idx
        rw [adddot_lane BM BN acc1T (Tile.dot [] aT1 w1T) idx.1 idx.2.1
            (accPartial s0 A W1 RMS M N BM BN sam sak sw1k sw1n srms BK idx.1 idx.2.1 t)
            (Finset.univ.sum fun e : Fin BK =>
              (aElem s0 A M N BM BN sam sak idx.1 (t * BK + e.val)
                * rmsElem s0 RMS srms (t * BK + e.val))
                * wElem s0 W1 N BN sw1k sw1n idx.2.1 (t * BK + e.val))
            (by rw [hacc1T])
            (tile_dot_data BM BK BN aT1 w1T idx.1 idx.2.1 _ _
              (fun e => by rw [haT1]) (fun e => by rw [hw1T]))]
        show _ = some (accPartial s0 A W1 RMS M N BM BN sam sak sw1k sw1n srms BK idx.1 idx.2.1 (t + 1))
        rw [accPartial_succ]))]
  -- 7: c = tl.load(w3_ptrs)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show @evalOp .real [BK, BN] (Op.decodeFloat8E5 .real (Op.load .int (.ptr (Op.ref .ptr [BK, BN] "w3_ptrs")) MaskOpt.none)) _
          = some w3T from by
        rw [ByteLoads.load W1 W3 RMS _ "w3_ptrs" w3pT (by simp [hw3pk])
          (by intro i; simp [hw3pT])
          (by
            intro i
            apply (ByteLoads.finite_memory _ raw0 (by simpa only [BlockState.setReg_mem, hsk] using hrawmem) _ _).mpr
            simpa [hw3pT, s0, ByteLoads.view, colIndex, colGlobal] using hbytes.2.1 t hclt i.1 i.2.1)]
        refine congrArg some ?_
        ext idx
        simp only [hw3pT, ByteLoads.view_setReg, BlockState.setReg_readMem, hvread, Region.cast_id, hw3T, wElem]
        rw [show idx.1.val * sw3k + colIndex s0 N BN idx.2.1 * sw3n + t * BK * sw3k
              = (t * BK + idx.1.val) * sw3k + colIndex s0 N BN idx.2.1 * sw3n from by ring]))]
  -- 8: acc2 += tl.dot(a, c)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Op.ref .real [BM, BN] "acc2")
            (Op.dot (batch := []) (Op.ref .real [BM, BK] "a") (Op.ref .real [BK, BN] "c"))) _
          = some (⟨fun idx : TileIndex [BM, BN] =>
              some (accPartial s0 A W3 RMS M N BM BN sam sak sw3k sw3n srms BK idx.1 idx.2.1 (t + 1))⟩
              : Tile .real [BM, BN]) from by
        rw [accdot_eval BM BK BN _ "acc2" "c" acc2T aT1 w3T
            (by simp [hsk, hacc2, hacc2T]) (by simp) (by simp)]
        refine congrArg some ?_
        ext idx
        rw [adddot_lane BM BN acc2T (Tile.dot [] aT1 w3T) idx.1 idx.2.1
            (accPartial s0 A W3 RMS M N BM BN sam sak sw3k sw3n srms BK idx.1 idx.2.1 t)
            (Finset.univ.sum fun e : Fin BK =>
              (aElem s0 A M N BM BN sam sak idx.1 (t * BK + e.val)
                * rmsElem s0 RMS srms (t * BK + e.val))
                * wElem s0 W3 N BN sw3k sw3n idx.2.1 (t * BK + e.val))
            (by rw [hacc2T])
            (tile_dot_data BM BK BN aT1 w3T idx.1 idx.2.1 _ _
              (fun e => by rw [haT1]) (fun e => by rw [hw3T]))]
        show _ = some (accPartial s0 A W3 RMS M N BM BN sam sak sw3k sw3n srms BK idx.1 idx.2.1 (t + 1))
        rw [accPartial_succ]))]
  -- 9–12: the four pointer advances
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (ptr_adv_eval _ "a_ptrs" BK sak apT (by simp [hapk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (ptr_adv_eval _ "w1_ptrs" BK sw1k w1pT (by simp [hw1pk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (ptr_adv_eval _ "w3_ptrs" BK sw3k w3pT (by simp [hw3pk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (ptr_adv_eval _ "rms_w_ptrs" BK srms rmspT (by simp [hrmspk])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  change ffInvariant A W1 W3 RMS s0 M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms numKBlocks (t + 1) (ByteLoads.view W1 W3 RMS _) ∧ _
  constructor
  swap
  · simpa only [BlockState.setReg_mem, hsk] using hrawmem
  simp only [Invariant, ffInvariant, ByteLoads.view_setReg]
  refine ⟨by simpa [hsk, ByteLoads.view] using hpids, by omega, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simpa [hsk, ByteLoads.view, s0] using hpm
  · simpa [hsk, ByteLoads.view, s0] using hpn
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    rw [hasumT1]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.scalar, hapT, NumericDType.add]
    ring
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.scalar, hw1pT, NumericDType.add]
    ring
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.scalar, hw3pT, NumericDType.add]
    ring
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.scalar, hrmspT, NumericDType.add]
    ring
  · intro rg o; simpa [hsk, ByteLoads.view] using hundef rg o
  · show _ = s0.mem
    simpa only [BlockState.setReg_mem, ByteLoads.view_setReg, hsk] using hmem

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LlamaFfTriton.ByteExecution
