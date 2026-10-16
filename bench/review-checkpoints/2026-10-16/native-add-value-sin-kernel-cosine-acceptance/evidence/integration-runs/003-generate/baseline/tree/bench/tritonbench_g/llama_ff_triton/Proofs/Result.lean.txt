import bench.tritonbench_g.llama_ff_triton.Proofs.Source

namespace VeriTile.Bench.TritonBenchG.LlamaFfTriton.RealResult
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

private theorem scatter_frame {ι : Type} (indices : List ι) (OUT : RegionName)
    (offset : ι → Nat) (value : ι → ℝ) (mask : ι → Bool)
    (s : BlockState) (r : RegionName) (o : Nat)
    (h : r ≠ OUT ∨ ∀ i, mask i = Bool.true → o ≠ offset i) :
    (indices.foldl (fun t i => if mask i then t.writeMem OUT (offset i) (value i) else t) s).mem r o = s.mem r o := by
  induction indices generalizing s with
  | nil => rfl
  | cons i rest ih =>
    rw [List.foldl_cons, ih]
    by_cases hm : mask i = Bool.true
    · simp only [hm, ↓reduceIte]
      rcases h with hr | ho
      · simp [BlockState.writeMem, BlockState.writeCell, hr]
      · simp [BlockState.writeMem, BlockState.writeCell, ho i hm]
    · simp [hm]

/-- The actual erased output tail writes the independent formula on every
active lane and preserves every other typed cell. Injectivity is needed only
between active output lanes, so partial tiles and N < BN remain allowed. -/
theorem postLoop_of_registers (A W1 W3 OUT RMS : RegionName) (s0 : BlockState)
    (M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms soutm soutn numKBlocks K : Nat) (EPS : ℝ)
    (hInj : ∀ i j : TileIndex [BM, BN], active s0 M N BM BN i → active s0 M N BM BN j →
      outOffset s0 N BM BN soutm soutn i = outOffset s0 N BM BN soutm soutn j → i = j)
    (st : BlockState)
    (hinv : ffInvariant A W1 W3 RMS s0 M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms
      numKBlocks numKBlocks {st with mem := s0.mem}) :
    ∃ sfin, stepStmts (RealSource.tail Bool.false OUT M N BM BN BK K soutm soutn EPS) st = some sfin ∧
      (∀ idx : TileIndex [BM, BN], active s0 M N BM BN idx →
        sfin.readMem OUT (outOffset s0 N BM BN soutm soutn idx) =
          ffSpec s0 A W1 W3 RMS M N BM BN sam sak sw1k sw1n sw3k sw3n srms BK numKBlocks K EPS idx.1 idx.2.1) ∧
      (∀ r o, (r ≠ OUT ∨ ∀ idx : TileIndex [BM, BN], active s0 M N BM BN idx →
        o ≠ outOffset s0 N BM BN soutm soutn idx) → sfin.mem r o = st.mem r o) := by
  simp only [ffInvariant] at hinv
  obtain ⟨hpids, hcle, hpm, hpn, hacc1, hacc2, hasum, hap, hw1p, hw3p, hrmsp, hundef, hmem⟩ := hinv
  set nrm : Fin BM → ℝ :=
    fun i => normSpec s0 A M N BM BN sam sak BK numKBlocks K EPS i with hnrm
  set s1F : TileIndex [BM, BN] → ℝ :=
    fun idx => accPartial s0 A W1 RMS M N BM BN sam sak sw1k sw1n srms BK idx.1 idx.2.1 numKBlocks
    with hs1F
  set s3F : TileIndex [BM, BN] → ℝ :=
    fun idx => accPartial s0 A W3 RMS M N BM BN sam sak sw3k sw3n srms BK idx.1 idx.2.1 numKBlocks
    with hs3F
  set accT : Tile .real [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] =>
      some (ffSpec s0 A W1 W3 RMS M N BM BN sam sak sw1k sw1n sw3k sw3n srms
        BK numKBlocks K EPS idx.1 idx.2.1)⟩ with haccT
  set cpT : Tile .ptr [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] => (OUT.cast, outOffset s0 N BM BN soutm soutn idx)⟩ with hcpT
  set cmaskT : Tile .bool [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] =>
      (decide (rowGlobal s0 N BM BN idx.1 < M) && decide (colGlobal s0 N BN idx.2.1 < N))⟩
    with hcmaskT
  simp only [RealSource.tail, Bool.false_eq_true, ↓reduceIte, ffStoreTail,
    List.map_cons, List.map_nil, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, Region.cast_id,
    eq_mpr_eq_cast, eq_mp_eq_cast, cast_eq]
  -- a_mean
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (amean_eval BM BK K EPS st
        (fun i e => aSumPartial s0 A M N BM BN sam sak BK i e numKBlocks) (by rw [hasum])))]
  -- a_norm (canonicalized to `normSpec` via `Finset.sum_comm`)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.rsqrt (Op.ref .real [BM] "a_mean")) _
          = some (⟨fun idx : TileIndex [BM] => some (nrm idx.1)⟩ : Tile .real [BM]) from by
        rw [anorm_eval BM _
          (fun i => (∑ e : Fin BK, aSumPartial s0 A M N BM BN sam sak BK i e numKBlocks)
            / (K : ℝ) + EPS) (by simp)]
        refine congrArg some ?_
        ext idx
        simp only [hnrm, normSpec, aSumPartial]
        rw [Finset.sum_comm]))]
  -- acc1 = acc1 * a_norm[:, None]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (accscale_eval BM BN _ "acc1" s1F nrm (by simp [hacc1, hs1F]) (by simp)))]
  -- acc2 = acc2 * a_norm[:, None]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (accscale_eval BM BN _ "acc2" s3F nrm (by simp [hacc2, hs3F]) (by simp)))]
  -- accumulator = (acc1 * tl.sigmoid(acc1)) * acc2  (canonicalized to `ffSpec`)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
              (Op.ref .real [BM, BN] "acc1")
              (Op.sigmoid (Op.ref .real [BM, BN] "acc1")))
            (Op.ref .real [BM, BN] "acc2")) _
          = some accT from by
        rw [gate_eval BM BN _
          (fun idx => s1F idx * nrm idx.1) (fun idx => s3F idx * nrm idx.1)
          (by simp) (by simp)]
        refine congrArg some ?_
        ext idx
        simp only [haccT, ffSpec, hs1F, hs3F, hnrm]
        exact congrArg some (gate_lane_eq
          (normSpec s0 A M N BM BN sam sak BK numKBlocks K EPS idx.1)
          (accPartial s0 A W1 RMS M N BM BN sam sak sw1k sw1n srms BK idx.1 idx.2.1 numKBlocks)
          (accPartial s0 A W3 RMS M N BM BN sam sak sw3k sw3n srms BK idx.1 idx.2.1 numKBlocks))))]
  -- offs_outm / offs_outn
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (offsout_eval _ BM BM (pidM (s0.pids 0) N BN) "pid_m" (by simp [hpm])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (offsout_eval _ BN BN (pidN (s0.pids 0) N BN) "pid_n" (by simp [hpn])))]
  -- out_ptrs
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase OUT)
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.mul .nat Broadcast.scalarL (Op.constNat soutm) (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_outm")))
            (Op.mul .nat Broadcast.scalarL (Op.constNat soutn) (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_outn"))))) _
          = some cpT from by
        rw [outptrs_eval _ OUT BM BN soutm soutn
          (fun i => rowGlobal s0 N BM BN i) (fun j => colGlobal s0 N BN j)
          (by simp [rowGlobal]) (by simp [colGlobal])]
        rfl))]
  -- out_mask
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_outm")) (Op.constNat M))
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_outn")) (Op.constNat N))) _
          = some cmaskT from by
        rw [outmask_eval _ M N BM BN
          (fun i => rowGlobal s0 N BM BN i) (fun j => colGlobal s0 N BN j)
          (by simp [rowGlobal]) (by simp [colGlobal])]))]
  -- abstract the post-assign state
  generalize hst9 : (((((((((st.setReg "a_mean" .real [BM]
      (⟨fun idx : TileIndex [BM] =>
        some ((∑ e : Fin BK, aSumPartial s0 A M N BM BN sam sak BK idx.1 e numKBlocks)
          / (K : ℝ) + EPS)⟩ : Tile .real [BM])).setReg "a_norm" .real [BM]
      (⟨fun idx : TileIndex [BM] => some (nrm idx.1)⟩ : Tile .real [BM])).setReg
      "acc1" .real [BM, BN]
      (⟨fun idx : TileIndex [BM, BN] => some (s1F idx * nrm idx.1)⟩ : Tile .real [BM, BN])).setReg
      "acc2" .real [BM, BN]
      (⟨fun idx : TileIndex [BM, BN] => some (s3F idx * nrm idx.1)⟩ : Tile .real [BM, BN])).setReg
      "accumulator" .real [BM, BN] accT).setReg "offs_outm" .nat [BM]
      (Tile.vec fun i : Fin BM => pidM (s0.pids 0) N BN * BM + i.val)).setReg "offs_outn" .nat [BN]
      (Tile.vec fun j : Fin BN => pidN (s0.pids 0) N BN * BN + j.val)).setReg
      "out_ptrs" .ptr [BM, BN] cpT).setReg "out_mask" .bool [BM, BN] cmaskT) = st9
  have hacc9 : st9.regs .real [BM, BN] "accumulator" = some accT := by rw [← hst9]; simp
  have hcp9 : st9.regs .ptr [BM, BN] "out_ptrs" = some cpT := by rw [← hst9]; simp
  have hcm9 : st9.regs .bool [BM, BN] "out_mask" = some cmaskT := by rw [← hst9]; simp
  have hmem9 : st9.mem = st.mem := by
    rw [← hst9]; funext region offset; simp only [BlockState.setReg_mem]
  have hstore : stepStmt (.store .real [BM, BN] (.ptr (.ref .ptr [BM, BN] "out_ptrs"))
      (.ref .real [BM, BN] "accumulator") (.mask (.ref .bool [BM, BN] "out_mask"))) st9 =
      some ((TileShape.allIndices [BM, BN]).foldl (fun t i =>
        if cmaskT.data i then t.writeMem OUT (outOffset s0 N BM BN soutm soutn i)
          (ffSpec s0 A W1 W3 RMS M N BM BN sam sak sw1k sw1n sw3k sw3n srms BK numKBlocks K EPS i.1 i.2.1)
        else t) st9) := by
    simp only [stepStmt, evalOp_ref, hacc9, hcp9, hcm9, bind, Option.bind_some,
      BlockState.writeMemTyped_real]
    congr 1
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_⟩
  · intro idx hact
    have hm : cmaskT.data idx = Bool.true := by
      simpa [hcmaskT, active, rowGlobal, colGlobal] using hact
    apply BlockState.scatter_readback_prop_masked_nd_of_true
      (P := fun i => cmaskT.data i = Bool.true) _ _ _ idx hm
    intro j hj he
    apply hInj j idx _ hact he
    simpa [hcmaskT, active, rowGlobal, colGlobal] using hj
  · intro r o h
    rw [scatter_frame, hmem9]
    rcases h with hr | ho
    · exact Or.inl hr
    · exact Or.inr (fun i hi => ho i (by simpa [hcmaskT, active, rowGlobal, colGlobal] using hi))

/-- Specialization when the real input memory is retained by the loop. -/
theorem postLoop (A W1 W3 OUT RMS : RegionName) (s0 : BlockState)
    (M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms soutm soutn numKBlocks K : Nat) (EPS : ℝ)
    (hInj : ∀ i j : TileIndex [BM, BN], active s0 M N BM BN i → active s0 M N BM BN j →
      outOffset s0 N BM BN soutm soutn i = outOffset s0 N BM BN soutm soutn j → i = j)
    (st : BlockState)
    (hinv : ffInvariant A W1 W3 RMS s0 M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms
      numKBlocks numKBlocks st) :
    ∃ sfin, stepStmts (RealSource.tail Bool.false OUT M N BM BN BK K soutm soutn EPS) st = some sfin ∧
      (∀ idx : TileIndex [BM, BN], active s0 M N BM BN idx →
        sfin.readMem OUT (outOffset s0 N BM BN soutm soutn idx) =
          ffSpec s0 A W1 W3 RMS M N BM BN sam sak sw1k sw1n sw3k sw3n srms BK numKBlocks K EPS idx.1 idx.2.1) ∧
      (∀ r o, (r ≠ OUT ∨ ∀ idx : TileIndex [BM, BN], active s0 M N BM BN idx →
        o ≠ outOffset s0 N BM BN soutm soutn idx) → sfin.mem r o = st.mem r o) := by
  apply postLoop_of_registers A W1 W3 OUT RMS s0 M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms soutm soutn numKBlocks K EPS hInj st
  rcases hinv with ⟨hpids, hcle, hpm, hpn, hacc1, hacc2, hasum, hap, hw1p, hw3p, hrmsp, hundef, _⟩
  exact ⟨hpids, hcle, hpm, hpn, hacc1, hacc2, hasum, hap, hw1p, hw3p, hrmsp, hundef, rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LlamaFfTriton.RealResult
