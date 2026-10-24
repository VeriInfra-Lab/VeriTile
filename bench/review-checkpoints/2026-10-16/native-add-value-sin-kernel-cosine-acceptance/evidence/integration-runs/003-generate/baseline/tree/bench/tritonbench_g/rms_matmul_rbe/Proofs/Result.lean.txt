import bench.tritonbench_g.rms_matmul_rbe.Proofs.Source

namespace VeriTile.Bench.TritonBenchG.RmsMatmulRbe.RealResult
open VeriTile.Triton
set_option maxHeartbeats 8000000
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

private theorem scatter_metadata {ι : Type} (indices : List ι) (OUT : RegionName)
    (offset : ι → Nat) (value : ι → ℝ) (mask : ι → Bool) (s : BlockState) :
    let t := indices.foldl (fun t i => if mask i then t.writeMem OUT (offset i) (value i) else t) s
    t.pids = s.pids ∧ t.undef = s.undef := by
  induction indices generalizing s with
  | nil => exact ⟨rfl, rfl⟩
  | cons i rest ih =>
    simp only [List.foldl_cons]
    have h := ih (if mask i then s.writeMem OUT (offset i) (value i) else s)
    cases hi : mask i <;> simpa [hi, BlockState.writeMem, BlockState.writeCell] using h

/-- The actual real output tail succeeds, returns the independent padded-input
formula on every active lane, and frames every other raw cell. Only active
output lanes must have distinct addresses; N may be smaller than BN. -/
theorem postLoop_of_registers (X W RMS OUT : RegionName) (s0 : BlockState)
    (M N BM BN BK sxb sxm sxk swk swn srms sob som son numKBlocks K : Nat) (EPS : ℝ)
    (hInj : ∀ i j : TileIndex [BM, BN], active s0 M N BM BN i → active s0 M N BM BN j →
      outOffset s0 N BM BN sob som son i = outOffset s0 N BM BN sob som son j → i = j)
    (st : BlockState)
    (hinv : rrmInvariant X W RMS s0 M N BM BN BK sxb sxm sxk swk swn srms
      numKBlocks numKBlocks {st with mem := s0.mem}) :
    ∃ sfin, stepStmts (RealSource.tail Bool.false OUT M N BM BN BK K sob som son EPS) st = some sfin ∧
      sfin.pids = st.pids ∧ sfin.undef = st.undef ∧
      (∀ idx : TileIndex [BM, BN], active s0 M N BM BN idx →
        sfin.readMem OUT (outOffset s0 N BM BN sob som son idx) =
          rmsSpec s0 X W RMS M N BM BN sxb sxm sxk swk swn srms BK numKBlocks K EPS idx.1 idx.2.1) ∧
      (∀ r o, (r ≠ OUT ∨ ∀ idx : TileIndex [BM, BN], active s0 M N BM BN idx →
        o ≠ outOffset s0 N BM BN sob som son idx) → sfin.mem r o = st.mem r o) := by
  simp only [rrmInvariant] at hinv
  obtain ⟨hpids, hcle, hpb, hpm, hpn, hacc, hxsum, hxp, hwp, hrmsp, hundef, hmem⟩ := hinv
  set nrm : Fin BM → ℝ :=
    fun i => normSpec s0 X M N BM BN sxb sxm sxk BK numKBlocks K EPS i with hnrm
  set sF : TileIndex [BM, BN] → ℝ :=
    fun idx => accPartial s0 X W RMS M N BM BN sxb sxm sxk swk swn srms BK idx.1 idx.2.1 numKBlocks
    with hsF
  set accT : Tile .real [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] =>
      some (rmsSpec s0 X W RMS M N BM BN sxb sxm sxk swk swn srms
        BK numKBlocks K EPS idx.1 idx.2.1)⟩ with haccT
  set cpT : Tile .ptr [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] => (OUT.cast, outOffset s0 N BM BN sob som son idx)⟩ with hcpT
  set cmaskT : Tile .bool [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] =>
      (decide (rowGlobal s0 N BM BN idx.1 < M) && decide (colGlobal s0 N BN idx.2.1 < N))⟩
    with hcmaskT
  simp only [RealSource.tail, Bool.false_eq_true, ↓reduceIte, rrmStoreTail,
    List.map_cons, List.map_nil, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, Region.cast_id,
    eq_mpr_eq_cast, eq_mp_eq_cast, cast_eq]
  -- x_mean
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (xmean_eval BM BK K EPS st
        (fun i e => xSumPartial s0 X M N BM BN sxb sxm sxk BK i e numKBlocks) (by rw [hxsum])))]
  -- x_norm (canonicalized to `normSpec` via `Finset.sum_comm`)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.rsqrt (Op.ref .real [BM] "x_mean")) _
          = some (⟨fun idx : TileIndex [BM] => some (nrm idx.1)⟩ : Tile .real [BM]) from by
        rw [xnorm_eval BM _
          (fun i => (∑ e : Fin BK, xSumPartial s0 X M N BM BN sxb sxm sxk BK i e numKBlocks)
            / (K : ℝ) + EPS) (by simp)]
        refine congrArg some ?_
        ext idx
        simp only [hnrm, normSpec, xSumPartial]
        rw [Finset.sum_comm]))]
  -- accumulator = accumulator * x_norm[:, None]  (canonicalized to `rmsSpec`)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
            (Op.ref .real [BM, BN] "accumulator")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "x_norm"))) _
          = some accT from by
        rw [accscale_eval BM BN _ "accumulator" sF nrm (by simp [hacc, hsF]) (by simp)]
        refine congrArg some ?_
        ext idx
        simp only [haccT, rmsSpec, hsF, hnrm]
        exact congrArg some (mul_comm _ _)))]
  -- offs_m / offs_n (tail re-derivation, un-wrapped)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (offsout_eval _ BM BM (pidM (s0.pids 1) N BN) "pid_m" (by simp [hpm])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (offsout_eval _ BN BN (pidN (s0.pids 1) N BN) "pid_n" (by simp [hpn])))]
  -- out_ptrs (with the batch offset)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase OUT)
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_batch") (Op.constNat sob))
              (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat som)))
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.constNat son)))) _
          = some cpT from by
        rw [batchptrs_eval _ OUT BM BN sob som son (s0.pids 0) "offs_m" "offs_n"
          (fun i => rowGlobal s0 N BM BN i) (fun j => colGlobal s0 N BN j)
          (by simp [hpb]) (by simp [rowGlobal]) (by simp [colGlobal])]
        rfl))]
  -- out_mask
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat M))
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.constNat N))) _
          = some cmaskT from by
        rw [outmask_eval _ M N BM BN
          (fun i => rowGlobal s0 N BM BN i) (fun j => colGlobal s0 N BN j)
          (by simp [rowGlobal]) (by simp [colGlobal])]))]
  -- abstract the post-assign state
  generalize hst8 : (((((((st.setReg "x_mean" .real [BM]
      (⟨fun idx : TileIndex [BM] =>
        some ((∑ e : Fin BK, xSumPartial s0 X M N BM BN sxb sxm sxk BK idx.1 e numKBlocks)
          / (K : ℝ) + EPS)⟩ : Tile .real [BM])).setReg "x_norm" .real [BM]
      (⟨fun idx : TileIndex [BM] => some (nrm idx.1)⟩ : Tile .real [BM])).setReg
      "accumulator" .real [BM, BN] accT).setReg "offs_m" .nat [BM]
      (Tile.vec fun i : Fin BM => pidM (s0.pids 1) N BN * BM + i.val)).setReg "offs_n" .nat [BN]
      (Tile.vec fun j : Fin BN => pidN (s0.pids 1) N BN * BN + j.val)).setReg
      "out_ptrs" .ptr [BM, BN] cpT).setReg "out_mask" .bool [BM, BN] cmaskT) = st8
  have hacc8 : st8.regs .real [BM, BN] "accumulator" = some accT := by rw [← hst8]; simp
  have hcp8 : st8.regs .ptr [BM, BN] "out_ptrs" = some cpT := by rw [← hst8]; simp
  have hcm8 : st8.regs .bool [BM, BN] "out_mask" = some cmaskT := by rw [← hst8]; simp
  have hmem8 : st8.mem = st.mem := by
    rw [← hst8]; funext region offset; simp only [BlockState.setReg_mem]
  have hstore : stepStmt (.store .real [BM, BN] (.ptr (.ref .ptr [BM, BN] "out_ptrs"))
      (.ref .real [BM, BN] "accumulator") (.mask (.ref .bool [BM, BN] "out_mask"))) st8 =
      some ((TileShape.allIndices [BM, BN]).foldl (fun t i =>
        if cmaskT.data i then t.writeMem OUT (outOffset s0 N BM BN sob som son i)
          (rmsSpec s0 X W RMS M N BM BN sxb sxm sxk swk swn srms BK numKBlocks K EPS i.1 i.2.1)
        else t) st8) := by
    simp only [stepStmt, evalOp_ref, hacc8, hcp8, hcm8, bind, Option.bind_some,
      BlockState.writeMemTyped_real]
    apply congrArg some
    apply List.foldl_ext
    intro t i _
    simp [hcpT, haccT, Region.cast_id, FloatDType.storeValue]
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_⟩
  · rw [(scatter_metadata _ _ _ _ _ st8).1, ← hst8]
    rfl
  · rw [(scatter_metadata _ _ _ _ _ st8).2, ← hst8]
    rfl
  · intro idx hact
    have hm : cmaskT.data idx = Bool.true := by
      simpa [hcmaskT, active, rowGlobal, colGlobal] using hact
    apply BlockState.scatter_readback_prop_masked_nd_of_true
      (P := fun i => cmaskT.data i = Bool.true) _ _ _ idx hm
    intro j hj he
    apply hInj j idx _ hact he
    simpa [hcmaskT, active, rowGlobal, colGlobal] using hj
  · intro r o h
    rw [scatter_frame, hmem8]
    rcases h with hr | ho
    · exact Or.inl hr
    · exact Or.inr (fun i hi => ho i (by simpa [hcmaskT, active, rowGlobal, colGlobal] using hi))

/-- Specialization for an unchanged mathematical input memory. -/
theorem postLoop (X W RMS OUT : RegionName) (s0 : BlockState)
    (M N BM BN BK sxb sxm sxk swk swn srms sob som son numKBlocks K : Nat) (EPS : ℝ)
    (hInj : ∀ i j : TileIndex [BM, BN], active s0 M N BM BN i → active s0 M N BM BN j →
      outOffset s0 N BM BN sob som son i = outOffset s0 N BM BN sob som son j → i = j)
    (st : BlockState)
    (hinv : rrmInvariant X W RMS s0 M N BM BN BK sxb sxm sxk swk swn srms
      numKBlocks numKBlocks st) :
    ∃ sfin, stepStmts (RealSource.tail Bool.false OUT M N BM BN BK K sob som son EPS) st = some sfin ∧
      sfin.pids = st.pids ∧ sfin.undef = st.undef ∧
      (∀ idx : TileIndex [BM, BN], active s0 M N BM BN idx →
        sfin.readMem OUT (outOffset s0 N BM BN sob som son idx) =
          rmsSpec s0 X W RMS M N BM BN sxb sxm sxk swk swn srms BK numKBlocks K EPS idx.1 idx.2.1) ∧
      (∀ r o, (r ≠ OUT ∨ ∀ idx : TileIndex [BM, BN], active s0 M N BM BN idx →
        o ≠ outOffset s0 N BM BN sob som son idx) → sfin.mem r o = st.mem r o) := by
  apply postLoop_of_registers X W RMS OUT s0 M N BM BN BK sxb sxm sxk swk swn srms
    sob som son numKBlocks K EPS hInj st
  rcases hinv with ⟨hpids, hcle, hpb, hpm, hpn, hacc, hxsum, hxp, hwp, hrmsp, hundef, _⟩
  exact ⟨hpids, hcle, hpb, hpm, hpn, hacc, hxsum, hxp, hwp, hrmsp, hundef, rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsMatmulRbe.RealResult
