import bench.tritonbench_g.rms_matmul_rbe.Proofs.Real
import VeriTile.Triton.Memory.TraceProofs

namespace VeriTile.Bench.TritonBenchG.RmsMatmulRbe.AccessSafety
open VeriTile.Triton
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- All lanes of an unmasked source load belong to the allocation. -/
theorem load_safe (bounds : RegionBounds) (s : BlockState) (name : RegName)
    (d : TileDType) (p : Tile .ptr sh)
    (hp : s.regs .ptr sh name = some p)
    (hb : ∀ i, (p.data i).2 < bounds (p.data i).1) :
    Op.SafeAt bounds s (.load d (.ptr (.ref .ptr sh name)) .none) := by
  simp only [Op.SafeAt.eq_def, true_and, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, Op.PointerAddressesSafeOn, Op.MemorySafe.eq_def,
    MaskOpt.Active]
  intro ptrs he i _
  rw [evalOp_ref, hp] at he
  cases he
  exact hb i

/-- Safety of one complete source iteration in both weight arms. A failing
byte decoder performs no subsequent accesses; its success is not assumed. -/
theorem iteration (fp8 : Bool) (bounds : RegionBounds) (s : BlockState)
    (BM BN BK sxk swk srms : Nat)
    (xp : Tile .ptr [BM, BK]) (wp : Tile .ptr [BK, BN]) (rp : Tile .ptr [1, BK])
    (hx : s.regs .ptr [BM, BK] "x_ptrs" = some xp)
    (hw : s.regs .ptr [BK, BN] "w_ptrs" = some wp)
    (hr : s.regs .ptr [1, BK] "rms_w_ptrs" = some rp)
    (bx : ∀ i, (xp.data i).2 < bounds (xp.data i).1)
    (bw : ∀ i, (wp.data i).2 < bounds (wp.data i).1)
    (br : ∀ i, (rp.data i).2 < bounds (rp.data i).1) :
    Stmt.TraceSafeList bounds (RealSource.loop fp8 BM BN BK sxk swk srms) s := by
  cases fp8
  all_goals
    simp only [RealSource.loop, rrmLoopBody, List.map_cons, List.map_nil,
      List.cons_append, List.nil_append, List.flatMap_cons, List.flatMap_nil,
      RealSource.weightLoad, Bool.false_and, Bool.true_and, Bool.false_eq_true,
      Bool.or_eq_true, beq_iff_eq, String.reduceEq, false_or, or_false, ↓reduceIte,
      Stmt.eraseDType, Op.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      MemAccess.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
      IntegralDType.eraseDType, ComparableDType.eraseDType, FloatDType.toTileDType]
    apply TraceProofs.assign_cons
    · exact load_safe bounds s "x_ptrs" .real xp hx bx
    · intro xtile
      apply TraceProofs.assign_cons
      · simp [Op.SafeAt.eq_def]
      · intro sumtile
        apply TraceProofs.assign_cons
        · exact load_safe bounds _ "rms_w_ptrs" _ rp (by simpa using hr) br
        · intro rtile
          repeat' (apply TraceProofs.assign_cons; (solve | simp [Op.SafeAt.eq_def]); intro)
          apply TraceProofs.assign_cons
          · exact load_safe bounds _ "w_ptrs" _ wp (by simpa using hw) bw
          · intro wtile
            apply TraceProofs.all_safe
            simp [Stmt.TraceSafe, Op.SafeAt.eq_def]

theorem prologue (bounds : RegionBounds) (s : BlockState) (X W RMS : RegionName)
    (M N BM BN BK sxb sxm sxk swk swn srms : Nat) :
    Stmt.TraceSafeList bounds
      (rrmPrologue X W RMS M N BM BN BK sxb sxm sxk swk swn srms) s := by
  apply TraceProofs.all_safe
  simp [rrmPrologue, Stmt.TraceSafe, Op.SafeAt.eq_def]

/-- Explicit bounds include the batch offset, wrapped inactive M/N lanes
and every unmasked lane of the ceiling-counted K traversal. -/
def Reads (bounds : RegionBounds) (s : BlockState) (X W RMS : RegionName)
    (M N K BM BN BK sxb sxm sxk swk swn srms : Nat) : Prop :=
  (∀ t < cdiv K BK, ∀ (i : Fin BM) (e : Fin BK),
    s.pids 0 * sxb + rowIndex s M N BM BN i * sxm + (t * BK + e.val) * sxk < bounds X) ∧
  (∀ t < cdiv K BK, ∀ (e : Fin BK) (j : Fin BN),
    (t * BK + e.val) * swk + colIndex s N BN j * swn < bounds W) ∧
  (∀ t < cdiv K BK, ∀ e : Fin BK, (t * BK + e.val) * srms < bounds RMS)

theorem iteration_of_invariant (fp8 : Bool) (bounds : RegionBounds)
    (X W RMS : RegionName) (initial s : BlockState)
    (M N K BM BN BK sxb sxm sxk swk swn srms t : Nat)
    (ht : t < cdiv K BK)
    (h : rrmInvariant X W RMS initial M N BM BN BK sxb sxm sxk swk swn srms (cdiv K BK) t s)
    (hb : Reads bounds initial X W RMS M N K BM BN BK sxb sxm sxk swk swn srms) :
    Stmt.TraceSafeList bounds (RealSource.loop fp8 BM BN BK sxk swk srms)
      (s.setReg "_i" .nat [] (Tile.scalar t)) := by
  obtain ⟨_, _, _, _, _, _, _, hx, hw, hr, _, _⟩ := h
  apply iteration fp8 bounds _ BM BN BK sxk swk srms _ _ _
    (by simpa using hx) (by simpa using hw) (by simpa using hr)
  · intro i
    simpa [Nat.add_mul, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hb.1 t ht i.1 i.2.1
  · intro i
    simpa [Nat.add_mul, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hb.2.1 t ht i.1 i.2.1
  · intro i
    simpa [Nat.add_mul, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hb.2.2 t ht i.2.1

/-- Only active output addresses require bounds. The actual tail preserves
the batch component while rebuilding the unwrapped row and column indices. -/
theorem tail (variant : Bool) (bounds : RegionBounds) (initial s : BlockState)
    (OUT : RegionName) (M N BM BN BK K sob som son : Nat) (EPS : ℝ)
    (hpb : s.regs .nat [] "pid_batch" = some (Tile.scalar (initial.pids 0)))
    (hpm : s.regs .nat [] "pid_m" = some (Tile.scalar (pidM (initial.pids 1) N BN)))
    (hpn : s.regs .nat [] "pid_n" = some (Tile.scalar (pidN (initial.pids 1) N BN)))
    (hb : ∀ i : TileIndex [BM, BN], active initial M N BM BN i →
      outOffset initial N BM BN sob som son i < bounds OUT) :
    Stmt.TraceSafeList bounds (RealSource.tail variant OUT M N BM BN BK K sob som son EPS) s := by
  cases variant
  all_goals
    simp only [RealSource.tail, Bool.false_eq_true, ↓reduceIte, rrmStoreTail,
      List.map_cons, List.map_nil, List.take_succ_cons, List.take_zero,
      List.drop_succ_cons, List.drop_zero, List.cons_append, List.nil_append,
      Stmt.eraseDType, Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
      MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType,
      FloatDType.toTileDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      Region.cast_id, eq_mpr_eq_cast, eq_mp_eq_cast, cast_eq]
    iterate 3 (apply TraceProofs.assign_cons; (solve | simp [Op.SafeAt.eq_def]); intro)
    apply TraceProofs.assign_eval_cons
    · simp [Op.SafeAt.eq_def]
    · intro om hom
      try rw [RealTransport.output_index] at hom
      rw [offsout_eval _ BM BM (pidM (initial.pids 1) N BN) "pid_m" (by simpa using hpm), Option.some.injEq] at hom
      subst om
      apply TraceProofs.assign_eval_cons
      · simp [Op.SafeAt.eq_def]
      · intro on hon
        rw [offsout_eval _ BN BN (pidN (initial.pids 1) N BN) "pid_n" (by simpa using hpn), Option.some.injEq] at hon
        subst on
        apply TraceProofs.assign_eval_cons
        · simp [Op.SafeAt.eq_def]
        · intro ptrs hptr
          rw [batchptrs_eval _ OUT BM BN sob som son (initial.pids 0) "offs_m" "offs_n"
            (fun i => rowGlobal initial N BM BN i) (fun j => colGlobal initial N BN j)
            (by simpa using hpb) (by simp [rowGlobal]) (by simp [colGlobal]), Option.some.injEq] at hptr
          subst ptrs
          apply TraceProofs.assign_eval_cons
          · simp [Op.SafeAt.eq_def]
          · intro masks hmasks
            rw [outmask_eval _ M N BM BN
              (fun i => rowGlobal initial N BM BN i) (fun j => colGlobal initial N BN j)
              (by simp [rowGlobal]) (by simp [colGlobal]), Option.some.injEq] at hmasks
            subst masks
            apply Stmt.TraceSafeList.cons_intro
            · simp only [Stmt.TraceSafe, Op.SafeAt.eq_def, MemAccess.SafeAt.eq_def,
                MaskOpt.SafeAt.eq_def, true_and, MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
                Op.PointerAddressesSafeOn, Op.MemorySafe.eq_def, MaskOpt.Active]
              intro ptrs hp i hi
              simp only [evalOp_ref, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
                not_false_eq_true, BlockState.setReg_same] at hp
              cases hp
              rcases hi with ⟨masks, hm, hi⟩
              simp only [evalOp_ref, BlockState.setReg_same] at hm
              cases hm
              exact hb i (by simpa [active] using hi)
            · intro _ _
              exact Stmt.TraceSafeList.nil_intro

/-- The actual non-FP8 source is safe for all admitted accesses. The loop
invariant follows the dynamic ceiling count, including the unmasked padding. -/
theorem complete_half (input : TypedSource.Storage) (outputFP8 variant : Bool) (bounds : RegionBounds)
    (X W RMS OUT : RegionName) (s : BlockState)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK : Nat)
    (start : Int) (rbe : Bool) (theta EPS : ℝ)
    (hu : ∀ rg o, s.undef rg o = 0)
    (hb : Reads bounds s X W RMS M N K BM BN BK sxb sxm sxk swk swn srms)
    (ho : ∀ i : TileIndex [BM, BN], active s M N BM BN i →
      outOffset s N BM BN sob som son i < bounds OUT) :
    (TypedSource.kernel input Bool.false outputFP8 variant X W RMS OUT M N K sxb sxm sxk swk swn srms
      sob som son BM BN BK start rbe theta EPS).eraseDType.toAlgKernel.TraceSafe bounds s := by
  obtain ⟨s0, hpre, hP0⟩ := preLoop X W RMS s M N sxb sxm sxk swk swn srms
    BM BN BK (cdiv K BK) hu
  have hpre' : stepStmts (rrmPrologue X W RMS M N BM BN BK sxb sxm sxk swk swn srms) s = some s0 := hpre
  obtain ⟨final, sLoop, hLoop, hfinal, hPLoop⟩ :=
    forRange_inv (idx := "_i") (start := 0) (stop := cdiv K BK) (step := 1)
      (by omega) hP0
      (fun t st hlt hinv => by
        have := rrm_step X W RMS s M N BM BN BK sxb sxm sxk swk swn srms
          (cdiv K BK) t st hlt hinv
        simpa using this)
  have hf : final = cdiv K BK := le_antisymm hPLoop.2.1 hfinal
  rw [hf] at hPLoop
  have hloop : stepStmt (.forRangeDyn "_i" (.constNat 0) (RealSource.ceilDivOp K BK)
      (.constNat 1) (RealSource.loop Bool.false BM BN BK sxk swk srms)) s0 = some sLoop := by
    simp only [stepForRangeAux.forRangeDyn_unfold, evalOp_constNat,
      RealTransport.ceil_eval, Option.bind_some, Tile.scalar_data]
    rw [ProgramRewrite.range_congr _ _ _ _ _ _ (RealTransport.loop BM BN BK sxk swk srms)]
    simpa only [stepStmt] using hLoop
  rw [ComputeKernel.toAlgKernel, RealSource.projection]
  change Stmt.TraceSafeList bounds (RealSource.body Bool.false variant X W RMS OUT M N K
    sxb sxm sxk swk swn srms sob som son BM BN BK EPS) s
  rw [RealSource.body, RealTransport.prologue]
  apply TraceProofs.append_safe _ s (prologue bounds s X W RMS M N BM BN BK sxb sxm sxk swk swn srms)
  intro t ht
  have he : t = s0 := Option.some.inj (ht.symm.trans hpre')
  subst t
  rw [TraceProofs.cons_of_run_iff hloop]
  refine ⟨?_, tail variant bounds s sLoop OUT M N BM BN BK K sob som son EPS
    hPLoop.2.2.1 hPLoop.2.2.2.1 hPLoop.2.2.2.2.1 ho⟩
  rw [Stmt.TraceSafe]
  refine ⟨by simp [Op.SafeAt.eq_def], by simp [RealSource.ceilDivOp, Op.SafeAt.eq_def],
    by simp [Op.SafeAt.eq_def], ?_⟩
  simp only [evalOp_constNat, RealTransport.ceil_eval, Tile.scalar_data]
  apply TraceProofs.range_safe (P := rrmInvariant X W RMS s M N BM BN BK sxb sxm sxk swk swn srms (cdiv K BK)) ?_ ?_ 0 s0 hP0
  · intro i st hi hinv
    exact iteration_of_invariant Bool.false bounds X W RMS s st M N K BM BN BK sxb sxm sxk swk swn srms i hi hinv hb
  · intro i st st' hi hinv hr
    rw [RealTransport.loop] at hr
    obtain ⟨u, he, hu⟩ := rrm_step X W RMS s M N BM BN BK sxb sxm sxk swk swn srms (cdiv K BK) i st hi hinv
    have ht : st' = u := Option.some.inj (hr.symm.trans he)
    subst st'
    exact hu

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsMatmulRbe.AccessSafety
