import bench.repairs.tritonbench_g.attention_llama.Proofs.RealKernel
import bench.tritonbench_g.attention_llama.Proofs.RealSafetyPrologue
import VeriTile.Triton.Memory.TraceRewrite

namespace VeriTile.Bench.Repairs.AttentionLlama.AccessSafety
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionLlama
open VeriTile.Bench.TritonBenchG.AttentionLlama.RealKernel
open LoopInvariant SourceExecution MaskCoverage
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

theorem prefix_safe (bounds : RegionBounds) (sc : ℝ) (N BM BN BD SN : Nat)
    (s : BlockState) (K : RegionName) (base skn skk : Nat)
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hoff : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hptr : s.regs .ptr [BN, BD] "k_ptrs" = some (alPtrTile K base skn skk SN BN BD))
    (hb : ∀ i : TileIndex [BN, BD], SN + i.1.val < N →
      base + (SN + i.1.val) * skn + i.2.1.val * skk < bounds K) :
    Stmt.TraceSafeList bounds (SourceExecution.loopPrefix sc N BM BN BD) s := by
  unfold SourceExecution.loopPrefix
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
        (Op.ref .nat [BN] "offs_n")) s
        = some (Tile.vec (fun j : Fin BN => SN + j.val)) from by
      rw [evalOp_add]
      simp only [evalOp_ref, hsn, hoff, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext j
      simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, castTile_self,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add]))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
  apply TraceProofs.assign_cons
  · simp only [Op.SafeAt.eq_def, true_and]
    simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, Op.PointerAddressesSafeOn, Op.MemorySafe.eq_def, true_and, MaskOpt.Active]
    intro ptrs hp i hi
    rw [evalOp_ref, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hptr] at hp
    cases hp
    rcases hi with ⟨mask, hm, hi⟩
    rw [al_colmask_eval _ SN N (by simp)] at hm
    cases hm
    exact hb i (of_decide_eq_true hi)
  · intro _
    apply TraceProofs.all_safe
    simp [Stmt.TraceSafe, Op.SafeAt.eq_def]

theorem tail_safe (bounds : RegionBounds) (skn svk N BM BN BD SN : Nat)
    (s : BlockState) (V : RegionName) (base svn : Nat)
    (hoff : s.regs .nat [BN] "block_n_offs" = some (Tile.vec (fun j : Fin BN => SN+j.val)))
    (hptr : s.regs .ptr [BN,BD] "v_ptrs" = some (alPtrTile V base svk svn SN BN BD))
    (hb : ∀ i : TileIndex [BN,BD], SN+i.1.val < N →
      base+(SN+i.1.val)*svk+i.2.1.val*svn < bounds V) :
    Stmt.TraceSafeList bounds (sourceTail skn svk N BM BN BD Bool.false) s := by
  have he : Stmt.TraceSafeList bounds (sourceTail skn svk N BM BN BD Bool.false) s ↔
      Stmt.TraceSafeList bounds (alLoopTailG skn svk N BM BN BD Bool.false) s := by
    simp only [sourceTail,alLoopTailG,Bool.false_eq_true,↓reduceIte,List.drop_succ_cons,List.drop_zero]
    simp only [Stmt.TraceSafeList,Stmt.TraceSafe,stepStmt,FloatDType.toTileDType]
    erw [evalOp_maximumFloat_real]
    conv_lhs => arg 1; rw [Op.SafeAt.eq_def]
    conv_rhs => arg 1; rw [Op.SafeAt.eq_def]
    rfl
  exact he.mpr (RealSafetyFragments.tail_safe bounds skn svk N BM BN BD SN s V base svn Bool.false hoff hptr hb)

theorem iteration (bounds : RegionBounds) (c : Context) (n : Nat) (st : BlockState)
    (h : Invariant c n st)
    (hK : ∀ k, k < c.N → ∀ d : Fin c.D,
      alBase c.initial c.NH c.skz c.skh+k*c.skn+d.val*c.skk < bounds c.K)
    (hV : ∀ k, k < c.N → ∀ d : Fin c.D,
      alBase c.initial c.NH c.svz c.svh+k*c.svk+d.val*c.svn < bounds c.V) :
    Stmt.TraceSafeList bounds (sourceLoop c.scale c.skn c.svk c.N c.SP 64 64 c.D c.causal)
      (st.setReg "start_n" .nat [] (Tile.scalar (n*64))) := by
  let se := st.setReg "start_n" .nat [] (Tile.scalar (n*64))
  obtain ⟨sp, hpre, hpp, hpm, hpu, hpqk, hpbn, hpmp, hplp, hpacc,
      hpq, hpom, hpon, hpbi, hpoh, hpkp, hpvp⟩ :=
    prefix_steps_with_masked_keys c.scale c.N 64 64 c.D (n*64) se c.K
      (alBase c.initial c.NH c.skz c.skh) c.skn c.skk
      (⟨fun idx => some (c.query idx)⟩) (c.keyTile n)
      (by simp [se]) (by simpa [se] using h.columns) (by simpa [se] using h.query)
      (by simpa [se] using h.keyPtr)
      (by intro idx; simp only [Context.keyTile, Context.key, se, BlockState.setReg_readMem]
          unfold BlockState.readMem; rw [h.mem])
      (by simpa [se] using h.undef)
  unfold sourceLoop
  apply TraceProofs.append_safe
  · exact prefix_safe bounds c.scale c.N 64 64 c.D (n*64) se c.K
      (alBase c.initial c.NH c.skz c.skh) c.skn c.skk
      (by simp [se]) (by simpa [se] using h.columns) (by simpa [se] using h.keyPtr)
      (fun idx hk => hK _ hk idx.2.1)
  · intro t ht
    rw [hpre] at ht
    cases ht
    rw [TraceRewrite.constant_branch]
    have hpv : sp.regs .ptr [64,c.D] "v_ptrs" =
        some (alPtrTile c.V (alBase c.initial c.NH c.svz c.svh) c.svk c.svn (n*64) 64 c.D) := by
      rw [hpvp]; simpa [se] using h.valuePtr
    cases c.causal
    · simp only [Bool.false_eq_true,↓reduceIte,List.nil_append]
      exact tail_safe bounds c.skn c.svk c.N 64 64 c.D (n*64) sp c.V
        (alBase c.initial c.NH c.svz c.svh) c.svn hpbn hpv (fun idx hk => hV _ hk idx.2.1)
    · simp only [↓reduceIte,List.cons_append,List.nil_append,alCausalWhereStmt]
      apply TraceProofs.assign_cons (by simp [Op.SafeAt.eq_def])
      intro tile
      exact tail_safe bounds c.skn c.svk c.N 64 64 c.D (n*64) _ c.V
        (alBase c.initial c.NH c.svz c.svh) c.svn (by simpa using hpbn) (by simpa using hpv)
        (fun idx hk => hV _ hk idx.2.1)

theorem complete (bounds : RegionBounds) (c : Context) (Out : RegionName) (soz soh som son : Nat)
    (hD : 0 < c.D) (hN : 0 < c.N)
    (hSP : c.causal = Bool.true → c.SP ≤ c.initial.pids 0 * 64)
    (hundef : ∀ rg o, c.initial.undef rg o = 0)
    (hQ : ∀ idx : TileIndex [64,c.D], alActive c.initial c.H 64 idx →
      alBase c.initial c.NH c.sqz c.sqh+(c.initial.pids 0*64+idx.1.val)*c.sqm+idx.2.1.val*c.sqk < bounds c.Q)
    (hK : ∀ k, k < c.N → ∀ d : Fin c.D,
      alBase c.initial c.NH c.skz c.skh+k*c.skn+d.val*c.skk < bounds c.K)
    (hV : ∀ k, k < c.N → ∀ d : Fin c.D,
      alBase c.initial c.NH c.svz c.svh+k*c.svk+d.val*c.svn < bounds c.V)
    (hO : ∀ idx : TileIndex [64,c.D], alActive c.initial c.H 64 idx →
      alOutOffset c.initial c.NH soz soh som son 64 idx < bounds Out) :
    (TypedSource.repaired c.Q c.K c.V Out c.scale c.sqz c.sqh c.sqm c.sqk
      c.skz c.skh c.skn c.skk c.svz c.svh c.svk c.svn soz soh som son
      c.NH c.H c.N c.SP 64 64 c.D c.causal).eraseDType.toAlgKernel.TraceSafe bounds c.initial := by
  obtain ⟨sp,hpre,hp,hm,hu,hsm,hbi,hoh,hom,hon,hmp,hlp,ha,hq,hkp,hvp,hend⟩ :=
    alPreLoop_evalG c.initial c.Q c.K c.V c.sqz c.sqh c.sqm c.sqk
      c.skz c.skh c.skn c.skk c.svz c.svh c.svk c.svn c.NH c.H c.N 64 64 c.D hundef
  have hinv : Invariant c 0 sp := {
    m := fun _ => 0
    l := fun _ => 0
    a := fun _ => 0
    pids := hp
    mem := hm
    undef := hu
    maximum := hmp
    denominator := hlp
    accumulator := ha
    query := hq
    rows := hom
    columns := hon
    batch := hbi
    head := hoh
    keyPtr := hkp
    valuePtr := hvp
    numerical := by intro r d; exact TailMath.initial_consistent _ _
  }
  let bound := stop c.N c.causal c.SP (c.initial.pids 0)
  let sc := if c.causal then sp.setReg "block_n_end" .nat [] (Tile.scalar bound) else sp
  have hsc : Nonempty (Invariant c 0 sc) := by
    cases hc : c.causal
    · exact ⟨by simpa [sc,hc] using hinv⟩
    · exact ⟨by simpa [sc,hc] using RealKernel.Invariant.setEnd c 0 sp hinv bound⟩
  obtain ⟨hinvc⟩ := hsc
  have hb : sc.regs .nat [] "block_n_end" = some (Tile.scalar bound) := by
    cases hc : c.causal <;> simp [sc,hc,bound,stop,hend]
  obtain ⟨sl,hl,⟨hinvl⟩⟩ := RealKernel.loop c sc bound hinvc hb hD hN hSP
  have hloopSafe : Stmt.TraceSafe bounds (.forRangeDyn "start_n" (.constNat 0)
      (.ref .nat [] "block_n_end") (.constNat 64)
      (sourceLoop c.scale c.skn c.svk c.N c.SP 64 64 c.D c.causal)) sc := by
    simp only [Stmt.TraceSafe,Op.SafeAt.eq_def,evalOp_constNat,evalOp_ref,hb,Tile.scalar_data,true_and]
    apply TraceProofs.range_safe
      (P := fun i st => i%64=0 ∧ Nonempty (Invariant c (i/64) st)) _ _ 0 sc
      ⟨by simp,⟨by simpa using hinvc⟩⟩
    · intro i st hi hp
      obtain ⟨hm,⟨inv⟩⟩ := hp
      have he : i/64*64=i := by omega
      simpa only [he] using iteration bounds c (i/64) st inv hK hV
    · intro i st t hi hp hr
      obtain ⟨hm,⟨inv⟩⟩ := hp
      have he : i/64*64=i := by omega
      obtain ⟨t',hr',hinv'⟩ := LoopInvariant.iteration c (i/64) st inv hD hN hSP
      rw [he,hr,Option.some.injEq] at hr'
      subst t'
      exact ⟨by omega,by simpa only [show (i+64)/64=i/64+1 by omega] using hinv'⟩
  have hafter : Stmt.TraceSafeList bounds
      (.forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "block_n_end") (.constNat 64)
        (sourceLoop c.scale c.skn c.svk c.N c.SP 64 64 c.D c.causal) ::
        alPostLoopG Out soz soh som son c.H 64 c.D) sc := by
    apply (TraceProofs.cons_of_run_iff hl).mpr
    exact ⟨hloopSafe,RealSafetyPrologue.postamble_safe bounds Out c.initial soz soh som son c.NH c.H
      64 c.D sl hinvl.pids hinvl.batch hinvl.head hO⟩
  change Stmt.TraceSafeList bounds _ c.initial
  rw [repaired_body]
  apply TraceProofs.append_safe
  · exact RealSafetyPrologue.preamble_safe bounds c.initial c.Q c.K c.V
      c.sqz c.sqh c.sqm c.sqk c.skz c.skh c.skn c.skk c.svz c.svh c.svk c.svn c.NH c.H c.N 64 64 c.D hQ
  · intro t ht
    rw [hpre] at ht
    cases ht
    rw [TraceRewrite.constant_branch]
    cases hc : c.causal
    · simpa [sc,hc] using hafter
    · simp only [↓reduceIte,List.cons_append,List.nil_append]
      have he : evalOp (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil
            (Op.add .nat Broadcast.nil (.ref .nat [] "start_m") (.constNat 1)) (.constNat 64))
          (.constNat c.SP)) sp = some (Tile.scalar bound) := by
        rw [evalOp_add,evalOp_mul,evalOp_add]
        simp only [evalOp_ref,hsm,evalOp_constNat,Option.bind_eq_bind,Option.bind_some]
        apply congrArg some
        ext u
        simp [bound,MaskCoverage.stop,hc,Tile.bop_data,Tile.scalar_data,NumericDType.add,NumericDType.mul,
          Broadcast.leftIndex,Broadcast.rightIndex]
      unfold alBnEndCausalStmt
      rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some he)]
      exact ⟨by simp [Stmt.TraceSafe,Op.SafeAt.eq_def],by simpa [sc,hc] using hafter⟩

end VeriTile.Bench.Repairs.AttentionLlama.AccessSafety
