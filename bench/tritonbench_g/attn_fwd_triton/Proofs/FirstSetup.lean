import bench.tritonbench_g.attn_fwd_triton.Proofs.Setup

/-! Execute the first helper's actual local copies and stage branch,
including the repaired context tail bound and independent pointer advances. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.FirstSetup
open VeriTile.Triton Memory Preparation
open VeriTile.Bench.TritonBenchG.AttnFwdCausal
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxHeartbeats 800000

abbrev copies := RealFirstSetup.copies
abbrev firstLower := RealLoopFormula.firstLower

def firstUpper (startM BM N : Nat) (diagonal : Bool) : Nat :=
  if diagonal then min ((startM+1)*BM) N else startM*BM

structure Caller (s : BlockState) (c : Config) (BN BD : Nat) : Prop where
  row : s.regs .nat [] "start_m" = some (Tile.scalar c.startM)
  key : s.regs .ptr [BD,BN] "K_ptrs" = some (keyPointer c BD BN 0)
  scale : s.regs .ptr [] "K_scale_ptr" = some (keyScalePointer c BN 0)
  value : s.regs .ptr [BN,BD] "V_ptrs" = some (valuePointer c BN BD 0)

def code (BM BN BD : Nat) (diagonal : Bool) : List Stmt :=
  copies BM BN BD ++ [.ifThen (.constBool diagonal) (Setup.code BM BN BD Bool.false)]

theorem copies_execute (s : BlockState) (c : Config) (BM BN BD : Nat)
    (caller : Caller s c BN BD) :
    ∃ t, stepStmts (copies BM BN BD) s = some t ∧
      t.regs .nat [] "first_lo" = some (Tile.scalar 0) ∧
      t.regs .nat [] "first_hi" = some (Tile.scalar (c.startM*BM)) ∧
      Pointers t c BN BD 0 Bool.false ∧ Frame (Setup.writes Bool.false) s t := by
  let sl := s.setReg "first_lo" .nat [] (Tile.scalar 0)
  let sh := sl.setReg "first_hi" .nat [] (Tile.scalar (c.startM*BM))
  let sk := sh.setReg "K_ptrs_first" .ptr [BD,BN] (keyPointer c BD BN 0)
  let ss := sk.setReg "K_scale_ptr_first" .ptr [] (keyScalePointer c BN 0)
  have hhigh : evalOp (RealSetup.lowerExpr BM) sl = some (Tile.scalar (c.startM*BM)) := by
    simp [RealSetup.lowerExpr, evalOp.eq_def, sl, caller.row]; rfl
  have hk : evalOp (.ref .ptr [BD,BN] "K_ptrs") sh = some (keyPointer c BD BN 0) := by
    simp [sh, sl, caller.key]
  have hs : evalOp (.ref .ptr [] "K_scale_ptr") sk = some (keyScalePointer c BN 0) := by
    simp [sk, sh, sl, caller.scale]
  have hv : evalOp (.ref .ptr [BN,BD] "V_ptrs") ss = some (valuePointer c BN BD 0) := by
    simp [ss, sk, sh, sl, caller.value]
  simp only [copies,RealFirstSetup.copies]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_constNat 0 s)),
    stepStmts.cons_some (stepStmt_assign_eq_some hhigh),
    stepStmts.cons_some (stepStmt_assign_eq_some hk),
    stepStmts.cons_some (stepStmt_assign_eq_some hs),
    stepStmts.cons_some (stepStmt_assign_eq_some hv), stepStmts.nil]
  refine ⟨_, rfl, by simp [ss, sk, sh, sl], by simp [ss, sk, sh],
    ⟨by simp [ss, sk, keyName,RealMemory.keyName], by simp [ss, scaleName,RealMemory.scaleName], by simp [valueName,RealMemory.valueName]⟩,
    rfl, rfl, rfl, ?_⟩
  intro dt shape name hn
  simp only [Setup.writes,Setup.lowerName,Setup.upperName,RealSetup.lowerName, RealSetup.upperName, keyName,RealMemory.keyName,
    scaleName,RealMemory.scaleName, valueName,RealMemory.valueName, Bool.false_eq_true, ↓reduceIte, List.mem_cons, not_or] at hn
  simp [ss, sk, sh, sl, hn.1, hn.2.1, hn.2.2.1, hn.2.2.2.1, hn.2.2.2.2.1]

theorem executes (s : BlockState) (c : Config) (BM BN BD : Nat) (diagonal : Bool)
    (caller : Caller s c BN BD)
    (hctx : s.regs .nat [] "n_ctx" = some (Tile.scalar c.p.context))
    (hkr : s.regs .nat [] "stride_kn64" = some (Tile.scalar c.p.keyRow))
    (hvr : s.regs .nat [] "stride_vk64" = some (Tile.scalar c.p.valueRow)) :
    ∃ t, stepStmts (code BM BN BD diagonal) s = some t ∧
      t.regs .nat [] "first_lo" = some (Tile.scalar (firstLower c.startM BM diagonal)) ∧
      t.regs .nat [] "first_hi" = some (Tile.scalar (firstUpper c.startM BM c.p.context diagonal)) ∧
      Pointers t c BN BD (firstLower c.startM BM diagonal) Bool.false ∧
      Frame (Setup.writes Bool.false) s t := by
  obtain ⟨sc, hcopy, hlo, hhi, hp, hf⟩ := copies_execute s c BM BN BD caller
  cases diagonal with
  | «false» =>
    refine ⟨sc, ?_, hlo, hhi, hp, hf⟩
    rw [code, stepStmts.append_some hcopy]
    have hskip : stepStmt (.ifThen (.constBool Bool.false) (Setup.code BM BN BD Bool.false)) sc = some sc := by
      conv_lhs => unfold stepStmt
      simp only [evalOp.eq_def, bind, Option.bind, Tile.scalar, Bool.false_eq_true, ↓reduceIte]
    rw [stepStmts.cons_some hskip, stepStmts.nil]
  | «true» =>
    obtain ⟨t, ht, hl, hh, hptr, hframe⟩ := Setup.executes sc c BM BN BD Bool.false
      ((hf.regs _ _ "start_m" (by decide)).trans caller.row) hp.key hp.scale hp.value
      ((hf.regs _ _ "n_ctx" (by decide)).trans hctx)
      ((hf.regs _ _ "stride_kn64" (by decide)).trans hkr)
      ((hf.regs _ _ "stride_vk64" (by decide)).trans hvr)
    refine ⟨t, ?_, hl, hh, hptr, hf.trans hframe⟩
    rw [code, stepStmts.append_some hcopy]
    have hbranch : stepStmt (.ifThen (.constBool Bool.true) (Setup.code BM BN BD Bool.false)) sc = some t := by
      conv_lhs => unfold stepStmt
      simpa only [evalOp.eq_def, bind, Option.bind, Tile.scalar, Bool.true_eq, ↓reduceIte] using ht
    rw [stepStmts.cons_some hbranch, stepStmts.nil]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.FirstSetup
