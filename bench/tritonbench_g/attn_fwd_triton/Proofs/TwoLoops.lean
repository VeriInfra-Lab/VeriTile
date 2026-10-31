import bench.tritonbench_g.attn_fwd_triton.Proofs.SetupSource

/-! Compose both complete helper calls in their source order, including
pointer reset for the second and the stage-2 repeated diagonal branch. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.TwoLoops
open VeriTile.Triton Memory
open VeriTile.Bench.TritonBenchG.AttnFwdCausal
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealScores (Frame)
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealLoopFormula (Carry)
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def writes : List RegName := Setup.writes Bool.false ++ Iteration.writes Bool.false ++
  Setup.writes Bool.true ++ Iteration.writes Bool.true

theorem fixed_setup {mem s t : BlockState} {c : Config} {BM BN BD : Nat} {second : Bool}
    (f : Iteration.Fixed mem s c BM BN BD) (hf : Frame (Setup.writes second) s t) :
    Iteration.Fixed mem t c BM BN BD := by
  refine ⟨hf.mem.trans f.memory,?_,?_,?_,?_,?_,?_,?_⟩
  · exact (hf.regs _ _ "q" (by cases second <;> decide)).trans f.query
  · exact (hf.regs _ _ "q_scale" (by cases second <;> decide)).trans f.queryScale
  · exact (hf.regs _ _ "offs_m" (by cases second <;> decide)).trans f.rows
  · exact (hf.regs _ _ "offs_n" (by cases second <;> decide)).trans f.keys
  · exact (hf.regs _ _ "n_ctx" (by cases second <;> decide)).trans f.context
  · exact (hf.regs _ _ "stride_kn64" (by cases second <;> decide)).trans f.keyStride
  · exact (hf.regs _ _ "stride_vk64" (by cases second <;> decide)).trans f.valueStride

theorem carry_setup {s t : BlockState} {BM BD : Nat} {carry : Carry BM BD} {second : Bool}
    (hc : Iteration.Carries s carry) (hf : Frame (Setup.writes second) s t) : Iteration.Carries t carry :=
  ⟨(hf.regs _ _ "m_i" (by cases second <;> decide)).trans hc.center,
   (hf.regs _ _ "l_i" (by cases second <;> decide)).trans hc.denominator,
   (hf.regs _ _ "acc" (by cases second <;> decide)).trans hc.numerator⟩

theorem caller_frame {s t : BlockState} {c : Config} {BN BD : Nat}
    (hc : FirstSetup.Caller s c BN BD) (hf : Frame writes s t) : FirstSetup.Caller t c BN BD :=
  ⟨(hf.regs _ _ "start_m" (by decide)).trans hc.row,
   (hf.regs _ _ "K_ptrs" (by decide)).trans hc.key,
   (hf.regs _ _ "K_scale_ptr" (by decide)).trans hc.scale,
   (hf.regs _ _ "V_ptrs" (by decide)).trans hc.value⟩

def firstResult (mem : BlockState) (c : Config) (BM BN BD : Nat) (diagonal : Bool) (seed : Carry BM BD) : Carry BM BD :=
  Loop.after mem c BM BN BD (FirstSetup.firstLower c.startM BM diagonal) Bool.false diagonal seed
    (Loop.count (FirstSetup.firstLower c.startM BM diagonal) (FirstSetup.firstUpper c.startM BM c.p.context diagonal) BN)

def bothResult (mem : BlockState) (c : Config) (BM BN BD : Nat) (diagonal : Bool) (seed : Carry BM BD) : Carry BM BD :=
  Loop.after mem c BM BN BD (c.startM*BM) Bool.true diagonal (firstResult mem c BM BN BD diagonal seed)
    (Loop.count (c.startM*BM) (min ((c.startM+1)*BM) c.p.context) BN)

def code (BM BN BD D : Nat) (diagonal : Bool) : List Stmt :=
  FirstSetup.code BM BN BD diagonal ++ [loop BM BN BD D Bool.false diagonal] ++
    Setup.code BM BN BD Bool.true ++ [loop BM BN BD D Bool.true diagonal]

theorem statement (BM BN BD D lower upper : Nat) (second diagonal : Bool) (s t : BlockState)
    (hl : if second || diagonal then
      s.regs .int [] (Setup.lowerName second) = some (Tile.scalar (lower : Int))
      else s.regs .nat [] (Setup.lowerName second) = some (Tile.scalar lower))
    (hh : s.regs .int [] (Setup.upperName second) = some (Tile.scalar (upper : Int)))
    (hloop : stepForRangeIntAux "start_n" (lower : Int) (upper : Int) BN
      (sourceIteration BM BN BD D second diagonal) s = some t) :
    stepStmt (loop BM BN BD D second diagonal) s = some t := by
  cases second <;> cases diagonal
  all_goals simp only [Setup.lowerName,Setup.upperName,RealSetup.lowerName,RealSetup.upperName,
    Bool.false_or,Bool.true_or,Bool.false_eq_true,Bool.true_eq,↓reduceIte] at hl hh
  all_goals simpa only [loop,Setup.lowerName,Setup.upperName,RealSetup.lowerName,RealSetup.upperName,
    Bool.false_or,Bool.true_or,Bool.false_eq_true,Bool.true_eq,↓reduceIte,
    stepStmt,evalOp.eq_def,hl,hh,bind,Option.bind,Tile.scalar] using hloop

theorem executes (mem s : BlockState) (c : Config) (BM BN BD : Nat) (diagonal : Bool) (seed : Carry BM BD)
    (hN : 0 < BN) (f : Iteration.Fixed mem s c BM BN BD) (hc : Iteration.Carries s seed)
    (caller : FirstSetup.Caller s c BN BD) :
    ∃ t,stepStmts (code BM BN BD c.p.headDim diagonal) s = some t ∧
      Iteration.Fixed mem t c BM BN BD ∧ Iteration.Carries t (bothResult mem c BM BN BD diagonal seed) ∧ Frame writes s t := by
  obtain ⟨sa,hastep,halo,hahi,haptr,haf⟩ := FirstSetup.executes s c BM BN BD diagonal caller f.context f.keyStride f.valueStride
  have fa := fixed_setup f haf
  obtain ⟨sb,hbloop,hbr,hbf⟩ := Loop.executes mem sa c BM BN BD
    (FirstSetup.firstLower c.startM BM diagonal) (FirstSetup.firstUpper c.startM BM c.p.context diagonal)
    Bool.false diagonal seed hN ⟨fa,carry_setup hc haf,haptr⟩
  have hbstep := statement BM BN BD c.p.headDim _ _ Bool.false diagonal sa sb (by simpa using halo) hahi hbloop
  have hbframe : Frame writes s sb := (haf.mono (by intro name hn; simp only [writes,List.mem_append]; tauto)).trans (hbf.mono (by intro name hn; simp only [writes,List.mem_append]; tauto))
  have cb := caller_frame caller hbframe
  obtain ⟨sc,hcstep,hclo,hchi,hcptr,hcf⟩ := Setup.executes sb c BM BN BD Bool.true
    cb.row cb.key cb.scale cb.value hbr.fixed.context hbr.fixed.keyStride hbr.fixed.valueStride
  have fc := fixed_setup hbr.fixed hcf
  obtain ⟨t,htloop,htr,htf⟩ := Loop.executes mem sc c BM BN BD (c.startM*BM) (min ((c.startM+1)*BM) c.p.context)
    Bool.true diagonal (firstResult mem c BM BN BD diagonal seed) hN ⟨fc,carry_setup hbr.carry hcf,hcptr⟩
  have htstep := statement BM BN BD c.p.headDim _ _ Bool.true diagonal sc t (by simpa using hclo) hchi htloop
  refine ⟨t,?_,htr.fixed,htr.carry,
    ((hbframe.trans (hcf.mono (by intro name hn; simp only [writes,List.mem_append]; tauto))).trans (htf.mono (by intro name hn; simp only [writes,List.mem_append]; tauto)))⟩
  have hb : stepStmts [loop BM BN BD c.p.headDim Bool.false diagonal] sa = some sb := by
    rw [stepStmts.cons_some hbstep,stepStmts.nil]
  have ht : stepStmts [loop BM BN BD c.p.headDim Bool.true diagonal] sc = some t := by
    rw [stepStmts.cons_some htstep,stepStmts.nil]
  exact (stepStmts.append_some ((stepStmts.append_some ((stepStmts.append_some hastep).trans hb)).trans hcstep)).trans ht

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.TwoLoops
