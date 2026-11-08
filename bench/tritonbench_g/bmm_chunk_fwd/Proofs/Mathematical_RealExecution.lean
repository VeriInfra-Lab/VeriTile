import bench.tritonbench_g.bmm_chunk_fwd.Proofs.Mathematical_RealFinish

/-! Complete real execution of both variants, including the causal no-access
return. The active loop state is constructed by induction from initial memory;
no successful execution or precomputed contraction is an assumption. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealExecution
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false
open VeriTile.Bench.Repairs.BmmChunkFwd.ControlArithmetic
attribute [local simp 2000] VeriTile.Triton.eval_truncDiv VeriTile.Triton.eval_truncMod

def activeCode (swapped : Bool) (BM BN BK : Nat) (p : RealProgram.Parameters) : List Stmt :=
  RealProgram.realPart (RealProgram.parts swapped BM BN BK p).before ++
    [.forRangeInt "k" (.castNatToInt (.constNat 0)) (RealProgram.loopBound p.K BK) (.constNat 1)
      (RealProgram.realPart (RealProgram.parts swapped BM BN BK p).body)] ++
    RealProgram.realPart (RealProgram.parts swapped BM BN BK p).after

def finished (s t : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters) : BlockState :=
  RealFinish.result t BM BN p (s.pids 1) (RealSetup.chunk s p) (RealSetup.head s p)
    (RealSetup.tileM s BN p) (RealSetup.tileN s BN p) (RealSetup.seqBase s p)
    (RealLoop.carry s BM BN BK p (BmmChunkBwd.RealMath.blocks p.K BK))

theorem active_executes (swapped : Bool) (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters) :
    ∃ t, RealLoop.Holds s BM BN BK p (BmmChunkBwd.RealMath.blocks p.K BK) t ∧
      stepStmts (activeCode swapped BM BN BK p) (RealSetup.prologueResult s BM BN p) =
        some (finished s t BM BN BK p) := by
  obtain ⟨t, hp, h⟩ := RealLoop.prefix_executes swapped s BM BN BK p
  have hb : t.regs .int [] "pid_b" = some (Tile.scalar ((s.pids 1) : Int)) := by
    have hv := h.fixed .int [] "pid_b" (by decide)
    cases hs : p.hasSeq <;> cases hc : p.causal <;>
      simpa [RealSetup.result, RealSetup.prologueResult, hs, hc] using hv
  have hc : t.regs .int [] "pid_c" = some (Tile.scalar ((RealSetup.chunk s p) : Int)) := by
    have hv := h.fixed .int [] "pid_c" (by decide)
    cases hs : p.hasSeq <;> cases hc : p.causal <;>
      simpa [RealSetup.result, RealSetup.prologueResult, hs, hc] using hv
  have hh : t.regs .int [] "pid_h" = some (Tile.scalar ((RealSetup.head s p) : Int)) := by
    have hv := h.fixed .int [] "pid_h" (by decide)
    cases hs : p.hasSeq <;> cases hc : p.causal <;>
      simpa [RealSetup.result, RealSetup.prologueResult, hs, hc] using hv
  have hm : t.regs .int [] "pid_m" = some (Tile.scalar ((RealSetup.tileM s BN p) : Int)) := by
    have hv := h.fixed .int [] "pid_m" (by decide)
    cases hs : p.hasSeq <;> cases hc : p.causal <;>
      simpa [RealSetup.result, RealSetup.prologueResult, hs, hc] using hv
  have hn : t.regs .int [] "pid_n" = some (Tile.scalar ((RealSetup.tileN s BN p) : Int)) := by
    have hv := h.fixed .int [] "pid_n" (by decide)
    cases hs : p.hasSeq <;> cases hc : p.causal <;>
      simpa [RealSetup.result, RealSetup.prologueResult, hs, hc] using hv
  have hseq : p.hasSeq = Bool.true → t.regs .ptr [] "seq_idx_ptr" =
      some (Tile.scalar (Region.cast p.Seq, RealSetup.seqBase s p)) := by
    intro hs
    have hv := h.fixed .ptr [] "seq_idx_ptr" (by decide)
    simpa [RealSetup.result, hs] using hv
  refine ⟨t, h, ?_⟩
  rw [activeCode, stepStmts.append_some hp, RealPostlude.matches_source]
  exact RealFinish.executes t BM BN p (s.pids 1) (RealSetup.chunk s p) (RealSetup.head s p)
    (RealSetup.tileM s BN p) (RealSetup.tileN s BN p) (RealSetup.seqBase s p)
    (RealLoop.carry s BM BN BK p (BmmChunkBwd.RealMath.blocks p.K BK)) hb hc hh hm hn hseq h.acc

theorem prologue_active (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters) :
    evalOp (RealProgram.condition BM BN BK p) (RealSetup.prologueResult s BM BN p) =
      some (Tile.scalar (RealSetup.active s BM BN p)) := by
  cases hc : p.causal
  all_goals
    simp (config := {maxSteps := 300000}) [RealProgram.condition,RealProgram.branch,
      RealProgram.source,TypedSource.specialize,
      VeriTile.Bench.Repairs.BmmChunkFwd.Kernels.mainSource,
      VeriTile.Bench.Repairs.BmmChunkFwd.originalKernel,
      Source.conditional?,Source.signedRange?,ComputeKernel.surfaceBody,
      Op.eraseDType.eq_def,VeriTile.Triton.eraseDType,
      NumericDType.eraseDType,ComparableDType.eraseDType,
      RealSetup.prologueResult,RealSetup.active,hc,evalOp.eq_def,bind,Option.bind]
  all_goals
    try simp only [NumericDType.mul,NumericDType.add,←Int.natCast_one,
      ←Int.natCast_add,←Int.natCast_mul,Int.ofNat_le]
    try split_ifs with h
    all_goals simp [Tile.uop,Tile.scalar,-Tile.scalar_eta]
    all_goals simp only [←Int.natCast_mul,←Int.natCast_add,Int.ofNat_le] at h
    all_goals funext i
    all_goals symm
    all_goals first
      | exact decide_eq_false_iff_not.mpr (by omega)
      | exact decide_eq_true_eq.mpr (by omega)

theorem kernel_split (swapped : Bool) (BM BN BK : Nat) (p : RealProgram.Parameters) :
    (RealProgram.kernel swapped BM BN BK p).toAlgKernel.body =
      RealSetup.prologue BM BN BK p ++
        [.ifThen (RealProgram.condition BM BN BK p) (activeCode swapped BM BN BK p)] := by
  rw [RealProgram.body_split]
  cases swapped <;> rfl

theorem skipped_executes (swapped : Bool) (s : BlockState) (BM BN BK : Nat)
    (p : RealProgram.Parameters) (hskip : RealSetup.active s BM BN p = Bool.false) :
    stepStmts (RealProgram.kernel swapped BM BN BK p).toAlgKernel.body s =
      some (RealSetup.prologueResult s BM BN p) := by
  rw [kernel_split, stepStmts.append_some (RealSetup.prologue_executes s BM BN BK p)]
  simp only [stepStmts, stepStmt, prologue_active, hskip, Tile.scalar,
    bind, Option.bind_some, Bool.false_eq_true, ↓reduceIte]

theorem kernel_executes (swapped : Bool) (s : BlockState) (BM BN BK : Nat)
    (p : RealProgram.Parameters) (hactive : RealSetup.active s BM BN p = Bool.true) :
    ∃ t, RealLoop.Holds s BM BN BK p (BmmChunkBwd.RealMath.blocks p.K BK) t ∧
      stepStmts (RealProgram.kernel swapped BM BN BK p).toAlgKernel.body s =
        some (finished s t BM BN BK p) := by
  obtain ⟨t, ht, he⟩ := active_executes swapped s BM BN BK p
  refine ⟨t, ht, ?_⟩
  rw [kernel_split, stepStmts.append_some (RealSetup.prologue_executes s BM BN BK p)]
  simp only [stepStmts, stepStmt, prologue_active, hactive, Tile.scalar,
    bind, Option.bind_some, ↓reduceIte, he]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealExecution
