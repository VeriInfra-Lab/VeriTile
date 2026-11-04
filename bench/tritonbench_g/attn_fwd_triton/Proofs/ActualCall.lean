import bench.tritonbench_g.attn_fwd_triton.Proofs.ActualRanges
import bench.tritonbench_g.attn_fwd_triton.Proofs.FirstSetup

/-! Execute one complete pasted helper invocation, retaining caller frames. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.ActualCall
open VeriTile.Triton Memory
open RegisterViews HelperRegisters
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealLoopFormula (Carry)
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

def writes : List RegName := ActualFrames.workspace ++
  ["acc","l_i","m_i","__veritile_python_underscore"]

def code (p : Parameters) (BD : Nat) (second : Bool) : List Stmt :=
  ActualSetup.code p BD second ++
    ActualSource.project (TypedExecution.loopCode (ActualSource.config p) BD second)

def result (mem : BlockState) (c : Config) (BD : Nat) (second : Bool)
    (seed : Carry 128 BD) : Carry 128 BD :=
  ActualReturn.returned second seed
    (Loop.after mem c 128 64 BD (ActualSetup.lower c second) second
      (decide (c.p.stage=2)) seed
      (Loop.count (ActualSetup.lower c second) (ActualSetup.upper c second) 64))

theorem fixed {mem s t : BlockState} {c : Config} {BD : Nat}
    (before : Iteration.Fixed mem s c 128 64 BD) (frame : Frame writes s t) :
    Iteration.Fixed mem t c 128 64 BD := by
  refine ⟨frame.mem.trans before.memory,?_,?_,?_,?_,?_,?_,?_⟩
  all_goals first
    | exact (frame.regs _ _ _ (by decide)).trans before.query
    | exact (frame.regs _ _ _ (by decide)).trans before.queryScale
    | exact (frame.regs _ _ _ (by decide)).trans before.rows
    | exact (frame.regs _ _ _ (by decide)).trans before.keys
    | exact (frame.regs _ _ _ (by decide)).trans before.context
    | exact (frame.regs _ _ _ (by decide)).trans before.keyStride
    | exact (frame.regs _ _ _ (by decide)).trans before.valueStride

theorem executes (mem s : BlockState) (c : Config) (BD : Nat) (second : Bool)
    (seed : Carry 128 BD)
    (fixedBefore : Iteration.Fixed mem s c 128 64 BD)
    (carry : Iteration.Carries s seed) (caller : FirstSetup.Caller s c 64 BD) :
    ∃ t, stepStmts (code c.p BD second) s=some t ∧
      Iteration.Fixed mem t c 128 64 BD ∧ Iteration.Carries t (result mem c BD second seed) ∧
      Frame writes s t := by
  obtain ⟨a,setupRan,ready,lo,hi⟩ := ActualSetup.executes mem s c BD second seed fixedBefore carry caller
  have setupFrame := ActualCallFrames.setup_frame setupRan
  obtain ⟨b,loopRan,registers,loopFrame⟩ := HelperLoops.executes mem a c 128 64 BD
    (ActualSetup.lower c second) (ActualSetup.upper c second) second seed (by decide) ready
  have rangeRan := ActualRanges.executes lo hi loopRan
  have privateFrame := setupFrame.trans ((ActualFrames.mapped loopFrame).mono (ActualFrames.loop_writes second))
  obtain ⟨t,returnRan,returned⟩ := ActualReturn.executes b c.p BD second seed _ registers.carry
    ((privateFrame.regs _ _ "m_i" (by decide)).trans carry.center)
  have returnFrame := ActualCallFrames.return_frame returnRan
  have frame : Frame writes s t :=
    (privateFrame.mono (by intro n h; exact List.mem_append_left _ h)).trans
      (returnFrame.mono (by cases second <;> simp only [ActualReturn.writes,writes,List.subset_def,List.mem_append,List.mem_cons,List.mem_singleton,Bool.false_eq_true,Bool.true_eq,↓reduceIte] <;> tauto))
  refine ⟨t,?_,fixed fixedBefore frame,returned,frame⟩
  rw [code,ActualRanges.source]
  exact (stepStmts.append_some setupRan).trans
    ((stepStmts.cons_some rangeRan).trans returnRan)

#axiomsClean executes
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.ActualCall
