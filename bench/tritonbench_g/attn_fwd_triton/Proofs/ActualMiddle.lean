import bench.tritonbench_g.attn_fwd_triton.Proofs.ActualCall

/-! Compose the two complete source helper calls, preserving the caller
pointer tiles and retaining the second call's discarded maximum result. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.ActualMiddle
open VeriTile.Triton Memory
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealLoopFormula (Carry)
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

abbrev writes := ActualCall.writes

def code (p : Parameters) (BD : Nat) : List Stmt :=
  ActualCall.code p BD Bool.false ++ ActualCall.code p BD Bool.true

def result (mem : BlockState) (c : Config) (BD : Nat) (seed : Carry 128 BD) : Carry 128 BD :=
  ActualCall.result mem c BD Bool.true (ActualCall.result mem c BD Bool.false seed)

theorem caller_frame {s t : BlockState} {c : Config} {BD : Nat}
    (caller : FirstSetup.Caller s c 64 BD) (frame : Frame writes s t) :
    FirstSetup.Caller t c 64 BD :=
  ⟨(frame.regs _ _ "start_m" (by decide)).trans caller.row,
   (frame.regs _ _ "K_ptrs" (by decide)).trans caller.key,
   (frame.regs _ _ "K_scale_ptr" (by decide)).trans caller.scale,
   (frame.regs _ _ "V_ptrs" (by decide)).trans caller.value⟩

theorem executes (mem s : BlockState) (c : Config) (BD : Nat) (seed : Carry 128 BD)
    (fixed : Iteration.Fixed mem s c 128 64 BD)
    (carry : Iteration.Carries s seed) (caller : FirstSetup.Caller s c 64 BD) :
    ∃ t, stepStmts (code c.p BD) s=some t ∧ Iteration.Fixed mem t c 128 64 BD ∧
      Iteration.Carries t (result mem c BD seed) ∧ Frame writes s t := by
  obtain ⟨a,first,afixed,acarry,aframe⟩ := ActualCall.executes mem s c BD Bool.false seed fixed carry caller
  obtain ⟨t,second,tfixed,tcarry,tframe⟩ := ActualCall.executes mem a c BD Bool.true
    (ActualCall.result mem c BD Bool.false seed) afixed acarry (caller_frame caller aframe)
  exact ⟨t,(stepStmts.append_some first).trans second,tfixed,tcarry,aframe.trans tframe⟩

#axiomsClean executes
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.ActualMiddle
