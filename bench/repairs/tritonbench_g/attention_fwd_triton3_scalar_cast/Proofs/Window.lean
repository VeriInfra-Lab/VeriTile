import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.Prelude
import VeriTile.Triton.Float.Integer

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Reference
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
open VeriTile.Bench.Repairs.ContextAttnMistral
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem lower_clamp (x : Int) : max x 0=if x<0 then 0 else x := by split_ifs <;> omega
theorem upper_clamp (x cap : Int) : min x cap=if cap<x then cap else x := by split_ifs <;> omega
attribute [local simp 1100] FP.Structural.eval_truncDiv

theorem window (c : Call) (o : Ops α) (initialState s : State α)
    (live : Started c o (inputs initialState) s) (memory : s.mem=initialState.mem) :
    ∃ t,run (engine o.algebra) ((parts c).before.drop 20) s=some t ∧
      Matches c o (inputs initialState) (initial c o (inputs initialState)) t ∧
      t.mem=initialState.mem := by
  rcases live with ⟨⟨h0,h1,h2,h3,h4,h5,h6⟩,h7,h8,h9,h10,h11⟩
  by_cases hw : c.window=0 <;> by_cases hc : c.complement=0
  all_goals
    by_cases hl : (((initialState.pids 0 : Int)*c.BM+c.offset-c.size+1).tdiv c.BN*c.BN)<0
    all_goals
      by_cases hh : (c.NK : Int)<((((initialState.pids 0 : Int)+1)*c.BM-1+c.offset+c.BN).tdiv c.BN*c.BN)
      all_goals
        simp (config := {maxSteps := 500000}) [parts,Kernels.source,Quoted.kernel,
          ComputeKernel.surfaceBody,Source.intRange?,run,step,evalExpr,evalComputeOp,evalOp_unfold,
          evalDeltas,Option.bind,ComputeDType.eraseDType,State.setReg,engine,Algebra.withDefaultPrecision,
          FPBackend.Ops.algebra,resolvePrecision,numeric,numericLt,numericLe,numericEq,
          natLt,natLe,integralFloorDiv,integralMod,ofFloat,toFloat,Region.cast,boundaryZero,
          h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,hw,hc,hl,hh,
          Matches,initial,lower,upper,lower_clamp,upper_clamp,inputs,
          Int.natCast_mul,Int.natCast_add,bop,Broadcast.leftIndex,Broadcast.rightIndex,memory]
  all_goals simp [funext_iff,bop,Broadcast.leftIndex,Broadcast.rightIndex]

theorem prelude (c : Call) (o : Ops α) (s : State α) :
    ∃ t,run (engine o.algebra) (parts c).before s=some t ∧
      Matches c o (inputs s) (initial c o (inputs s)) t ∧ t.mem=s.mem := by
  obtain ⟨headerState,hheader,headerLive,headerMemory⟩ := header c o s
  obtain ⟨startedState,hstarted,startedLive,startedMemory⟩ :=
    initialization c o s headerState headerLive headerMemory
  obtain ⟨ready,hready,readyLive,readyMemory⟩ :=
    window c o s startedState startedLive startedMemory
  refine ⟨ready,?_,readyLive,readyMemory⟩
  have split : (parts c).before=(parts c).before.take 16 ++
      ((parts c).before.drop 16).take 4 ++ (parts c).before.drop 20 := rfl
  rw [split,run_append,run_append,hheader]
  simp only [Option.bind_some,hstarted]
  exact hready

#axiomsClean prelude
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Reference
