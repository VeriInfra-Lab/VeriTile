import bench.tritonbench_g.fused_recurrent_hgrn.Proofs.ForwardLoop

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn
open VeriTile.Triton FP.Structural
namespace Execution
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 12000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def reverseInitial (p : Ops α) (s : State α) (V : Bool) (D BD : Nat) (j : TileIndex [BD]) : α :=
  if V then masked p s "dht" (slot s D BD j) (lane s BD j < D) else p.zero

def reversePrevious (p : Ops α) (s : State α) (U : Bool) (T D BD t : Nat)
    (j : TileIndex [BD]) : α :=
  if 0 < t then masked p s "o" (row s (t-1) T D BD j) (lane s BD j < D)
  else if U then masked p s "h0" (slot s D BD j) (lane s BD j < D) else p.zero

def reverseDx (p : Ops α) (s : State α) (T D BD t : Nat)
    (carry : TileIndex [BD] → α) (j : TileIndex [BD]) : α :=
  p.add (carry j) (masked p s "do" (row s t T D BD j) (lane s BD j < D))

def reverseDg (p : Ops α) (s : State α) (U : Bool) (T D BD t : Nat)
    (carry : TileIndex [BD] → α) (j : TileIndex [BD]) : α :=
  p.mul (reverseDx p s T D BD t carry j) (reversePrevious p s U T D BD t j)

def reverseNext (p : Ops α) (s : State α) (T D BD t : Nat)
    (carry : TileIndex [BD] → α) (j : TileIndex [BD]) : α :=
  p.mul (reverseDx p s T D BD t carry j) (masked p s "g" (row s t T D BD j) (lane s BD j < D))

theorem backward_prologue (p : Ops α) (s : State α) (T D BD : Nat) (V : Bool) :
    ∃ q, run (algebra p) (backwardPrologue T D BD V) s=some q ∧
      q.mem=s.mem ∧ q.pids=s.pids ∧
      q.regs .bool [BD] "mask"=some (fun j => decide (lane s BD j < D)) ∧
      q.regs .nat [] "i_bh"=some (fun _ => s.pids 1) ∧
      q.regs .nat [BD] "o_d"=some (lane s BD) ∧
      q.regs .real [BD] "b_dh"=some (reverseInitial p s V D BD) := by
  cases V <;>
    simp [backwardPrologue,backward,ComputeKernel.surfaceBody,run,step,evalExpr,
      evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,
      natLt,numericLt,Region.cast,State.setReg,algebra,reverseInitial,masked,read,lane,slot,
      VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.channel,
      VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.stateOffset,
      Broadcast.leftIndex,Broadcast.rightIndex,ofFloat,toFloat]
  all_goals repeat' constructor
  all_goals funext j
  all_goals rcases j with ⟨j,⟨⟩⟩
  all_goals first | rfl | simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]

theorem backward_step (p : Ops α) (s : State α) (T D BD n : Nat) (U : Bool)
    (carry : TileIndex [BD] → α)
    (hn : s.regs .nat [] "step"=some (fun _ => n))
    (hm : s.regs .bool [BD] "mask"=some (fun j => decide (lane s BD j < D)))
    (hb : s.regs .nat [] "i_bh"=some (fun _ => s.pids 1))
    (hd : s.regs .nat [BD] "o_d"=some (lane s BD))
    (hc : s.regs .real [BD] "b_dh"=some carry) :
    ∃ q, run (algebra p) (backwardBody T D BD U) s=some q ∧
      q.mem=(scatter (scatter s "dx" BD (row s (T-1-n) T D BD)
        (fun j => lane s BD j < D) (reverseDx p s T D BD (T-1-n) carry))
        "dg" BD (row s (T-1-n) T D BD) (fun j => lane s BD j < D)
        (reverseDg p s U T D BD (T-1-n) carry)).mem ∧
      q.pids=s.pids ∧
      q.regs .bool [BD] "mask"=s.regs .bool [BD] "mask" ∧
      q.regs .nat [] "i_bh"=s.regs .nat [] "i_bh" ∧
      q.regs .nat [BD] "o_d"=s.regs .nat [BD] "o_d" ∧
      q.regs .real [BD] "b_dh"=some (reverseNext p s T D BD (T-1-n) carry) := by
  have hrow : 0 < T-1-n → s.pids 1*T+(T-1-n)-1=s.pids 1*T+(T-1-n-1) := by omega
  cases U <;> by_cases hi : 0 < T-1-n <;>
    simp [backwardBody,backward,ComputeKernel.surfaceBody,run,step,evalExpr,
      evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,
      natLt,numericLt,Region.cast,algebra,store,State.setReg,hn,hm,hb,hd,hc,hi,
      Broadcast.leftIndex,Broadcast.rightIndex,ofFloat,toFloat,masked_regs,
      State.masked_scatter_pids]
  all_goals simp [reverseNext,reverseDx,reverseDg,reversePrevious,masked,read,scatter,row,lane,slot,hi,
    VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.rowOffset,
    VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.channel,
    VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.stateOffset,
    VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.scatter]
  all_goals constructor
  all_goals first
    | (try rw [hrow hi])
      apply VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.scatter_memory_congr
      apply VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.scatter_memory_congr
      rfl
    | funext j
      rcases j with ⟨j,⟨⟩⟩
      simp [reverseNext,reverseDx,masked,read,row,lane,FP.Structural.bop,
        Broadcast.leftIndex,Broadcast.rightIndex,
        VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.channel,
        VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.rowOffset]

#axiomsClean backward_step
end
end Execution
end VeriTile.Bench.Repairs.FusedRecurrentHgrn
