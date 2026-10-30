import bench.tritonbench_g.fused_recurrent_hgrn.Proofs.Base

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn
open VeriTile.Triton FP.Structural
namespace Execution
noncomputable section
set_option maxRecDepth 12000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false
set_option linter.unusedSectionVars false

abbrev Old := VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.forward
abbrev lane := @VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.channel
abbrev row := @VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.rowOffset
abbrev slot := @VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.stateOffset
abbrev scatter := @VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.scatter

variable {α : Type} [Inhabited α]

@[simp] theorem tile_eta {BD : Nat} (i : TileIndex [BD]) : (i.1,PUnit.unit)=i := by
  rcases i with ⟨i,⟨⟩⟩
  rfl

def read (s : State α) (r : RegionName) (a : Nat) : α := (s.mem r a).read .real

def masked (p : Ops α) (s : State α) (r : RegionName) (a : Nat) (active : Prop)
    [Decidable active] : α := if active then read s r a else p.zero

def initial (p : Ops α) (s : State α) (U : Bool) (D BD : Nat) (j : TileIndex [BD]) : α :=
  if U then p.add p.zero (masked p s "h0" (slot s D BD j) (lane s BD j<D)) else p.zero

def next (p : Ops α) (s : State α) (T D BD t : Nat) (carry : TileIndex [BD] → α)
    (j : TileIndex [BD]) : α :=
  p.add (p.mul (masked p s "g" (row s t T D BD j) (lane s BD j<D)) (carry j))
    (masked p s "x" (row s t T D BD j) (lane s BD j<D))

@[simp] theorem scatter_regs (s : State α) (r : RegionName) (BD : Nat)
    (off : TileIndex [BD] → Nat) (active : TileIndex [BD] → Prop) [DecidablePred active]
    (v : TileIndex [BD] → α) : (scatter s r BD off active v).regs=s.regs := by
  unfold scatter VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.scatter
  rw [State.masked_scatter_eq_filter,State.scatter_regs]

@[simp] theorem scatter_pids (s : State α) (r : RegionName) (BD : Nat)
    (off : TileIndex [BD] → Nat) (active : TileIndex [BD] → Prop) [DecidablePred active]
    (v : TileIndex [BD] → α) : (scatter s r BD off active v).pids=s.pids := by
  exact State.masked_scatter_pids r off (fun j => .mk .real (v j)) active _ s

theorem prologue (p : Ops α) (s : State α) (T D BD : Nat) (U : Bool) :
    ∃ q, run (algebra p) (forwardPrologue T D BD U) s=some q ∧
      q.mem=s.mem ∧ q.pids=s.pids ∧
      q.regs .bool [BD] "mask"=some (fun j => decide (lane s BD j<D)) ∧
      q.regs .nat [] "i_bh"=some (fun _ => s.pids 1) ∧
      q.regs .nat [BD] "o_d"=some (lane s BD) ∧
      q.regs .ptr [BD] "p_x"=some (fun j => (("x" : RegionName),row s 0 T D BD j)) ∧
      q.regs .ptr [BD] "p_g"=some (fun j => (("g" : RegionName),row s 0 T D BD j)) ∧
      q.regs .ptr [BD] "p_o"=some (fun j => (("o" : RegionName),row s 0 T D BD j)) ∧
      q.regs .real [BD] "b_h"=some (initial p s U D BD) := by
  cases U <;>
    simp [forwardPrologue,forward,ComputeKernel.surfaceBody,run,step,evalExpr,
      evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,
      natLt,numericLt,Region.cast,State.setReg,algebra,initial,masked,read,lane,row,slot,
      VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.channel,
      VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.rowOffset,
      VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.stateOffset,
      Broadcast.leftIndex,Broadcast.rightIndex,ofFloat,toFloat]
  all_goals repeat' constructor
  all_goals funext j
  all_goals rcases j with ⟨j,⟨⟩⟩
  all_goals first | rfl | simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]


@[simp] theorem masked_regs {ι : Type} (r : RegionName) (off : ι → Nat)
    (value : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (l : List ι) (s : State α) :
    (l.foldl (fun q j => if active j then q.write r (off j) (value j) else q) s).regs=s.regs := by
  rw [State.masked_scatter_eq_filter,State.scatter_regs]

theorem forward_step (p : Ops α) (s : State α) (T D BD t : Nat)
    (carry : TileIndex [BD] → α)
    (hm : s.regs .bool [BD] "mask"=some (fun j => decide (lane s BD j<D)))
    (hx : s.regs .ptr [BD] "p_x"=some (fun j => (("x" : RegionName),row s t T D BD j)))
    (hg : s.regs .ptr [BD] "p_g"=some (fun j => (("g" : RegionName),row s t T D BD j)))
    (ho : s.regs .ptr [BD] "p_o"=some (fun j => (("o" : RegionName),row s t T D BD j)))
    (hc : s.regs .real [BD] "b_h"=some carry) :
    ∃ q, run (algebra p) (forwardBody D BD) s=some q ∧
      q.mem=(scatter s "o" BD (row s t T D BD)
        (fun j => lane s BD j<D) (next p s T D BD t carry)).mem ∧
      q.pids=s.pids ∧
      q.regs .bool [BD] "mask"=s.regs .bool [BD] "mask" ∧
      q.regs .nat [] "i_bh"=s.regs .nat [] "i_bh" ∧
      q.regs .nat [BD] "o_d"=s.regs .nat [BD] "o_d" ∧
      q.regs .ptr [BD] "p_x"=some (fun j => (("x" : RegionName),row s (t+1) T D BD j)) ∧
      q.regs .ptr [BD] "p_g"=some (fun j => (("g" : RegionName),row s (t+1) T D BD j)) ∧
      q.regs .ptr [BD] "p_o"=some (fun j => (("o" : RegionName),row s (t+1) T D BD j)) ∧
      q.regs .real [BD] "b_h"=some (next p s T D BD t carry) := by
  simp [forwardBody,forward,ComputeKernel.surfaceBody,run,step,evalExpr,
    evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,
    natLt,numericLt,Region.cast,algebra,store,State.setReg,hm,hx,hg,ho,hc,
    Broadcast.leftIndex,Broadcast.rightIndex,ofFloat,toFloat,masked_regs]
  simp [next,masked,read,scatter,row,lane,
    VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.rowOffset,
    VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.channel,
    VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.scatter,Nat.add_mul,Nat.add_assoc]
  refine ⟨?_,?_,?_,?_,?_⟩
  · apply VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.scatter_memory_congr
    rfl
  · funext j
    dsimp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    congr 1 <;> omega
  · funext j
    dsimp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    congr 1 <;> omega
  · funext j
    dsimp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    congr 1 <;> omega
  · funext j
    rcases j with ⟨j,⟨⟩⟩
    simp [next,masked,read,row,lane,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
      VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.channel,
      VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.rowOffset,Nat.add_mul,Nat.add_assoc]

end
end Execution
end VeriTile.Bench.Repairs.FusedRecurrentHgrn
