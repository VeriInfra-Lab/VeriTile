import bench.tritonbench_g.decay_cumsum.Proofs.FPBackwardStep

namespace VeriTile.Bench.Repairs.DecayCumsum.FPBackwardControl
open VeriTile.Triton FP.Structural FP.ProfiledRewrite FPOps
open VeriTile.Bench.Repairs.ChunkCumsumVector.FPStep (assign_compute)
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def firstCode (h : Call) (p : Phase) : List ComputeStmt :=
  [reverseIndex,assign32 h.BK "_g" (gateOp h.BK "p_g"),lastGate h.BK p.span]

def gateValue (M : Algebra α) (h : Call) (off : Fin h.BK → Nat) (u : State α) : Values α .real [h.BK] :=
  fun i => FPReference.gate M (raw h.BK "g" off u i)
def prefixState (M : Algebra α) (h : Call) (p : Phase) (n : Nat) (off : Fin h.BK → Nat) (u : State α) : State α :=
  let v := u.setReg "j" .int [] (fun _ => Int.ofNat (p.span-1-n))
  let v := v.setReg "_g" .real [h.BK] (gateValue M h off u)
  if n=0 then v.setReg "last_g" .real [h.BK] (gateValue M h off u) else v

theorem reverse_eval (M : Algebra α) (span n : Nat) (hn : n<span) (u : State α)
    (hi : u.regs .int [] "__rev_j"=some (fun _ => Int.ofNat n-Int.ofNat (span-1))) :
    step (engine M) reverseIndex u=some (u.setReg "j" .int [] (fun _ => Int.ofNat (span-1-n))) := by
  have value : (0:Int)-(Int.ofNat n-Int.ofNat (span-1))=Int.ofNat (span-1-n) := by
    simp only [Int.ofNat_eq_natCast]
    omega
  simp only [reverseIndex,step,evalExpr,FP.Structural.evalOp,hi]
  change some (u.setReg "j" .int [] (fun _ => (0:Int)-(Int.ofNat n-Int.ofNat (span-1))))=_
  rw [value]

theorem last_eval (M : Algebra α) (BK span n : Nat) (hn : n<span)
    (g : Values α .real [BK]) (u : State α)
    (hj : u.regs .int [] "j"=some (fun _ => Int.ofNat (span-1-n)))
    (hg : u.regs .real [BK] "_g"=some g) :
    step (engine M) (lastGate BK span) u=
      some (if n=0 then u.setReg "last_g" .real [BK] g else u) := by
  have condition : (span-1-n=span-1) ↔ n=0 := by omega
  simp only [lastGate,step,FP.Structural.evalOp,hj]
  simp [numeric,numericEq,bop,Broadcast.leftIndex,Broadcast.rightIndex,condition,
    assign32,refOp,run,step,evalExpr,evalComputeOp,FP.Structural.evalOp,ComputeDType.eraseDType,hg]
  split <;> rfl

theorem prefix_executes (M : Algebra α) (h : Call) (p : Phase) (n : Nat) (hn : n<p.span)
    (off : Fin h.BK → Nat) (u : State α)
    (hi : u.regs .int [] "__rev_j"=some (fun _ => Int.ofNat n-Int.ofNat (p.span-1)))
    (hp : u.regs .ptr [h.BK] "p_g"=some (ptr h.BK "g" off))
    (hm : u.regs .bool [h.BK] "mask"=some (fun _ => Bool.true)) :
    run (engine M) (firstCode h p) u=some (prefixState M h p n off u) := by
  unfold firstCode assign32
  rw [run,reverse_eval M p.span n hn u hi]
  simp only [bind,Option.bind_some]
  rw [assign_compute _ _ _ _ _ _ _ (gate_load M h.BK "g" off "p_g" _ (by simp [hp]) (by simp [hm]))]
  rw [run,last_eval M h.BK p.span n hn (gateValue M h off u) _ (by simp) (by simp [gateValue,raw]; rfl)]
  simp only [bind,Option.bind_some,run]
  rfl

#axiomsClean prefix_executes
end
end VeriTile.Bench.Repairs.DecayCumsum.FPBackwardControl
