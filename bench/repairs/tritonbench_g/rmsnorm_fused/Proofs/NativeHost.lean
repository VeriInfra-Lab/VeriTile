import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.Native
import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.Host

namespace VeriTile.Bench.Repairs.RmsnormFused.Native
open VeriTile Triton FP.Structural
noncomputable section

/-- Raw caller words enter only through leaves; compiler selectors cannot read them. -/
def entry (memory : RegionName → Nat → Word) : State Term where
  mem := fun r o => .mk .real (.leaf r o (memory r o))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d

/-- Independent scalar graph with the actual reduction tree and local compiler
fusion/operand order. No execution, output oracle or whole-result law appears. -/
def value (I : Instructions) (R : Reductions) (S : Schedule)
    (memory : RegionName → Nat → Word) (X W : RegionName) (q : Host.Request) (row k : Nat) : Word :=
  eval I S (Row.value (algebra R) (entry memory) X W row q.columns q.block q.epsilon k)

theorem whole_native (I : Instructions) (R : Reductions) (S : Schedule)
    (q : Host.Request) (memory : RegionName → Nat → Word)
    (call : Host.Preparation (State Term) (Cell Term) State.mem q)
    (hbefore : call.before=entry memory) (schedule : List Nat)
    (all : schedule.Perm q.geometry.grid) :
    ∃ result,Host.invoke (algebra R) q call schedule=some result ∧
      result.tensor=call.output ∧ result.tensor.shape=q.shape ∧
      result.tensor.dtype="float32" ∧ result.tensor.device=call.input.device ∧
      result.currentDevice=call.callerDevice ∧
      result.launchDevice=(if q.rows=0 then none else some call.input.device) ∧
      (∀ row k,row<q.rows → k<q.columns →
        observe I S result.state result.tensor.region (row*q.columns+k)=
          value I R S memory call.input.region call.weight.region q row k) ∧
      (∀ r o,r≠result.tensor.region ∨ q.size≤o →
        observe I S result.state r o=memory r o) := by
  obtain ⟨result,he,ht,hs,hd,hdev,hctx,hlaunch,hv,hf⟩ :=
    Host.whole_typed (algebra R) q call schedule all
  refine ⟨result,he,ht,hs,hd,hdev,hctx,hlaunch,?_,?_⟩
  · intro row k hr hk
    unfold observe
    rw [hv row k hr hk,hbefore]
    rfl
  · intro r o h
    unfold observe
    rw [hf r o h,hbefore]
    rfl

#axiomsClean whole_native
#specNonCircular entry avoiding [FP.Structural.exec,Host.invoke,Row.source]
#specNonCircular value avoiding [FP.Structural.exec,Host.invoke,Row.source]
end
end VeriTile.Bench.Repairs.RmsnormFused.Native
