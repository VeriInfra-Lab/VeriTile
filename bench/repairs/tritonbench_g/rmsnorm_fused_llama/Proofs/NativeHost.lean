import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.Native
import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.Host

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.Native
open VeriTile Triton FP.Structural
noncomputable section

/-- Raw caller words enter only through leaves; compiler selectors cannot read them. -/
def entry (memory : RegionName → Nat → Word) : State Term where
  mem := fun r o => .mk .real (.leaf r o (memory r o))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d

/-- Independent scalar source graph with a valid selected reduction tree and
local compiler fusion/operand ordering. There is no execution or output oracle. -/
def value (I : Instructions) (R : Reductions) (S : Schedule)
    (memory : RegionName → Nat → Word) (X W : RegionName) (q : Host.Request) (row k : Nat) : Word :=
  eval I S (Row.value (algebra R) (entry memory) X W row q.columns 16384 q.epsilon k)

/-- Successful full source grid, actual returned descriptor, every output word,
and caller frame. The same-call entry binding and fresh allocation are ordinary
runtime contracts. Actual local compiler choices and scalar instruction behavior
are parameters; no native-to-Real or whole-output hypothesis is assumed. -/
theorem whole_native (I : Instructions) (R : Reductions) (S : Schedule)
    (q : Host.Request) (memory : RegionName → Nat → Word)
    (call : Host.Preparation (State Term) (Cell Term) State.mem q)
    (hbefore : call.before=entry memory) :
    ∃ result, Host.invoke (algebra R) q call=some result ∧
      result.tensor=call.output ∧ result.tensor.shape=q.shape ∧
      result.tensor.dtype="float32" ∧ result.tensor.device=call.input.device ∧
      (∀ row, row<q.rows → ∀ k, k<q.columns →
        observe I S result.state result.tensor.region (row*q.columns+k)=
          value I R S memory call.input.region call.weight.region q row k) ∧
      (∀ r o, r≠result.tensor.region ∨ q.size≤o →
        observe I S result.state r o=memory r o) := by
  obtain ⟨result,he,ht,hs,hd,hdev,hv,hf⟩ := Host.whole_typed (algebra R) q call
  refine ⟨result,he,ht,hs,hd,hdev,?_,?_⟩
  · intro row hr k hk
    unfold observe
    rw [hv row hr k hk,hbefore]
    rfl
  · intro r o h
    unfold observe
    rw [hf r o h,hbefore]
    rfl

#axiomsClean whole_native
#specNonCircular entry avoiding [FP.Structural.exec,Host.invoke,Grid.run]
#specNonCircular value avoiding [FP.Structural.exec,Host.invoke,Grid.run]
end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.Native
