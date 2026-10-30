import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.IO
import bench.repairs.tritonbench_g.attention_llama.Proofs.HostResult

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.RealMathematical
open VeriTile Triton Source
open VeriTile.Bench.Repairs.AttentionLlama HostGrid HostResult MaskCoverage
noncomputable section
open scoped BigOperators

/-- Mathematical attention: the weighted exponential sum divided by its
normalizing sum, over exactly the source's logical causal/key range. -/
def attention (B : Nat) (c : Config) (scale : ℝ) (vals : (RealLayout.memory B c).Ctx) :
    (o : Fin (RealLayout.memory B c).nOut)→
      Fin ((RealLayout.memory B c).oarity o)→((RealLayout.memory B c).oty o).carrier :=
  fun _ j=>
    let q:=RealLayout.queryIndex B c j
    let i : Index B c.NH c.H c.D:=⟨q.1,q.2.1,q.2.2.1,q.2.2.2.1⟩
    TailMath.expected (HostResult.score (inputs B c vals) scale i)
      (fun k=>(inputs B c vals).value i.batch i.head k i.channel)
      (logicalKeys c.N c.causal c.SP i.query.val)

#specNonCircular attention avoiding [Triton.exec,Triton.stepStmts]
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.RealMathematical
