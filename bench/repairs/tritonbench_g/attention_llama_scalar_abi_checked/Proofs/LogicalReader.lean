import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.IO
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.RealPublicIO

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.LogicalReader
open VeriTile Triton Source
open VeriTile.Bench.Repairs.AttentionLlama HostGrid HostResult MaskCoverage
noncomputable section

theorem statement_iff (B : Nat) (c : Config) (scale : ℝ) (f : LogicalIO.Inputs B c → LogicalIO.Outputs B c)
    (connection : (LogicalIO.io B c scale).lift f = RealMathematical.attention B c scale) :
    (LogicalIO.io B c scale).ImplementsOn (LogicalIO.domain B c) f ↔
      (RealIO.io B c scale).ImplementsOn (RealLayout.domain B c) (RealMathematical.attention B c scale) := by
  change (RealIO.io B c scale).ImplementsOn (RealLayout.domain B c) ((LogicalIO.io B c scale).lift f) ↔ _
  rw [connection]

theorem correct (B : Nat) (c : Config) (scale : ℝ) (hD : 0<c.D) (hN : 0<c.N) (hSP : c.causal=Bool.true → c.SP=0) (sep : c.fp8=Bool.true → c.Q≠c.K ∧ c.Q≠c.V ∧ c.Out≠c.K ∧ c.Out≠c.V) (hinj : Function.Injective (IOLayout.output (B:=B) c)) (f : LogicalIO.Inputs B c → LogicalIO.Outputs B c)
    (connection : (LogicalIO.io B c scale).lift f = RealMathematical.attention B c scale) :
    (LogicalIO.io B c scale).ImplementsOn (LogicalIO.domain B c) f :=
  (statement_iff B c scale f connection).mpr (RealPublicIO.implements B c scale hD hN hSP sep hinj)

#axiomsClean statement_iff
#axiomsClean correct
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.LogicalReader
