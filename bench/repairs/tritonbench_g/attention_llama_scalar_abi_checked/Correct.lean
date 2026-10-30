import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.LogicalReader
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.RealIO
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.RealPublicIO

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.RealCorrectness
open VeriTile Triton Source
open VeriTile.Bench.Repairs.AttentionLlama HostGrid HostResult MaskCoverage
open scoped BigOperators VeriTile.Triton.LogicalDomainIO
noncomputable section

/-- Independent attention on the finite logical arrays. The selected byte
K/V roles are decoded by the existing input view; the key set has exactly the
logical N bound and causal position test. -/
def attention (B : Nat) (c : Config) (scale : ℝ) (xs : LogicalIO.Inputs B c)
    (b : Fin B) (h : Fin c.NH) (query : Fin c.H) (channel : Fin c.D) : ℝ :=
  let score := fun k =>
    (∑ d : Fin c.D, xs.query b h query d * xs.key b h k d) * scale
  let keys := (Finset.range c.N).filter (fun k => c.causal = Bool.false ∨ k + c.SP ≤ query.val)
  (∑ k ∈ keys, Real.exp (score k) * xs.value b h k channel) /
    (∑ k ∈ keys, Real.exp (score k))

/-- The actual runtime-scale source implements mathematical attention over
all query/batch/head programs and caller strides. The byte branch requires
finite E5M2 inputs only in masked blocks reached by the source. -/
specification attention_kernel_grid (B : Nat) (c : Config) (scale : ℝ)
    (hD : 0<c.D) (hN : 0<c.N) (hSP : c.causal=Bool.true → c.SP=0)
    (sep : c.fp8=Bool.true → c.Q≠c.K ∧ c.Q≠c.V ∧ c.Out≠c.K ∧ c.Out≠c.V)
    (hinj : Function.Injective (IOLayout.output (B:=B) c)) :
    Spec.Real (LogicalIO.io B c scale ⊨{LogicalIO.domain B c} attention B c scale) :=
  LogicalReader.correct B c scale hD hN hSP sep hinj _ rfl

#specNonCircular attention avoiding [Triton.exec,Triton.stepStmts]
#axiomsClean attention_kernel_grid
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.RealCorrectness
