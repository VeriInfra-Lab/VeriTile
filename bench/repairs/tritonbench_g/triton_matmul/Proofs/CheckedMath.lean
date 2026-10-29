import bench.repairs.tritonbench_g.triton_matmul.Proofs.FiniteHost
import bench.repairs.tritonbench_g.triton_matmul.Proofs.Real
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.CheckedMath
open VeriTile.Triton FP.Structural
open Contract TensorBinding InputAdapter
noncomputable section

/-- Independent full-K real GEMM of original logical tensor values. `decode`
is the fixed ideal interpretation of a storage payload; native rounding is
retained separately by the floating-point result. -/
def expected (read : Read) (decode : Word → ℝ) (a b : Tensor)
    (la : Laws (storage a.metadata) a) (lb : Laws (storage b.metadata) b) (i : OutputIndex a b) : ℝ :=
  ∑ k : Fin (columns a.metadata),
    decode (logicalValue read (storage a.metadata) a la (i.1.val*columns a.metadata+k.val)) *
      decode (logicalValue read (storage b.metadata) b lb (k.val*columns b.metadata+i.2.val))

theorem reference (read : Read) (decode : Word → ℝ) (a b : Tensor)
    (la : Laws (storage a.metadata) a) (lb : Laws (storage b.metadata) b)
    (h : InputContract a.metadata b.metadata) (s : State ℝ) (bound : Inputs read decode a b s) (i : OutputIndex a b) :
    logicalReference idealNumerics (request a b) (CheckedHost.left read decode a b la)
      (CheckedHost.right read decode a b lb) i.1 i.2=expected read decode a b la lb i := by
  have input := CheckedHost.source_inputs read decode a b la lb h s bound
  rw [←reference_logical idealNumerics (request a b) s _ _ input.1 input.2 i.1 i.2,
    ideal_reference (request a b) s _ _ input.1 input.2 i.1 i.2]
  rfl

#specNonCircular expected avoiding [FiniteHost.call,CheckedHost.call,Source.typed,FP.Structural.exec]
#axiomsClean reference
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.CheckedMath
