import bench.tritonbench_g.matmul_triton1.Proofs.FPSoftware
import bench.tritonbench_g.matmul_triton1.Proofs.FPShape
namespace VeriTile.Bench.TritonBenchG.MatmulTriton1.FPCorrect
open VeriTile.Triton FP.Structural FPValue FP.Equational
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Entire original K-loop and store, arbitrary block count, original program
addresses, and arbitrary aliases. Dot, initialization and independent outer
addition remain explicit backend parameters. -/
specification matmul_triton1_backend_correctness {α : Type} [Inhabited α]
    (rest : Algebra α) (P : Primitives α) (p : Parameters) (initial : State α)
    (hBK : 0<p.blockInner) (hN : p.blockColumns≤p.columns) :
    ∃ final, FP.Structural.exec (backend rest P) (source p) initial=some final ∧
      Post P p initial final := correctness rest P p initial hBK hN

/-- Complete binary32 software result with a valid explicit reduction tree for
every dot, rounded fp32 products, and a separately rounded accumulator addition.
The source retains allow_tf32=false. No finite-input or alias restriction is
added; native dot scheduling/fusion and storage/launch binding remain pending. -/
specification matmul_triton1_machine_correctness
    (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (initial : State Word)
    (hBK : 0<p.blockInner) (hN : p.blockColumns≤p.columns) :
    ∃ final, FP.Structural.exec (backend rest (software cfg plans)) (source p) initial=some final ∧
      (∀ i, final.mem p.output (cOffset p initial i)=.mk .fp32 (expected cfg plans p initial i)) ∧
      (∀ r o, (r≠p.output ∨ ∀ i : CIndex p, o≠cOffset p initial i) → final.mem r o=initial.mem r o) ∧
      (∀ k : Fin p.blockInner, some k ∈ (plans p.blockInner).tree.leaves) := by
  obtain ⟨final,he,hv,hf⟩ := correctness rest (software cfg plans) p initial hBK hN
  exact ⟨final,he,hv,hf,fun k => dot_retains_all_inputs _ k⟩

#axiomsClean matmul_triton1_backend_correctness
#axiomsClean matmul_triton1_machine_correctness
end
end VeriTile.Bench.TritonBenchG.MatmulTriton1.FPCorrect
