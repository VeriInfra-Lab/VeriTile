import bench.tritonbench_g.matmul_triton2.Proofs.FPSoftware
import bench.tritonbench_g.matmul_triton2.Proofs.FPShape
namespace VeriTile.Bench.TritonBenchG.MatmulTriton2.FPCorrect
open VeriTile.Triton FP.Structural FPValue FP.Equational
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Entire accepted K=BK*steps program with all group calculations, six strides,
K-load masks and final M/N mask. No input/output separation or finite-input
condition is introduced; dot precision remains an explicit backend obligation. -/
specification matmul_triton2_backend_correctness {α : Type} [Inhabited α]
    (rest : Algebra α) (P : Primitives α) (p : Parameters) (initial : State α)
    (hBK : 0<p.blockInner) (hInj : Function.Injective (FPValue.cOffset p initial)) :
    ∃ final, FP.Structural.exec (backend rest P) (source p) initial=some final ∧
      Post P p initial final := correctness rest P p initial hBK hInj

/-- Explicit software input precision plus complete per-dot tree and separate
outer fp32 additions. Source dot options are {}, not allow_tf32=false.
No mode is asserted to be the unspecified native default. The two TF32 modes
are concrete named software rules, with device binding still outstanding. -/
specification matmul_triton2_machine_correctness
    (rest : Algebra Word) (mode : InputMode) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (initial : State Word)
    (hBK : 0<p.blockInner) (hInj : Function.Injective (FPValue.cOffset p initial)) :
    ∃ final, FP.Structural.exec (backend rest (software mode cfg plans)) (source p) initial=some final ∧
      (∀ i, FPValue.active p initial i → final.mem p.output (FPValue.cOffset p initial i)=
        .mk .fp32 (expected mode cfg plans p initial i)) ∧
      (∀ r o, (r≠p.output ∨ ∀ i : CIndex p, FPValue.active p initial i → o≠FPValue.cOffset p initial i) →
        final.mem r o=initial.mem r o) ∧
      (∀ k : Fin p.blockInner, some k ∈ (plans p.blockInner).tree.leaves) := by
  obtain ⟨final,he,hv,hf⟩ := correctness rest (software mode cfg plans) p initial hBK hInj
  exact ⟨final,he,hv,hf,fun k => dot_retains_all_inputs _ k⟩

/-- The old IO face's row-major address conditions directly instantiate the
same complete floating value contract, without extra stride/alias conditions. -/
specification matmul_triton2_row_major_machine_correctness
    (rest : Algebra Word) (mode : InputMode) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (initial : State Word)
    (hBK : 0<p.blockInner) (hcn : p.outputColumn=1) (hBN : p.blockColumns≤p.outputRow) :
    ∃ final, FP.Structural.exec (backend rest (software mode cfg plans)) (source p) initial=some final ∧
      Post (software mode cfg plans) p initial final :=
  correctness rest (software mode cfg plans) p initial hBK (rowMajor_injective p initial hcn hBN)

#axiomsClean matmul_triton2_backend_correctness
#axiomsClean matmul_triton2_machine_correctness
#axiomsClean matmul_triton2_row_major_machine_correctness
end
end VeriTile.Bench.TritonBenchG.MatmulTriton2.FPCorrect
