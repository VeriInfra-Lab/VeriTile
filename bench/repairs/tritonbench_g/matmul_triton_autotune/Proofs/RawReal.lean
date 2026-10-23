import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.KernelRaw
import bench.tritonbench_g.matmul_triton_autotune.Proofs.Real

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel.RawReal
open VeriTile.Triton FP.Structural Whole
noncomputable section

/-- Independent GEMM and optional leaky-ReLU over the initial raw memory. -/
def mathematical (g : Geometry) (before : State ℝ) (i : Fin g.rows) (j : Fin g.columns) : ℝ :=
  Whole.mathematical g.request
    (fun i k => inputA g.request before i.val k.val)
    (fun k j => inputB g.request before k.val j.val) i j

/-- Every selected source program executes; all outputs implement the independent
GEMM and every cell outside the output footprint is preserved. -/
specification mathematical_kernel (g : Geometry) (before : State ℝ) (ids : List Nat)
    (order : ids.Perm (List.range (programs g.request))) (sepA : g.C≠g.A) (sepB : g.C≠g.B) :
    VeriTile.Spec.Real (∃ final,
      Repair.runPrograms (idealAlgebra id) idealDot g.request ids before=some final ∧
      (∀ i : Fin g.rows,∀ j : Fin g.columns,
        final.mem g.C (Whole.outputOffset g.request i.val j.val)=Cell.mk .fp16 (mathematical g before i j)) ∧
      (∀ region offset,¬Whole.Footprint g.request region offset → final.mem region offset=before.mem region offset)) := by
  obtain ⟨final,ran,values,frame⟩ := candidate_grid_order_correct (idealAlgebra id) idealDot
    (idealNumerics id) (ideal_primitives id) g.request before ids order sepA sepB
  refine ⟨final,ran,?_,frame⟩
  intro i j
  have value := values i.val i.isLt j.val j.isLt
  rw [ideal_reference id g.request before
    (fun i k => inputA g.request before i.val k.val)
    (fun k j => inputB g.request before k.val j.val)
    (fun _ _ => rfl) (fun _ _ => rfl) i j] at value
  exact value

#specNonCircular mathematical avoiding [Source.typed,Repair.runPrograms]
#axiomsClean mathematical_kernel
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel.RawReal
