import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.KernelRaw
import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.KernelNative

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel.RawNative
open VeriTile.Triton FP.Structural Whole
noncomputable section
local instance : Inhabited Floating.Word := ⟨FP.Value.zero .fp32⟩

/-- Initial-memory padded MMA/MFMA recurrence with the original scalar FP32
operations and mandatory FP16 narrowing. -/
def floating (instructions : Floating.Instructions) (g : Geometry)
    (before : State Floating.Word) (row col : Nat) : Floating.Word :=
  Whole.reference (Floating.numerics instructions g.config) g.request before row col

/-- Complete shared-source execution for raw word memory, native output
observations and full exterior-cell frame. The fixed instruction boundary is unchanged. -/
specification floating_kernel (instructions : Floating.Instructions) (g : Geometry)
    (before : State Floating.Word) (ids : List Nat)
    (order : ids.Perm (List.range (programs g.request))) (sepA : g.C≠g.A) (sepB : g.C≠g.B) :
    ∃ final,Repair.runPrograms Floating.scalar (Floating.dot instructions g.config)
        g.request ids before=some final ∧
      (∀ row,row<g.rows → ∀ col,col<g.columns →
        final.mem g.C (Whole.outputOffset g.request row col)=Cell.mk .fp16 (floating instructions g before row col)) ∧
      (∀ region offset,¬Whole.Footprint g.request region offset → final.mem region offset=before.mem region offset) :=
  candidate_grid_order_correct Floating.scalar (Floating.dot instructions g.config)
    (Floating.numerics instructions g.config) (Floating.primitives instructions g.config)
    g.request before ids order sepA sepB

#specNonCircular floating avoiding [Source.typed,Repair.runPrograms]
#axiomsClean floating_kernel
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel.RawNative
