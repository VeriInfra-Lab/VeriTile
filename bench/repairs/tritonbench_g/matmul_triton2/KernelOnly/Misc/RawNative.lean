import bench.repairs.tritonbench_g.matmul_triton2.revision001.Proofs.Grid
import bench.repairs.tritonbench_g.matmul_triton2.revision001.Proofs.Native
namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.RawKernel
open VeriTile.Triton FP.Structural Revision001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section

local instance : Inhabited Native.Word := ⟨FP.Value.zero .fp32⟩

/-- Successful full-source execution and exact selected native instruction
outputs from raw FP32 cells, with every exterior cell preserved. -/
theorem kernel_native (ops : Native.Reference) (choice : Fin 8)
    (rest : Algebra Native.Word) (p : Core.Parameters)
    (geometry : Geometry.Valid (Core.geometry p)) (inner : 0 < p.blockInner)
    (input : p.input ≠ p.output) (weight : p.weight ≠ p.output) (before : State Native.Word) :
    ∃ after, Core.grid rest (Native.primitives (Native.referenceInstructions ops) (Native.family choice)) p before=some after ∧
      (∀ x : Core.LogicalIndex p,after.mem p.output (Geometry.address (Core.geometry p) x)=
        .mk .real (Core.logical (Native.primitives (Native.referenceInstructions ops) (Native.family choice))
          p before x.1.val x.2.val)) ∧
      (∀ r i,(r≠p.output ∨ ∀ x : Core.LogicalIndex p,i≠Geometry.address (Core.geometry p) x) →
        after.mem r i=before.mem r i) := by
  obtain ⟨after,run,values,frame⟩ := Core.grid_correct rest
    (Native.primitives (Native.referenceInstructions ops) (Native.family choice)) p geometry inner input weight before
  refine ⟨after,run,fun x => values x trivial,?_⟩
  intro r i outside
  apply frame r i
  exact outside.imp id (fun h x _ => h x)

#axiomsClean kernel_native
end
end VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.RawKernel
