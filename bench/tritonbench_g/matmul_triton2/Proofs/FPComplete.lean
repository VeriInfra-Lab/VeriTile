import bench.tritonbench_g.matmul_triton2.Proofs.FPReference
namespace VeriTile.Bench.Repairs.MatmulTriton2.Revision001.Core
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def written (P : Primitives α) (p : Parameters) (initial : State α) : State α :=
  (TileShape.allIndices [p.blockRows,p.blockColumns]).foldl
    (fun t i => if active p initial i then t.write p.output (cOffset p initial i)
      (.mk .real (accumulation P p initial (steps p) i)) else t) initial

def Post (P : Primitives α) (p : Parameters) (initial final : State α) : Prop :=
  (∀ i, active p initial i → final.mem p.output (cOffset p initial i)=
    .mk .real (accumulation P p initial (steps p) i)) ∧
  (∀ r o, (r≠p.output ∨ ∀ i : CIndex p, active p initial i → o≠cOffset p initial i) →
    final.mem r o=initial.mem r o)

/-- The row-major conditions of the old IO headline imply the original
arbitrary-stride summary's address injectivity condition. -/
theorem rowMajor_injective (p : Parameters) (initial : State α)
    (hcn : p.outputColumn=1) (hBN : p.blockColumns≤p.outputRow) :
    Function.Injective (cOffset p initial) := by
  have heq : cOffset p initial = fun i : CIndex p =>
      (rowBlock p initial*p.blockRows*p.outputRow+columnBlock p initial*p.blockColumns)+
        i.1.val*p.outputRow+i.2.1.val := by
    funext i
    simp only [cOffset,row,column,hcn]
    ring
  rw [heq]
  exact rowMajor2D_inj _ p.outputRow hBN

#axiomsClean rowMajor_injective
end
end VeriTile.Bench.Repairs.MatmulTriton2.Revision001.Core
