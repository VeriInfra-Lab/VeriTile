import bench.tritonbench_g.triton_matmul.Proofs.Mathematical_FPReference
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue
open VeriTile.Triton FP.Structural
open KcacheCopyTriton.TypedCopy (masked_readback_active)
noncomputable section
def written {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial : State α) : State α :=
  (TileShape.allIndices [BM p,BN p]).foldl (fun t i => if active p initial i then
    t.write p.C (cOffset p initial i) (stored st (outputValue M D p st initial i)) else t) initial

def Post {α : Type} (p : Parameters) (st : Storage) (initial final : State α) (v : CIndex p → α) : Prop :=
  (∀ i, active p initial i → final.mem p.C (cOffset p initial i)=stored st (v i)) ∧
  (∀ r o, (r≠p.C ∨ ∀ i, active p initial i → o≠cOffset p initial i) → final.mem r o=initial.mem r o)


/-- Readback and frame of the independent mathematical masked-write fold. -/
theorem written_post {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial : State α)
    (hw : ∀ i j,active p initial i→active p initial j→cOffset p initial i=cOffset p initial j→i=j) :
    Post p st initial (written M D p st initial) (outputValue M D p st initial) := by
  constructor
  · intro i hi
    exact masked_readback_active _ _ _ _ _ i hi hw
  · intro r o h
    exact State.masked_scatter_frame _ _ _ _ _ _ _ (h.imp id (fun h i _=>h i)) _
#axiomsClean written_post
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue
