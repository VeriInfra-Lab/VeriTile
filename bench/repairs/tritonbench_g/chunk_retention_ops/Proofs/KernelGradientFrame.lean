import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelGradientReadback
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelGradient
open VeriTile Triton FP.Structural
open KernelBackward (Parameters PID Matrix)
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
open VeriTile.Bench.Repairs.ChunkDeltaFwd (foldRange_observe)
open VeriTile.Bench.Repairs.ChunkLinearAttn (valueBlocks scatter_frame GradientFootprint)
noncomputable section
variable {α : Type} [Inhabited α]
theorem gradient_frame (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input m : Memory α) (scale : ℝ) (r : RegionName) (o : Nat)
    (safe : ¬GradientFootprint p pid r o) : gradientMemory M D p pid (p.B*p.H) input m scale r o=m r o := by
  have hk : ∀ (dest : RegionName), (dest="queryGradient" ∨ dest="keyGradient") → ∀ (a : Matrix α p.BT p.BK) mem,
      gradKeyWrite M p pid dest a mem r o=mem r o := by
    intro dest hd a mem
    apply scatter_frame
    by_cases hr : r=dest
    · right
      intro i _ hi he
      exact safe (Or.inl ⟨hr ▸ hd,i,hi,he⟩)
    · exact Or.inl hr
  unfold gradientMemory
  rw [hk "keyGradient" (Or.inr rfl),hk "queryGradient" (Or.inl rfl)]
  apply foldRange_observe _ (fun mem : Memory α => mem r o)
  intro vb hv mem
  apply scatter_frame
  by_cases hr : r="valuePartials"
  · right
    intro i _ hi he
    exact safe (Or.inr ⟨hr,vb,hv,i,hi,he⟩)
  · exact Or.inl hr


#axiomsClean gradient_frame
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelGradient
