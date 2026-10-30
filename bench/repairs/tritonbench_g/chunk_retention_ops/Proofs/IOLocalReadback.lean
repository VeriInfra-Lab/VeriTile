import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelGradientReadback
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.Geometry

namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelGradient
open VeriTile Triton FP.Structural
open KernelBackward (Parameters PID Matrix)
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
open VeriTile.Bench.Repairs.ChunkDeltaFwd (foldRange_last scatter_readback)
open VeriTile.Bench.Repairs.ChunkLinearAttn (valueBlocks scatter_frame)
noncomputable section
variable {α : Type} [Inhabited α]

/-- Local scatter uniqueness needs no head-range premise: both addresses have
the same program base. The broader original arbitrary-PID observation domain
is therefore retained when exposing finite partial-value outputs. -/
theorem partial_local_collision (p : Parameters) (heads : Nat) (pid : PID) (va vb : Nat)
    (i j : TileIndex [p.BT,p.BV]) (hi : gradValueActive p pid va i) (hj : gradValueActive p pid vb j)
    (he : gradValueOffset p pid heads va i=gradValueOffset p pid heads vb j) : va=vb ∧ i=j := by
  unfold gradValueOffset at he
  rw [Nat.add_assoc,Nat.add_assoc] at he
  have he := Nat.add_left_cancel he
  obtain ⟨ht,hv⟩ := digit_injective _ _ _ _ _ hi.2 hj.2 he
  have ht := Nat.add_left_cancel ht
  obtain ⟨hb,hl⟩ := digit_injective _ _ _ _ _ i.2.1.isLt j.2.1.isLt hv
  exact ⟨hb,Prod.ext (Fin.ext ht) (Prod.ext (Fin.ext hl) (Subsingleton.elim _ _))⟩

theorem gradient_read_value_local (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (heads : Nat) (input m : Memory α) (scale : ℝ) (vb : Nat) (i : TileIndex [p.BT,p.BV])
    (hv : vb<valueBlocks p) (hi : gradValueActive p pid vb i) :
    gradientMemory M D p pid heads input m scale "valuePartials" (gradValueOffset p pid heads vb i)=
      .mk .real (VeriTile.Bench.Repairs.ChunkLinearAttn.cast M (gradValue M D p pid input scale vb i)) := by
  have hk : ∀ (dest : RegionName), dest≠"valuePartials" → ∀ a mem,
      gradKeyWrite M p pid dest a mem "valuePartials" (gradValueOffset p pid heads vb i)=
      mem "valuePartials" (gradValueOffset p pid heads vb i) := by
    intro dest hd a mem
    exact scatter_frame _ _ _ _ _ _ _ _ (Or.inl (Ne.symm hd))
  unfold gradientMemory
  rw [hk "keyGradient" (by decide),hk "queryGradient" (by decide)]
  apply foldRange_last _ (fun mem : Memory α=>mem "valuePartials" (gradValueOffset p pid heads vb i))
    (valueBlocks p) vb m _ hv
  · intro mem
    exact scatter_readback _ _ _ _ _ i hi (fun a b ha hb he=>
      (partial_local_collision p heads pid vb vb a b ha hb he).2)
  · intro other hother _ mem
    apply scatter_frame
    right
    intro j _ hj he
    have hb := (partial_local_collision p heads pid vb other i j hi hj he).1
    omega

#axiomsClean partial_local_collision
#axiomsClean gradient_read_value_local
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelGradient
