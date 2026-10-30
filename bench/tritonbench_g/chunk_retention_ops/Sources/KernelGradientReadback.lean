import bench.tritonbench_g.chunk_retention_ops.Sources.KernelGradientProgram
import bench.tritonbench_g.chunk_linear_attn.Readback
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelGradient
open VeriTile Triton FP.Structural
open KernelBackward (Parameters PID Matrix)
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
open VeriTile.Bench.Repairs.ChunkDeltaFwd (foldRange_last scatter_readback)
open VeriTile.Bench.Repairs.ChunkLinearAttn (valueBlocks scatter_frame)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]
theorem gradKey_write_read (M : Algebra α) (p : Parameters) (pid : PID) (r : RegionName)
    (a : Matrix α p.BT p.BK) (m : Memory α) (i : TileIndex [p.BT,p.BK]) (hi : gradKeyActive p pid i) :
    gradKeyWrite M p pid r a m r (gradKeyOffset p pid i)=.mk .real (VeriTile.Bench.Repairs.ChunkLinearAttn.cast M (a i)) :=
  scatter_readback _ _ _ _ _ i hi (fun a b ha hb he => (VeriTile.Bench.Repairs.ChunkLinearAttn.Geometry.gradient_collision p pid pid a b ha hb he).2)

theorem gradient_read_query (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input m : Memory α) (scale : ℝ) (i : TileIndex [p.BT,p.BK]) (hi : gradKeyActive p pid i) :
    gradientMemory M D p pid (p.B*p.H) input m scale "queryGradient" (gradKeyOffset p pid i)=
      .mk .real (VeriTile.Bench.Repairs.ChunkLinearAttn.cast M (gradQueryResult M D p pid input scale (gradStates M D p pid input scale (valueBlocks p)) i)) := by
  unfold gradientMemory
  rw [show ∀ a mem, gradKeyWrite M p pid "keyGradient" a mem "queryGradient" (gradKeyOffset p pid i)=
      mem "queryGradient" (gradKeyOffset p pid i) from fun a mem => scatter_frame _ _ _ _ _ _ _ _ (Or.inl (by decide))]
  exact gradKey_write_read M p pid "queryGradient" _ _ i hi

theorem gradient_read_key (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input m : Memory α) (scale : ℝ) (i : TileIndex [p.BT,p.BK]) (hi : gradKeyActive p pid i) :
    gradientMemory M D p pid (p.B*p.H) input m scale "keyGradient" (gradKeyOffset p pid i)=
      .mk .real (VeriTile.Bench.Repairs.ChunkLinearAttn.cast M (gradKeyResult M D p pid input scale (gradStates M D p pid input scale (valueBlocks p)) i)) :=
  gradKey_write_read M p pid "keyGradient" _ _ i hi

theorem gradient_read_value (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input m : Memory α) (scale : ℝ) (vb : Nat) (i : TileIndex [p.BT,p.BV])
    (hh : pid.z<p.B*p.H) (hv : vb<valueBlocks p) (hi : gradValueActive p pid vb i) :
    gradientMemory M D p pid (p.B*p.H) input m scale "valuePartials" (gradValueOffset p pid (p.B*p.H) vb i)=
      .mk .real (VeriTile.Bench.Repairs.ChunkLinearAttn.cast M (gradValue M D p pid input scale vb i)) := by
  have hk : ∀ (dest : RegionName), dest≠"valuePartials" → ∀ a mem,
      gradKeyWrite M p pid dest a mem "valuePartials" (gradValueOffset p pid (p.B*p.H) vb i)=
      mem "valuePartials" (gradValueOffset p pid (p.B*p.H) vb i) := by
    intro dest hd a mem
    exact scatter_frame _ _ _ _ _ _ _ _ (Or.inl (Ne.symm hd))
  unfold gradientMemory
  rw [hk "keyGradient" (by decide),hk "queryGradient" (by decide)]
  apply foldRange_last _ (fun mem : Memory α => mem "valuePartials" (gradValueOffset p pid (p.B*p.H) vb i))
    (valueBlocks p) vb m _ hv
  · intro mem
    exact scatter_readback _ _ _ _ _ i hi (fun a b ha hb he =>
      (VeriTile.Bench.Repairs.ChunkLinearAttn.Geometry.partial_collision p (p.B*p.H) pid pid vb vb a b hh hh ha hb he).2.2)
  · intro other hvother hother mem
    apply scatter_frame
    right
    intro j _ hj he
    have hb := (VeriTile.Bench.Repairs.ChunkLinearAttn.Geometry.partial_collision p (p.B*p.H) pid pid vb other i j hh hh hi hj he).2.1
    omega


#axiomsClean gradient_read_query
#axiomsClean gradient_read_key
#axiomsClean gradient_read_value
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelGradient
