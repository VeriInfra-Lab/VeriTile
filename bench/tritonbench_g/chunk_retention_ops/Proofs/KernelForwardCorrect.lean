import bench.tritonbench_g.chunk_retention_ops.Sources.KernelForwardReal
import bench.tritonbench_g.chunk_retention_ops.Sources.KernelForwardReadback
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelReal
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open KernelBackward (PID)
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (withDot)
open VeriTile.Bench.Repairs.ChunkLinearAttn (hActive hOffset fOffset)
noncomputable section

/-- Full forward-state quotation, every active history store, optional final
state, and exact exterior frame. The independent target is the dense recurrence. -/
theorem forward_correct (c : Shape) (seed final : Bool) (pid : PID) (m : Memory ℝ) :
    ∃ t,FP.Structural.exec (engine (withDot algebra dotPrimitive))
        (KernelForward.kernel c seed final) (KernelForward.state c pid m)=some t ∧
      (∀ n,n<c.chunks → ∀ i : TileIndex [c.BK,c.BV],hActive (parameters c) pid i →
        t.mem "h" (hOffset (parameters c) pid n i)=.mk .real
          (MathTarget.state c (inputs c m seed Bool.false) Bool.true pid.z (c.boundary n)
            (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val))) ∧
      (final=Bool.true → ∀ i : TileIndex [c.BK,c.BV],hActive (parameters c) pid i →
        t.mem "ht" (fOffset (parameters c) pid i)=.mk .real
          (MathTarget.finalState c (inputs c m seed Bool.false) Bool.true pid.z
            (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val))) ∧
      (∀ r off,¬KernelForward.Footprint c pid final r off → t.mem r off=m r off) := by
  obtain ⟨t,ht,hm⟩:=KernelForward.source algebra dotPrimitive c seed final pid m
  refine ⟨t,ht,?_,?_,?_⟩
  · intro n hn i hi
    rw [hm,KernelForward.state_read_history algebra dotPrimitive c pid m m seed final n i hn hi]
    change Cell.mk .real (KernelForward.values algebra dotPrimitive c pid m seed n i)=_
    rw [forward_values_math c pid m seed Bool.false n (Nat.le_of_lt hn) i hi]
  · intro hf i hi
    subst final
    rw [hm,KernelForward.state_read_final algebra dotPrimitive c pid m m seed i hi]
    change Cell.mk .real (KernelForward.values algebra dotPrimitive c pid m seed c.chunks i)=_
    rw [forward_values_math c pid m seed Bool.false c.chunks le_rfl i hi]
    simp only [MathTarget.finalState,c.boundary_final]
  · intro r off h
    rw [hm]
    exact KernelForward.memory_frame algebra dotPrimitive c pid m m seed final r off h

#axiomsClean forward_correct
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelReal
