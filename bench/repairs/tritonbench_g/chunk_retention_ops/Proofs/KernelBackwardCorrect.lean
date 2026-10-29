import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelBackwardReal
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelBackwardFrame
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelReal
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (withDot)
open VeriTile.Bench.Repairs.ChunkLinearAttn (hActive hOffset fOffset)
noncomputable section

/-- Kernel-only reverse-state correctness, including all history stores,
optional initial-state gradient, and exterior logical memory. -/
theorem backward_correct (c : Shape) (scale : ℝ) (seed terminal initial : Bool) (s : State ℝ)
    (hp : VeriTile.Bench.Repairs.ChunkLinearAttn.Geometry.StatePID (parameters c)
      ⟨s.pids 0,s.pids 1,s.pids 2⟩) :
    let p:=parameters c
    let pid : KernelBackward.PID:=⟨s.pids 0,s.pids 1,s.pids 2⟩
    ∃ t, FP.Structural.exec (engine (withDot algebra dotPrimitive)) (KernelBackward.kernel p scale terminal initial) s=some t ∧
      (∀ n,n<c.chunks → ∀ i : TileIndex [c.BK,c.BV],hActive p pid i →
        t.mem "dhistory" (hOffset p pid (KernelBackward.time p n) i)=.mk .real
          (MathTarget.future c (inputs c s.mem seed terminal) scale pid.z (c.boundary (c.chunks-n))
            (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val))) ∧
      (initial=Bool.true → ∀ i : TileIndex [c.BK,c.BV],hActive p pid i →
        t.mem "initialGradient" (fOffset p pid i)=.mk .real
          (MathTarget.initialGradient c (inputs c s.mem seed terminal) scale pid.z
            (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val))) ∧
      (∀ r off,KernelBackward.Exterior p r off → t.mem r off=s.mem r off) := by
  dsimp only
  let pid : KernelBackward.PID:=⟨s.pids 0,s.pids 1,s.pids 2⟩
  obtain ⟨t,ht,hm,hi⟩:=KernelBackward.program algebra dotPrimitive (parameters c) scale terminal initial s.mem s
    (KernelBackward.InputEq.refl _) (by simp [parameters])
  refine ⟨t,ht,?_,?_,?_⟩
  · intro n hn i hactive
    rw [hm,KernelBackward.state_read_history algebra dotPrimitive (parameters c) pid s.mem s.mem scale terminal initial n i hn hactive]
    change Cell.mk .real (KernelBackward.values algebra dotPrimitive (parameters c) pid s.mem scale terminal n i)=_
    rw [values_math c pid s.mem scale seed terminal n (Nat.le_of_lt hn) i hactive]
  · intro hflag i hactive
    subst initial
    rw [hm,KernelBackward.state_read_final algebra dotPrimitive (parameters c) pid s.mem s.mem scale terminal i hactive]
    change Cell.mk .real (KernelBackward.values algebra dotPrimitive (parameters c) pid s.mem scale terminal c.chunks i)=_
    rw [values_math c pid s.mem scale seed terminal c.chunks le_rfl i hactive]
    simp [MathTarget.initialGradient,Shape.boundary,pid]
  · intro r off h
    rw [hm]
    exact KernelBackward.memory_frame algebra dotPrimitive (parameters c) pid hp s.mem s.mem scale terminal initial r off h

#axiomsClean backward_correct
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelReal
