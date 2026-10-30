import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelGradientRealResults
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelGradientFrame
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.RealBody
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelReal
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open KernelBackward (PID)
open KernelGradient
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (withDot)
noncomputable section
set_option maxHeartbeats 6000000

/-- Standalone QKV correctness for its genuine history inputs. The target is the
independent chunk adjoint contraction. dV retains its per-key-block output. -/
theorem gradient_correct (c : Shape) (scale : ℝ) (s : State ℝ)
    (hheads : s.numPids 2=c.B*c.H) (hh : s.pids 2<c.B*c.H) (hc : s.pids 1<c.chunks) :
    let p:=parameters c
    let pid : PID:=⟨s.pids 0,s.pids 1,s.pids 2⟩
    let xs:=inputs c s.mem Bool.false Bool.false
    ∃ t,FP.Structural.exec (engine (withDot algebra dotPrimitive)) (kernel p scale) s=some t ∧
      (∀ i : TileIndex [64,c.BK],gradKeyActive p pid i →
        t.mem "queryGradient" (gradKeyOffset p pid i)=.mk .real
          (ScalarSource.queryTile c xs scale pid.z pid.y i.1.val (pid.x*c.BK+i.2.1.val) (history c pid s.mem))) ∧
      (∀ i : TileIndex [64,c.BK],gradKeyActive p pid i →
        t.mem "keyGradient" (gradKeyOffset p pid i)=.mk .real
          (ScalarSource.keyTile c xs scale pid.z pid.y i.1.val (pid.x*c.BK+i.2.1.val) (dhistory c pid s.mem))) ∧
      (∀ vb,vb<c.valueBlocks → ∀ i : TileIndex [64,c.BV],gradValueActive p pid vb i →
        t.mem "valuePartials" (gradValueOffset p pid (c.B*c.H) vb i)=.mk .real
          (ScalarSource.valuePartial c xs scale pid.x pid.z pid.y i.1.val (vb*c.BV+i.2.1.val) (dhistory c pid s.mem))) ∧
      (∀ r off,¬VeriTile.Bench.Repairs.ChunkLinearAttn.GradientFootprint p pid r off → t.mem r off=s.mem r off) := by
  dsimp only
  let pid : PID:=⟨s.pids 0,s.pids 1,s.pids 2⟩
  obtain ⟨t,ht,hm⟩:=gradientProgram algebra dotPrimitive (parameters c) scale s (by simp [parameters])
  rw [hheads] at hm
  refine ⟨t,ht,?_,?_,?_,?_⟩
  · intro i hi
    rw [hm]
    trans Cell.mk .real (VeriTile.Bench.Repairs.ChunkLinearAttn.cast algebra
      (gradQueryResult algebra dotPrimitive (parameters c) pid s.mem scale (accumulated c pid s.mem scale) i))
    · exact gradient_read_query algebra dotPrimitive (parameters c) pid s.mem s.mem scale i hi
    change Cell.mk .real (gradQueryResult algebra dotPrimitive (parameters c) pid s.mem scale (accumulated c pid s.mem scale) i)=_
    rw [query_result_eq c pid s.mem scale Bool.false Bool.false i]
    exact congrArg (Cell.mk .real) (RealScalar.query_real c _ scale pid.z pid.y i.1.val _ _ hi.1 i.1.isLt hi.2)
  · intro i hi
    rw [hm]
    trans Cell.mk .real (VeriTile.Bench.Repairs.ChunkLinearAttn.cast algebra
      (gradKeyResult algebra dotPrimitive (parameters c) pid s.mem scale (accumulated c pid s.mem scale) i))
    · exact gradient_read_key algebra dotPrimitive (parameters c) pid s.mem s.mem scale i hi
    change Cell.mk .real (gradKeyResult algebra dotPrimitive (parameters c) pid s.mem scale (accumulated c pid s.mem scale) i)=_
    have hl:= (RealScalar.time_lane c pid.y i.1.val hc i.1.isLt).mp hi.1
    rw [key_result_eq c pid s.mem scale Bool.false Bool.false i]
    exact congrArg (Cell.mk .real) (RealScalar.key_real c _ scale pid.z pid.y i.1.val _ _ hc hl hi.2)
  · intro vb hv i hi
    rw [hm]
    trans Cell.mk .real (VeriTile.Bench.Repairs.ChunkLinearAttn.cast algebra
      (gradValue algebra dotPrimitive (parameters c) pid s.mem scale vb i))
    · exact gradient_read_value algebra dotPrimitive (parameters c) pid s.mem s.mem scale vb i hh hv hi
    change Cell.mk .real (gradValue algebra dotPrimitive (parameters c) pid s.mem scale vb i)=_
    have hl:= (RealScalar.time_lane c pid.y i.1.val hc i.1.isLt).mp hi.1
    rw [value_result_eq c pid s.mem scale Bool.false Bool.false vb i]
    exact congrArg (Cell.mk .real) (RealScalar.value_real c _ scale pid.x pid.z pid.y i.1.val _ _ hc hl hi.2)
  · intro r off h
    rw [hm]
    exact gradient_frame algebra dotPrimitive (parameters c) pid s.mem s.mem scale r off h

#axiomsClean gradient_correct
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelReal
