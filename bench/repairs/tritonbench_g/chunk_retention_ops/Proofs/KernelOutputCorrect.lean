import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelOutputReal
import bench.repairs.tritonbench_g.chunk_linear_attn.Readback
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelOutput
open VeriTile Triton FP.Structural
open KernelBackward (PID)
open KernelReal (parameters)
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
open VeriTile.Bench.Repairs.ChunkDeltaFwd (scatter_readback)
open VeriTile.Bench.Repairs.ChunkLinearAttn (outputActive outputOffset scatter_frame)
noncomputable section
variable {α : Type} [Inhabited α]

def Footprint (c : Shape) (pid : PID) (r : RegionName) (off : Nat) :=
  r="o" ∧ ∃ i : TileIndex [64,c.BV],outputActive (parameters c) pid i ∧ off=outputOffset (parameters c) pid i

theorem memory_read (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ)
    (pid : PID) (input m : Memory α) (i : TileIndex [64,c.BV])
    (hi : outputActive (parameters c) pid i) :
    memory M D c scale pid input m "o" (outputOffset (parameters c) pid i)=
      .mk .real (outputValue M D c scale pid input i) :=
  scatter_readback _ _ _ _ _ i hi (fun a b ha hb he=>
    (VeriTile.Bench.Repairs.ChunkLinearAttn.Geometry.output_collision (parameters c) pid pid a b ha hb he).2)

theorem memory_frame (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ)
    (pid : PID) (input m : Memory α) (r : RegionName) (off : Nat)
    (safe : ¬Footprint c pid r off) : memory M D c scale pid input m r off=m r off := by
  apply scatter_frame
  by_cases hr : r="o"
  · right
    intro i _ hi he
    exact safe ⟨hr,i,hi,he⟩
  · exact Or.inl hr

#axiomsClean memory_read
#axiomsClean memory_frame
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelOutput

namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelReal
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open KernelBackward (PID)
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (withDot)
open VeriTile.Bench.Repairs.ChunkLinearAttn (outputActive outputOffset)
noncomputable section

/-- Standalone output correctness for the actual history input, expressed by an
independent dense chunk contraction. No history/output equality is a premise. -/
theorem output_correct (c : Shape) (scale : ℝ) (pid : PID) (m : Memory ℝ) :
    ∃ t,FP.Structural.exec (engine (withDot algebra dotPrimitive))
        (KernelOutput.kernel c scale) (KernelOutput.state c pid m)=some t ∧
      (∀ i : TileIndex [64,c.BV],outputActive (parameters c) pid i →
        t.mem "o" (outputOffset (parameters c) pid i)=.mk .real
          (ScalarSource.outputTile c (inputs c m Bool.false Bool.false) scale pid.z pid.y i.1.val
            (pid.x*c.BV+i.2.1.val) (O.history c (KernelOutput.state c pid m)))) ∧
      (∀ r off,¬KernelOutput.Footprint c pid r off → t.mem r off=m r off) := by
  obtain ⟨t,ht,hm⟩:=KernelOutput.source algebra dotPrimitive c scale pid m
  refine ⟨t,ht,?_,?_⟩
  · intro i hi
    rw [hm,KernelOutput.memory_read algebra dotPrimitive c scale pid m m i hi]
    change Cell.mk .real (VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource.value
      algebra O.dot (KernelOutput.shape c) scale (KernelOutput.state c pid m) i)=_
    rw [O.value_body]
    exact congrArg (Cell.mk .real) (RealScalar.output_real c _ scale pid.z pid.y i.1.val _ _ hi.1 i.1.isLt hi.2)
  · intro r off h
    rw [hm]
    exact KernelOutput.memory_frame algebra dotPrimitive c scale pid m m r off h

#axiomsClean output_correct
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelReal
