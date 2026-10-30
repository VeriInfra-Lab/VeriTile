import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.IOFormulaCongruence
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.IOReference
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelForwardCorrect
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelGradientCorrect

namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PublicIO
open VeriTile Triton FP.Structural KernelPacket
open KernelReal (parameters)
noncomputable section
set_option maxHeartbeats 1200000

/-- Existing source-to-scalar results, now uniformly indexed by the complete
public output channels. No Tensor or host-runtime theorem is imported. -/
theorem real_scalar_values (a : Call) (m : KernelPacket.Memory ℝ)
    (port : Fin (outputs a.shape).length) (j : TileIndex (outputs a.shape)[port].shape)
    (active : outputActive a port j) :
    sourceValues KernelReal.algebra KernelReal.dotPrimitive a m port j=
      scalarOutputs realOps a.scale a (readInputs 0 a (fun r o=>(m r o).read .real))
        (readHistory a (fun r o=>(m r o).read .real) Bool.false)
        (readHistory a (fun r o=>(m r o).read .real) Bool.true) port j := by
  rcases port with ⟨port,hp⟩
  have hport : port<8 := hp
  interval_cases port
  · exact KernelReal.forward_values_reference a.shape a.pid m a.inputState a.inputState
      j.1.val (Nat.le_of_lt j.1.isLt) j.2 active.2
  · exact KernelReal.forward_values_reference a.shape a.pid m a.inputState a.inputState
      a.shape.chunks le_rfl j active.2.2
  · exact KernelReal.O.value_body a.shape (KernelOutput.state a.shape a.pid m) a.scale j
  · exact KernelReal.values_reference a.shape a.pid m a.scale a.inputState a.inputState j.1.val j.2 active.2
  · exact KernelReal.values_reference a.shape a.pid m a.scale a.inputState a.inputState a.shape.chunks j active.2.2
  · exact KernelReal.query_result_eq a.shape a.pid m a.scale a.inputState a.inputState j
  · exact KernelReal.key_result_eq a.shape a.pid m a.scale a.inputState a.inputState j
  · exact KernelReal.value_result_eq a.shape a.pid m a.scale a.inputState a.inputState j.1.val j.2

/-- Reuse the already-proved scalar-to-dense algebra. The chunk premise is
needed only for the original QKV dense adjoint formulas. -/
theorem scalar_mathematical (a : Call) (xs : Inputs ℝ a.shape)
    (chunk : a.stage=.backwardGradient → a.pid.y<a.shape.chunks)
    (port : Fin (outputs a.shape).length) (j : TileIndex (outputs a.shape)[port].shape)
    (active : outputActive a port j) :
    scalarFormula realOps a.scale a xs port j=mathematical a xs port j := by
  rcases port with ⟨port,hp⟩
  have hport : port<8 := hp
  interval_cases port
  · change Reference.history realOps _ _ _ _ _ _=_
    rw [RealScalar.history_scalar _ _ _ _ _ _ (Nat.le_of_lt j.1.isLt) active.2.1 active.2.2,
      ScalarSource.forward_carry_correct _ _ _ _ _ (Nat.le_of_lt j.1.isLt)]
    rfl
  · change Reference.history realOps _ _ _ _ _ _=_
    rw [RealScalar.history_scalar _ _ _ _ _ _ le_rfl active.2.2.1 active.2.2.2,
      ScalarSource.forward_carry_correct _ _ _ _ _ le_rfl]
    simp only [mathematical,mathOutputs,MathTarget.finalState,a.shape.boundary_final]
    rfl
  · exact RealScalar.output_real _ _ _ _ _ _ _ _ active.2.1 j.1.isLt active.2.2
  · change Reference.reverseCarry realOps _ _ _ _ _ _ _=_
    rw [RealScalar.reverse_scalar _ _ _ _ _ _ _ (Nat.le_of_lt j.1.isLt) active.2.1 active.2.2,
      ScalarSource.reverse_carry_correct _ _ _ _ _ (Nat.le_of_lt j.1.isLt)]
    rfl
  · change Reference.reverseCarry realOps _ _ _ _ _ _ _=_
    rw [RealScalar.reverse_scalar _ _ _ _ _ _ _ le_rfl active.2.2.1 active.2.2.2,
      ScalarSource.reverse_carry_correct _ _ _ _ _ le_rfl]
    simp only [mathematical,mathOutputs,MathTarget.initialGradient,Nat.sub_self,Shape.boundary,Nat.zero_mul,Nat.zero_min]
    rfl
  · exact RealScalar.query_real _ _ _ _ _ _ _ _ active.2.1 j.1.isLt active.2.2
  · have hc:=chunk active.1
    exact RealScalar.key_real _ _ _ _ _ _ _ _ hc
      ((RealScalar.time_lane a.shape a.pid.y j.1.val hc j.1.isLt).mp active.2.1) active.2.2
  · have hc:=chunk active.1
    exact RealScalar.value_real _ _ _ _ _ _ _ _ _ hc
      ((RealScalar.time_lane a.shape a.pid.y j.2.1.val hc j.2.1.isLt).mp active.2.1) active.2.2

#axiomsClean real_scalar_values
#axiomsClean scalar_mathematical
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PublicIO
