import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelRealModel
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelBackwardProgram
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelReal
open VeriTile Triton FP.Structural
open KernelBackward (PID Carry)
open VeriTile.Bench.Repairs.ChunkLinearAttn (hActive)
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

theorem factor_eq (c : Shape) (pid : PID) (n : Nat) :
    KernelBackward.factor algebra (parameters c) pid n=
      decay realOps c pid.z (c.length (c.chunks-1-n)) := by
  unfold KernelBackward.factor
  rw [logRate_eq]
  simp [KernelBackward.L.mul,algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra,
    parameters,KernelBackward.time,decay,realOps,Shape.length]

theorem initial_eq (c : Shape) (pid : PID) (m : Memory ℝ) (seed terminal : Bool)
    (i : TileIndex [c.BK,c.BV]) (hi : hActive (parameters c) pid i) :
    KernelBackward.initialValue algebra (parameters c) pid m terminal i=
      Reference.reverseCarry realOps c (inputs c m seed terminal) 1 pid.z 0
        (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val) := by
  rcases hi with ⟨hk,hv⟩
  cases terminal <;> simp_all [KernelBackward.initialValue,Reference.reverseCarry,inputs,
    parameters,masked,realOps,algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra,
    VeriTile.Bench.Repairs.ChunkLinearAttn.zero]

theorem next_eq (c : Shape) (pid : PID) (m : Memory ℝ) (scale : ℝ) (seed terminal : Bool)
    (n : Nat) (carry : Carry (parameters c) ℝ) (target : Tile ℝ)
    (i : TileIndex [c.BK,c.BV]) (hi : hActive (parameters c) pid i)
    (hc : carry i=target (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val)) :
    KernelBackward.next algebra dotPrimitive (parameters c) pid m scale n carry i=
      Body.nextDH realOps c (inputs c m seed terminal) scale pid.z (c.chunks-1-n) target
        (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val) := by
  rcases hi with ⟨hk,hv⟩
  unfold KernelBackward.next
  rw [factor_eq]
  change decay realOps c pid.z (c.length (c.chunks-1-n))*carry i+
    (∑ t : Fin 64,KernelBackward.leftValues algebra (parameters c) pid m scale n (i.1,t,())*
      KernelBackward.rightValues algebra (parameters c) pid m n (t,i.2.1,()))=_
  rw [hc]
  unfold Body.nextDH
  rw [RealScalar.chain_sum]
  simp only [RealScalar.real_mul]
  congr 1
  rw [← Fin.sum_univ_eq_sum_range]
  apply Finset.sum_congr rfl
  intro t _
  simp only [KernelBackward.leftValues,KernelBackward.rightValues,weight_eq]
  simp [KernelBackward.L.load,KernelBackward.L.rawLoad,KernelBackward.L.cast,
    KernelBackward.L.mul,KernelBackward.L.scalar,KernelBackward.L.zero,
    algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra,parameters,
    KernelBackward.time,qAt,gAt,masked,inputs,realOps,hk,hv,Nat.add_assoc,
    Nat.add_comm,Nat.add_left_comm,and_comm]

theorem values_reference (c : Shape) (pid : PID) (m : Memory ℝ) (scale : ℝ)
    (seed terminal : Bool) (n : Nat) (i : TileIndex [c.BK,c.BV])
    (hi : hActive (parameters c) pid i) :
    KernelBackward.values algebra dotPrimitive (parameters c) pid m scale terminal n i=
      Reference.reverseCarry realOps c (inputs c m seed terminal) scale pid.z n
        (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val) := by
  induction n with
  | zero => exact initial_eq c pid m seed terminal i hi
  | succ n ih =>
    change KernelBackward.next algebra dotPrimitive (parameters c) pid m scale n
      (KernelBackward.values algebra dotPrimitive (parameters c) pid m scale terminal n) i=_
    rw [next_eq c pid m scale seed terminal n _ _ i hi ih]
    rfl

theorem values_math (c : Shape) (pid : PID) (m : Memory ℝ) (scale : ℝ)
    (seed terminal : Bool) (n : Nat) (hn : n≤c.chunks) (i : TileIndex [c.BK,c.BV])
    (hi : hActive (parameters c) pid i) :
    KernelBackward.values algebra dotPrimitive (parameters c) pid m scale terminal n i=
      MathTarget.future c (inputs c m seed terminal) scale pid.z (c.boundary (c.chunks-n))
        (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val) := by
  rw [values_reference c pid m scale seed terminal n i hi]
  have hk : pid.x*c.BK+i.1.val<c.K:=hi.1
  have hv : pid.y*c.BV+i.2.1.val<c.V:=hi.2
  rw [RealScalar.reverse_scalar c (inputs c m seed terminal) scale pid.z n _ _ hn hk hv]
  exact ScalarSource.reverse_carry_correct c (inputs c m seed terminal) scale pid.z n hn _ _

#axiomsClean next_eq
#axiomsClean values_reference
#axiomsClean values_math
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelReal
