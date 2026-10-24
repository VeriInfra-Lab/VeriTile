import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelForwardMemory
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelBackwardReal
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelReal
open VeriTile Triton FP.Structural
open KernelBackward (PID)
open VeriTile.Bench.Repairs.ChunkLinearAttn (hActive)
open scoped BigOperators
namespace F
export VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardHSource
  (initValue update keyLoad valueLoad gatedValue blockDecay timeDecay actualLength feature column active stateAddress)
end F
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

theorem shared_rate_eq (c : Shape) (pid : PID) :
    VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelModel.rate algebra (KernelOutput.shape c) pid.z=
      logRate realOps c pid.z := logRate_eq c pid

theorem forward_block_decay (c : Shape) (pid : PID) (m : Memory ℝ) (n : Nat) :
    F.blockDecay algebra (KernelOutput.shape c) (KernelForward.state c pid m) n=
      decay realOps c pid.z n := by
  unfold F.blockDecay
  change algebra.unary (some .fp32) .exp2
    (algebra.binary (some .fp32) .real .mul
      (if n=64 then algebra.literal (some .fp32) .real 64 else algebra.fromNat (some .fp32) n)
      (VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelModel.rate algebra (KernelOutput.shape c) pid.z))=_
  rw [shared_rate_eq]
  split_ifs with h <;> simp [algebra,decay,realOps,h]

theorem forward_time_decay (c : Shape) (pid : PID) (m : Memory ℝ) (n : Nat) (i : TileIndex [64]) :
    F.timeDecay algebra (KernelOutput.shape c) (KernelForward.state c pid m) n i=
      decay realOps c pid.z ((n:Int)-i.1.val-1) := by
  unfold F.timeDecay VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelModel.decay
  change algebra.unary (some .fp32) .exp2
    (algebra.binary (some .fp32) .real .mul (algebra.fromInt (some .fp32) ((n:Int)-i.1.val-1))
      (VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelModel.rate algebra (KernelOutput.shape c) pid.z))=_
  rw [shared_rate_eq]
  rfl

theorem forward_initial_eq (c : Shape) (pid : PID) (m : Memory ℝ) (seed terminal : Bool)
    (i : TileIndex [c.BK,c.BV]) :
    F.initValue algebra (KernelOutput.shape c) seed (KernelForward.state c pid m) i=
      Reference.history realOps c (inputs c m seed terminal) pid.z 0
        (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val) := by
  cases seed <;> simp [F.initValue,F.active,F.stateAddress,F.feature,F.column,Reference.history,
    inputs,masked,realOps,algebra,KernelForward.state,KernelOutput.shape,parameters,
    VeriTile.Bench.Repairs.ChunkLinearAttn.launchState,
    VeriTile.Bench.Repairs.ChunkDeltaFwd.memoryState,
    VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelModel.zero]

theorem forward_next_eq (c : Shape) (pid : PID) (m : Memory ℝ) (seed terminal : Bool)
    (n : Nat) (hn : n<c.chunks) (carry : TileIndex [c.BK,c.BV]→ℝ) (target : Tile ℝ)
    (i : TileIndex [c.BK,c.BV]) (hi : hActive (parameters c) pid i)
    (hc : carry i=target (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val)) :
    F.update algebra (KernelOutput.sharedDot dotPrimitive) (KernelOutput.shape c)
        (KernelForward.state c pid m) n carry i=
      Body.nextH realOps c (inputs c m seed terminal) pid.z n target
        (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val) := by
  have length : F.actualLength (KernelOutput.shape c) n=c.length n := (c.length_source n hn).symm
  unfold F.update
  rw [forward_block_decay,length]
  change decay realOps c pid.z (c.length n)*carry i+
    (∑ t : Fin 64,F.keyLoad algebra (KernelOutput.shape c) (KernelForward.state c pid m) n (i.1,t,())*
      F.gatedValue algebra (KernelOutput.shape c) (KernelForward.state c pid m) n (t,i.2.1,()))=_
  rw [hc]
  unfold Body.nextH
  rw [RealScalar.chain_sum]
  simp only [RealScalar.real_mul]
  congr 1
  rw [← Fin.sum_univ_eq_sum_range]
  apply Finset.sum_congr rfl
  intro t _
  simp only [F.gatedValue,forward_time_decay,length]
  simp [F.keyLoad,F.valueLoad,F.feature,F.column,KernelForward.state,KernelOutput.shape,parameters,
    VeriTile.Bench.Repairs.ChunkLinearAttn.launchState,
    VeriTile.Bench.Repairs.ChunkDeltaFwd.memoryState,
    VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelModel.zero,
    algebra,kAt,vAt,masked,inputs,realOps,and_comm,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]

theorem forward_values_reference (c : Shape) (pid : PID) (m : Memory ℝ) (seed terminal : Bool)
    (n : Nat) (hn : n≤c.chunks) (i : TileIndex [c.BK,c.BV]) (hi : hActive (parameters c) pid i) :
    KernelForward.values algebra dotPrimitive c pid m seed n i=
      Reference.history realOps c (inputs c m seed terminal) pid.z n
        (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val) := by
  induction n with
  | zero=>exact forward_initial_eq c pid m seed terminal i
  | succ n ih=>
    change F.update algebra (KernelOutput.sharedDot dotPrimitive) (KernelOutput.shape c)
      (KernelForward.state c pid m) n (KernelForward.values algebra dotPrimitive c pid m seed n) i=_
    rw [forward_next_eq c pid m seed terminal n (by omega) _ _ i hi (ih (by omega))]
    rfl

theorem forward_values_math (c : Shape) (pid : PID) (m : Memory ℝ) (seed terminal : Bool)
    (n : Nat) (hn : n≤c.chunks) (i : TileIndex [c.BK,c.BV]) (hi : hActive (parameters c) pid i) :
    KernelForward.values algebra dotPrimitive c pid m seed n i=
      MathTarget.state c (inputs c m seed terminal) Bool.true pid.z (c.boundary n)
        (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val) := by
  rw [forward_values_reference c pid m seed terminal n hn i hi]
  have hk : pid.x*c.BK+i.1.val<c.K:=hi.1
  have hv : pid.y*c.BV+i.2.1.val<c.V:=hi.2
  rw [RealScalar.history_scalar c (inputs c m seed terminal) pid.z n _ _ hn hk hv]
  exact ScalarSource.forward_carry_correct c (inputs c m seed terminal) Bool.true pid.z n hn _ _

#axiomsClean forward_next_eq
#axiomsClean forward_values_reference
#axiomsClean forward_values_math
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelReal
