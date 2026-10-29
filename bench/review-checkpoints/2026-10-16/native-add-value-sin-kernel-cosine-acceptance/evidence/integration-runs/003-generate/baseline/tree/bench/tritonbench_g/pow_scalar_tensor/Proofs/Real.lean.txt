import bench.tritonbench_g.pow_scalar_tensor.Proofs.Transport
import bench.tritonbench_g.pow_scalar_tensor.Proofs.GridReal

namespace VeriTile.Bench.TritonBenchG.PowScalarTensor.RealExecution
open VeriTile.Triton
set_option maxHeartbeats 400000
set_option maxRecDepth 16000

/-- Successful actual one-tile source, independent exponent array, and the
complete typed-cell frame. The input and output regions may alias here. -/
theorem one_tile (output : TypedSource.FloatingOutput) (candidate : Bool)
    (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat)
    (s : BlockState) (xs : Fin B → ℝ) (hstride : 0 < sy)
    (hu : s.undef = fun _ _ => 0)
    (hin : ∀ i, taskIndex (s.pids 0) B i < N →
      s.readMem X (taskIndex (s.pids 0) B i * sx) = xs i) :
    ∃ t, exec (TypedSource.kernel output Bool.true candidate base X Y sx sy N tasks tiles B).eraseDType.toAlgKernel s = some t ∧
      (∀ i, taskIndex (s.pids 0) B i < N →
        t.readMem Y (taskIndex (s.pids 0) B i * sy) = Real.rpow base (xs i)) ∧
      (∀ r o, (r ≠ Y ∨ ∀ i, taskIndex (s.pids 0) B i < N →
        o ≠ taskIndex (s.pids 0) B i * sy) → t.mem r o = s.mem r o) := by
  rw [RealSource.erase]
  obtain ⟨t, he, hv, hf⟩ := pow_one_tile_region_run base X Y sx sy N tasks tiles B
    hstride s (fun i => xs i.1) (fun i hi => hin i.1 hi)
  refine ⟨t, (RealTransport.whole Bool.true candidate base X Y sx sy N tasks tiles B s hu).1.trans he,
    fun i hi => hv (i, PUnit.unit) hi, ?_⟩
  intro r o hc
  apply hf r o
  rcases hc with hr | hn
  · exact Or.inl hr
  · exact Or.inr fun i hi => hn i.1 hi

/-- Complete multi-iteration actual source execution against an independent
input array, with all active output values and the full raw-cell frame. -/
theorem grid (output : TypedSource.FloatingOutput) (candidate : Bool)
    (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat)
    (s : BlockState) (xs : Fin tiles → Fin B → ℝ) (hstride : 0 < sy)
    (hdisj : X ≠ Y) (hgrid : 0 < s.numPids 0) (hu : s.undef = fun _ _ => 0)
    (hin : ∀ j i, taskIndex (s.pids 0 + j.val * s.numPids 0) B i < N →
      s.readMem X (taskIndex (s.pids 0 + j.val * s.numPids 0) B i * sx) = xs j i) :
    ∃ t, exec (TypedSource.kernel output Bool.false candidate base X Y sx sy N tasks tiles B).eraseDType.toAlgKernel s = some t ∧
      (∀ j i, taskIndex (s.pids 0 + j.val * s.numPids 0) B i < N →
        t.readMem Y (taskIndex (s.pids 0 + j.val * s.numPids 0) B i * sy) = Real.rpow base (xs j i)) ∧
      (∀ r o, (r ≠ Y ∨ ∀ (j : Fin tiles) i,
        taskIndex (s.pids 0 + j.val * s.numPids 0) B i < N →
        o ≠ taskIndex (s.pids 0 + j.val * s.numPids 0) B i * sy) → t.mem r o = s.mem r o) := by
  rw [RealSource.erase]
  obtain ⟨t, he, hf⟩ := GridReal.grid_run base X Y sx sy N tasks tiles B hstride s
  refine ⟨t, (RealTransport.whole Bool.false candidate base X Y sx sy N tasks tiles B s hu).1.trans he, ?_, ?_⟩
  · intro j i hi
    rw [pow_grid_stride_exec_correct base X Y sx sy N tasks tiles B s hstride hdisj hgrid t he j.val i j.isLt hi]
    change Real.rpow base (s.readMem X _) = Real.rpow base (xs j i)
    rw [hin j i hi]
  · intro r o hc
    apply hf r o
    rcases hc with hr | hn
    · exact Or.inl hr
    · exact Or.inr fun j i hj hi => hn ⟨j, hj⟩ i hi

theorem one_safe (output : TypedSource.FloatingOutput) (candidate : Bool)
    (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat)
    (bounds : RegionBounds) (s : BlockState) (hu : s.undef = fun _ _ => 0)
    (hin : ∀ i, taskIndex (s.pids 0) B i < N → taskIndex (s.pids 0) B i * sx < bounds X)
    (hout : ∀ i, taskIndex (s.pids 0) B i < N → taskIndex (s.pids 0) B i * sy < bounds Y) :
    (TypedSource.kernel output Bool.true candidate base X Y sx sy N tasks tiles B).eraseDType.toAlgKernel.TraceSafe bounds s := by
  rw [RealSource.erase]
  apply ((RealTransport.whole Bool.true candidate base X Y sx sy N tasks tiles B s hu).2 bounds).mpr
  exact pow_one_tile_traceSafe base X Y sx sy N tasks tiles B bounds s hin hout

theorem grid_safe (output : TypedSource.FloatingOutput) (candidate : Bool)
    (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat)
    (bounds : RegionBounds) (s : BlockState) (hu : s.undef = fun _ _ => 0) (hstride : 0 < sy)
    (hin : ∀ (j : Fin tiles) i, taskIndex (s.pids 0 + j.val * s.numPids 0) B i < N →
      taskIndex (s.pids 0 + j.val * s.numPids 0) B i * sx < bounds X)
    (hout : ∀ (j : Fin tiles) i, taskIndex (s.pids 0 + j.val * s.numPids 0) B i < N →
      taskIndex (s.pids 0 + j.val * s.numPids 0) B i * sy < bounds Y) :
    (TypedSource.kernel output Bool.false candidate base X Y sx sy N tasks tiles B).eraseDType.toAlgKernel.TraceSafe bounds s := by
  rw [RealSource.erase]
  apply ((RealTransport.whole Bool.false candidate base X Y sx sy N tasks tiles B s hu).2 bounds).mpr
  exact GridReal.grid_safe base X Y sx sy N tasks tiles B hstride bounds s hin hout

theorem flattenOk (output : TypedSource.FloatingOutput) (oneTile candidate : Bool)
    (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat) :
    (TypedSource.kernel output oneTile candidate base X Y sx sy N tasks tiles B).eraseDType.toAlgKernel.FlattenOk := by
  rw [RealSource.erase]
  unfold Kernel.FlattenOk
  rw [RealSource.body_eq]
  cases oneTile <;> cases candidate <;>
    simp [RealSource.body, RealSource.perTile, RealSource.loop, RealSource.numTiles,
      RealSource.offset, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.PowScalarTensor.RealExecution
