import bench.tritonbench_g.pow_scalar_tensor.Proofs.IntegerReal
import bench.tritonbench_g.pow_scalar_tensor.Proofs.Flat

namespace VeriTile.Bench.TritonBenchG.PowScalarTensor.IntegerContracts
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

/-- Integer-power finite-memory IO: actual source trace safety, successful
flattened execution, independent zpow of a typed integer input array, and full
raw-cell framing. Real output casts are erased; exponent types are retained. -/
def ExactIO {α : Type} (kernel : ComputeKernel) (alloc : FlatAlloc) (s : BlockState)
    (Y : RegionName) (base : ℝ) (write : α → Nat) (active : α → Prop)
    (xs : α → Int) : Prop :=
  kernel.eraseDType.toAlgKernel.TraceSafe alloc.extent s ∧
  ∃ t, exec (alloc.flattenKernel kernel.eraseDType.toAlgKernel) (alloc.flattenState s) = some t ∧
    (∀ i, active i → t.readMem alloc.flat (alloc.addr Y (write i)) = base ^ xs i) ∧
    (∀ r o, (r ≠ alloc.flat ∨ ∀ i, active i → o ≠ alloc.addr Y (write i)) →
      t.mem r o = (alloc.flattenState s).mem r o)

/-- All six destination profiles and both variants satisfy the same complete
single-tile int32 contract for arbitrary shape, strides and scalar base. -/
theorem one_tile (output : TypedSource.FloatingOutput) (candidate : Bool)
    (alloc : FlatAlloc) (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat)
    (s : BlockState) (xs : Fin B → Int)
    (hd : alloc.Disjoint) (hcov : ∀ r, r ∉ alloc.regions → alloc.extent r = 0)
    (hy : Y ∈ alloc.regions) (hstride : 0 < sy) (hu : s.undef = fun _ _ => 0)
    (hinput : ∀ i, taskIndex (s.pids 0) B i < N →
      s.readMemTyped .int X (taskIndex (s.pids 0) B i * sx) = some (xs i))
    (hvalid : ∀ i, taskIndex (s.pids 0) B i < N → Int32Bits.inRange (xs i) = Bool.true)
    (hin : ∀ i, taskIndex (s.pids 0) B i < N → taskIndex (s.pids 0) B i * sx < alloc.extent X)
    (hout : ∀ i, taskIndex (s.pids 0) B i < N → taskIndex (s.pids 0) B i * sy < alloc.extent Y) :
    ExactIO (TypedIntegerSource.kernel output Bool.true candidate base X Y sx sy N tasks tiles B)
      alloc s Y base (fun i => taskIndex (s.pids 0) B i * sy)
      (fun i => taskIndex (s.pids 0) B i < N) xs := by
  have hread : ∀ i, taskIndex (s.pids 0) B i < N →
      s.readMemValue .int X (taskIndex (s.pids 0) B i * sx) = xs i := by
    intro i hi
    simp only [BlockState.readMemValue, hinput i hi]
  have hrange : ∀ i, taskIndex (s.pids 0) B i < N →
      Int32Bits.inRange (s.readMemValue .int X (taskIndex (s.pids 0) B i * sx)) = Bool.true := by
    intro i hi
    rw [hread i hi]
    exact hvalid i hi
  obtain ⟨hs,t,he,hv,hf⟩ := IntegerReal.one_tile candidate base X Y sx sy N tasks tiles B
    alloc.extent s hstride hu hrange hin hout
  unfold ExactIO
  rw [IntegerSource.erase]
  refine ⟨hs, ?_⟩
  apply FlatIO.transport alloc hd hcov _
    (IntegerReal.flattenOk Bool.true candidate base X Y sx sy N tasks tiles B)
    s hu hs Y hy _ _ _ hout
  refine ⟨t,he,?_,hf⟩
  intro i hi
  simpa only [IntegerSteps.powSpec, hread i hi] using hv i hi

/-- Complete dynamic int32 grid execution for every trip count and shape.
Range obligations cover active typed inputs; zero undef supplies valid integer
zero on every computed masked lane, as established by the load proof. -/
theorem grid (output : TypedSource.FloatingOutput) (candidate : Bool)
    (alloc : FlatAlloc) (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat)
    (s : BlockState) (xs : Fin tiles → Fin B → Int)
    (hd : alloc.Disjoint) (hcov : ∀ r, r ∉ alloc.regions → alloc.extent r = 0)
    (hy : Y ∈ alloc.regions) (hstride : 0 < sy) (hu : s.undef = fun _ _ => 0)
    (hdisj : X ≠ Y) (hgrid : 0 < s.numPids 0)
    (hinput : ∀ j i, taskIndex (s.pids 0+j.val*s.numPids 0) B i < N →
      s.readMemTyped .int X (taskIndex (s.pids 0+j.val*s.numPids 0) B i * sx) = some (xs j i))
    (hvalid : ∀ j i, taskIndex (s.pids 0+j.val*s.numPids 0) B i < N →
      Int32Bits.inRange (xs j i) = Bool.true)
    (hin : ∀ (j : Fin tiles) i, taskIndex (s.pids 0+j.val*s.numPids 0) B i < N →
      taskIndex (s.pids 0+j.val*s.numPids 0) B i * sx < alloc.extent X)
    (hout : ∀ (j : Fin tiles) i, taskIndex (s.pids 0+j.val*s.numPids 0) B i < N →
      taskIndex (s.pids 0+j.val*s.numPids 0) B i * sy < alloc.extent Y) :
    ExactIO (TypedIntegerSource.kernel output Bool.false candidate base X Y sx sy N tasks tiles B)
      alloc s Y base
      (fun p : Fin tiles × Fin B => taskIndex (s.pids 0+p.1.val*s.numPids 0) B p.2 * sy)
      (fun p => taskIndex (s.pids 0+p.1.val*s.numPids 0) B p.2 < N)
      (fun p => xs p.1 p.2) := by
  have hread : ∀ j i, taskIndex (s.pids 0+j.val*s.numPids 0) B i < N →
      s.readMemValue .int X (taskIndex (s.pids 0+j.val*s.numPids 0) B i * sx) = xs j i := by
    intro j i hi
    simp only [BlockState.readMemValue, hinput j i hi]
  have hrange : ∀ (j : Fin tiles) i, taskIndex (s.pids 0+j.val*s.numPids 0) B i < N →
      Int32Bits.inRange (s.readMemValue .int X (taskIndex (s.pids 0+j.val*s.numPids 0) B i * sx)) = Bool.true := by
    intro j i hi
    rw [hread j i hi]
    exact hvalid j i hi
  obtain ⟨hs,t,he,hv,hf⟩ := IntegerReal.grid candidate base X Y sx sy N tasks tiles B
    alloc.extent s hstride hdisj hgrid hu hrange hin hout
  unfold ExactIO
  rw [IntegerSource.erase]
  refine ⟨hs, ?_⟩
  apply FlatIO.transport alloc hd hcov _
    (IntegerReal.flattenOk Bool.false candidate base X Y sx sy N tasks tiles B)
    s hu hs Y hy _ _ _ (fun p hp => hout p.1 p.2 hp)
  refine ⟨t,he,?_,hf⟩
  intro p hp
  simpa only [IntegerSteps.powSpec, hread p.1 p.2 hp] using hv p hp

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.PowScalarTensor.IntegerContracts
