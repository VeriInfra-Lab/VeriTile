import bench.tritonbench_g.pow_scalar_tensor.Proofs.Flat

namespace VeriTile.Bench.TritonBenchG.PowScalarTensor.Contracts
open VeriTile.Triton

/-- Exact-real finite-memory IO for scalar-base power on an independent
exponent array. Safety is of the actual source trace, execution must succeed,
and the frame compares complete typed cells outside all active writes. -/
def ExactIO {α : Type} (kernel : ComputeKernel) (alloc : FlatAlloc) (s : BlockState)
    (Y : RegionName) (base : ℝ) (write : α → Nat) (active : α → Prop)
    (xs : α → ℝ) : Prop :=
  kernel.eraseDType.toAlgKernel.TraceSafe alloc.extent s ∧
  ∃ t, exec (alloc.flattenKernel kernel.eraseDType.toAlgKernel) (alloc.flattenState s) = some t ∧
    (∀ i, active i → t.readMem alloc.flat (alloc.addr Y (write i)) = Real.rpow base (xs i)) ∧
    (∀ r o, (r ≠ alloc.flat ∨ ∀ i, active i → o ≠ alloc.addr Y (write i)) →
      t.mem r o = (alloc.flattenState s).mem r o)

/-- Full actual one-tile source correctness with a dimension-general finite
allocation, partial tiles and independent natural input/output strides. -/
theorem one_tile (output : TypedSource.FloatingOutput) (candidate : Bool)
    (alloc : FlatAlloc) (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat)
    (s : BlockState) (xs : Fin B → ℝ)
    (hd : alloc.Disjoint) (hcov : ∀ r, r ∉ alloc.regions → alloc.extent r = 0)
    (hy : Y ∈ alloc.regions) (hstride : 0 < sy) (hu : s.undef = fun _ _ => 0)
    (hinput : ∀ i, taskIndex (s.pids 0) B i < N →
      s.readMem X (taskIndex (s.pids 0) B i * sx) = xs i)
    (hin : ∀ i, taskIndex (s.pids 0) B i < N → taskIndex (s.pids 0) B i * sx < alloc.extent X)
    (hout : ∀ i, taskIndex (s.pids 0) B i < N → taskIndex (s.pids 0) B i * sy < alloc.extent Y) :
    ExactIO (TypedSource.kernel output Bool.true candidate base X Y sx sy N tasks tiles B)
      alloc s Y base (fun i => taskIndex (s.pids 0) B i * sy)
      (fun i => taskIndex (s.pids 0) B i < N) xs := by
  have hsafe := RealExecution.one_safe output candidate base X Y sx sy N tasks tiles B alloc.extent s hu hin hout
  refine ⟨hsafe, ?_⟩
  apply FlatIO.transport alloc hd hcov _
    (RealExecution.flattenOk output Bool.true candidate base X Y sx sy N tasks tiles B)
    s hu hsafe Y hy _ _ _ hout
  exact RealExecution.one_tile output candidate base X Y sx sy N tasks tiles B s xs hstride hu hinput

/-- Full actual grid-stride source correctness, including all dynamic
iterations and a partial final tile, against independent exponent arrays. -/
theorem grid (output : TypedSource.FloatingOutput) (candidate : Bool)
    (alloc : FlatAlloc) (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat)
    (s : BlockState) (xs : Fin tiles → Fin B → ℝ)
    (hd : alloc.Disjoint) (hcov : ∀ r, r ∉ alloc.regions → alloc.extent r = 0)
    (hy : Y ∈ alloc.regions) (hstride : 0 < sy) (hu : s.undef = fun _ _ => 0)
    (hdisj : X ≠ Y) (hgrid : 0 < s.numPids 0)
    (hinput : ∀ j i, taskIndex (s.pids 0 + j.val * s.numPids 0) B i < N →
      s.readMem X (taskIndex (s.pids 0 + j.val * s.numPids 0) B i * sx) = xs j i)
    (hin : ∀ (j : Fin tiles) i, taskIndex (s.pids 0 + j.val * s.numPids 0) B i < N →
      taskIndex (s.pids 0 + j.val * s.numPids 0) B i * sx < alloc.extent X)
    (hout : ∀ (j : Fin tiles) i, taskIndex (s.pids 0 + j.val * s.numPids 0) B i < N →
      taskIndex (s.pids 0 + j.val * s.numPids 0) B i * sy < alloc.extent Y) :
    ExactIO (TypedSource.kernel output Bool.false candidate base X Y sx sy N tasks tiles B)
      alloc s Y base
      (fun p : Fin tiles × Fin B => taskIndex (s.pids 0 + p.1.val * s.numPids 0) B p.2 * sy)
      (fun p => taskIndex (s.pids 0 + p.1.val * s.numPids 0) B p.2 < N)
      (fun p => xs p.1 p.2) := by
  have hsafe := RealExecution.grid_safe output candidate base X Y sx sy N tasks tiles B alloc.extent s hu hstride hin hout
  refine ⟨hsafe, ?_⟩
  apply FlatIO.transport alloc hd hcov _
    (RealExecution.flattenOk output Bool.false candidate base X Y sx sy N tasks tiles B)
    s hu hsafe Y hy _ _ _ (fun p hp => hout p.1 p.2 hp)
  obtain ⟨t, he, hv, hf⟩ := RealExecution.grid output candidate base X Y sx sy N tasks tiles B s xs hstride hdisj hgrid hu hinput
  exact ⟨t, he, fun p hp => hv p.1 p.2 hp,
    fun r o hc => hf r o (hc.elim Or.inl (fun hn => Or.inr (fun j i hi => hn (j,i) hi)))⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.PowScalarTensor.Contracts
