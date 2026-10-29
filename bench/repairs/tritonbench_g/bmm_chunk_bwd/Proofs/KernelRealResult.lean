import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.HostContract
import bench.repairs.tritonbench_g.bmm_chunk_bwd.Kernels

namespace VeriTile.Bench.Repairs.BmmChunkBwd.KernelRealResult
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
noncomputable section

/-- Complete raw flattened source grid with the independent Doutᵀ @ A plus
optional residual result at every active output, and the full physical frame. -/
theorem bmm_chunk_bwd_real (h : HostCall) (choice : Fin 9)
    (ha : h.aStorage = .fp32) (hd : h.doutStorage = .fp32)
    (alloc : FlatAlloc) (layout : HostContract.Layout h (tuning choice) alloc)
    (initial : BlockState) :
    ∃ final,
      HostGrid.run (alloc.flattenKernel (Kernels.fp32Source h (tuning choice)).eraseDType.toAlgKernel)
        (HostContract.grid h (tuning choice)) (HostGrid.programs (HostContract.grid h (tuning choice)))
        (alloc.flattenState initial) = some final ∧
      (∀ q ∈ HostGrid.programs (HostContract.grid h (tuning choice)), ∀ i,
        HostGrid.active (tuning choice).BM (tuning choice).BN (HostContract.parameters h) q i →
        final.readMem alloc.flat (alloc.addr h.Out
          (HostGrid.address (tuning choice).BM (tuning choice).BN (HostContract.parameters h) q i)) =
          HostGrid.expected (tuning choice).BM (tuning choice).BN (HostContract.parameters h) initial q i) ∧
      (∀ r off, (r ≠ alloc.flat ∨ ∀ q ∈ HostGrid.programs (HostContract.grid h (tuning choice)), ∀ i,
        HostGrid.active (tuning choice).BM (tuning choice).BN (HostContract.parameters h) q i →
        off ≠ alloc.addr h.Out
          (HostGrid.address (tuning choice).BM (tuning choice).BN (HostContract.parameters h) q i)) →
        final.mem r off = (alloc.flattenState initial).mem r off) := by
  obtain ⟨final, executed, values, frame⟩ := HostContract.complete h choice Bool.false alloc layout initial
  refine ⟨final, ?_, values, frame⟩
  rw [Kernels.fp32Source_eq h (tuning choice) ha hd]
  unfold HostContract.run at executed
  cases he : HostGrid.run (alloc.flattenKernel (h.source (tuning choice) Bool.false).eraseDType.toAlgKernel)
      (HostContract.grid h (tuning choice)) (HostGrid.programs (HostContract.grid h (tuning choice)))
      (alloc.flattenState initial) with
  | none => simp [he] at executed
  | some result => simpa [he] using congrArg (Option.map Prod.snd) executed

#axiomsClean bmm_chunk_bwd_real
#specNonCircular HostGrid.expected avoiding [HostGrid.run, Kernels.fp32Source]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkBwd.KernelRealResult
