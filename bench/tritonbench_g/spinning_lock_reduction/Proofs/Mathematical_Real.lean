import bench.tritonbench_g.spinning_lock_reduction.Proofs.Mathematical_Fresh
import Mathlib.Data.Real.Basic

noncomputable section
namespace VeriTile.Bench.TritonBenchG.SpinningLock

/-- Real projection of the source's zero initialization and additions. -/
def realBackend : Backend ℝ := ⟨0, (· + ·)⟩

theorem fresh_real (h : HostArgs) (buffers : HostBuffers h) (schedule : List Nat)
    (mem : Memory ℝ) (fresh : ∀ a, mem.locks a = 0) :
    ∃ result, FreshFacts realBackend h buffers schedule mem result :=
  fresh_source_backend realBackend (by simp [realBackend]) h buffers schedule mem fresh

#axiomsClean fresh_real
end VeriTile.Bench.TritonBenchG.SpinningLock
