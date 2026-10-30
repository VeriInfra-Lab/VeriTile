import bench.tritonbench_g.chunk_cumsum_kernel.IO
import bench.tritonbench_g.chunk_cumsum_kernel.Proofs.RealSafety
import bench.tritonbench_g.chunk_cumsum_kernel.Contract
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.PublicReal
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 12000

/-- Independent inclusive prefix over the initial logical row. -/
def formula {T : Nat} (xs : Fin T → ℝ) (i : Fin T) : ℝ :=
  mathematical (fun j => if h : j<T then xs ⟨j,h⟩ else 0) T i.val

theorem implements (input output : RegionName) (T : Nat) (cfg : Config)
    (separate : output ≠ input) :
    (io input output T cfg).RegionalImplements (fun _ _ vals _ => formula (vals 0)) := by
  apply UKernelIO.RegionalImplements3D.intro
  · intro bounds s vals _ hi ho _
    apply RealSafety.trace_safe input output T cfg.block bounds s
    · intro j hj
      exact hi 0 ⟨j,hj⟩ trivial
    · intro j hj
      exact ho 0 ⟨j,hj⟩ trivial
  · intro s vals hin
    obtain ⟨t,ran,values,frame⟩ := RealIndexBridge.complete input output T cfg s separate
    refine ⟨t,ran,?_,?_⟩
    · intro o j _
      fin_cases o
      change t.readMem output (s.pids 0*T+j.val) = formula (vals 0) j
      rw [values j.val j.isLt]
      unfold formula mathematical
      apply Finset.sum_congr rfl
      intro k hk
      have hkt := Finset.mem_range.mp (Finset.mem_filter.mp hk).1
      change s.readMem input (s.pids 0*T+k) = (if h : k<T then vals 0 ⟨k,h⟩ else 0)
      rw [dif_pos hkt]
      exact hin 0 ⟨k,hkt⟩ trivial
    · intro r address outside _
      apply frame
      by_cases hr : r=output
      · right
        intro j hj
        exact (outside 0 ⟨j,hj⟩ trivial).resolve_left (fun h => h hr)
      · exact Or.inl hr

#axiomsClean implements
end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.PublicReal
