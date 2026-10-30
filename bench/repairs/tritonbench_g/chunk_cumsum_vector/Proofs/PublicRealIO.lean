import bench.repairs.tritonbench_g.chunk_cumsum_vector.IO
import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.RealAccessSource
import bench.repairs.tritonbench_g.chunk_cumsum_vector.Contract
import VeriTile.Triton.Memory.RegionalScheduledKernelIO

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.PublicRealIO
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

def pair (h : Call) (time : Fin h.T) (j : Fin BS) : Fin (h.T*BS) :=
  ⟨time.val*BS+j.val, by have ht:=time.isLt; have hj:=j.isLt; dsimp [BS] at *; omega⟩

theorem pair_div (h : Call) (time : Fin h.T) (j : Fin BS) :
    (pair h time j).val/BS=time.val := by
  have hj := j.isLt
  dsimp [pair, BS] at *
  omega
theorem pair_mod (h : Call) (time : Fin h.T) (j : Fin BS) :
    (pair h time j).val%BS=j.val := by
  simp [pair, Nat.add_mod, Nat.mod_eq_of_lt j.isLt]

def timeIndex (h : Call) (j : Fin (h.T*BS)) : Fin h.T :=
  ⟨j.val/BS, (Nat.div_lt_iff_lt_mul (by decide : 0<BS)).mpr j.isLt⟩
def columnIndex (h : Call) (j : Fin (h.T*BS)) : Fin BS :=
  ⟨j.val%BS, Nat.mod_lt _ (by decide)⟩

def formula (h : Call) (xs : Fin (h.T*BS) → ℝ) (j : Fin (h.T*BS)) : ℝ :=
  mathematical (fun k => if inside : k<h.T then xs (pair h ⟨k,inside⟩ (columnIndex h j)) else 0)
    h.T (j.val/BS)

theorem implements (h : Call) (layout : Geometry.OutputLayout h) (c : Config)
    (chosen : c∈configurations) (p : Geometry.Program h) :
    (io h c p).RegionalScheduledImplements (fun _ => [launch h p])
      (fun xs _ j => formula h (xs (⟨0,by decide⟩ : Fin 1)) j) := by
  apply UKernelIO.RegionalScheduledImplements.intro
  · intro s xs binding bounds bounded
    rw [algorithmScheduleSafe_singleton]
    apply RealAccess.source_safe bounds h layout c (Kernels.configurations_positive c chosen).1 p
    · intro time j ht hj
      have input := bounded.1 (⟨0,by decide⟩ : Fin 1) (pair h ⟨time,ht⟩ j)
      have output := bounded.2.1 (⟨0,by decide⟩ : Fin 1) (pair h ⟨time,ht⟩ j)
      simp only [io, pair_div, pair_mod] at input output
      exact ⟨input hj,output hj⟩
    · simp [launch, AlgorithmLaunch.initial]
    · simp [launch, AlgorithmLaunch.initial]
  · intro s xs binding
    obtain ⟨final,ran,values,frame⟩ := RealExecution.source_executes h c chosen p layout
      ((launch h p).initial s) (by simp [launch,AlgorithmLaunch.initial])
      (by simp [launch,AlgorithmLaunch.initial])
    refine ⟨final,?_,?_,?_⟩
    · simpa only [runAlgorithmSchedule_singleton, AlgorithmLaunch.run] using ran
    · intro o j active
      have value := values (timeIndex h j).val (columnIndex h j)
        ⟨(timeIndex h j).isLt,active⟩
      change final.readMem h.output
        (h.outputStrides.address h.H p.row (j.val/BS) (p.feature*BS+j.val%BS)) = _
      dsimp only [RealLoop.address, timeIndex, columnIndex] at value
      rw [value]
      unfold RealData.expected formula mathematical
      apply Finset.sum_congr rfl
      intro k hk
      have kt := Finset.mem_range.mp (Finset.mem_filter.mp hk).1
      have input := binding (⟨0,by decide⟩ : Fin 1) (pair h ⟨k,kt⟩ (columnIndex h j))
      simp only [io, pair_div, pair_mod] at input
      simpa only [AlgorithmLaunch.initial_mem, dif_pos kt] using input active
    · intro r offset outside _
      symm
      apply frame r offset
      rintro ⟨same,time,j,valid,whereAt⟩
      have miss := outside (⟨0,by decide⟩ : Fin 1) (pair h ⟨time,valid.1⟩ j)
      simp only [io,pair_div,pair_mod] at miss
      rcases miss valid.2 with no | no
      · exact no same
      · exact no whereAt

#specNonCircular formula avoiding [Kernels.source,Kernels.sumLeft,exec]
#axiomsClean implements
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.PublicRealIO
