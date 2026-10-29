import bench.repairs.tritonbench_g.triton_matmul.Proofs.RawMemory

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
open VeriTile.Triton FP.Structural FPValue
noncomputable section

/-- Natural-coordinate interpretation is derived only for the launched group.
The public launch-order theorem supplies this ordinary integer bound. -/
theorem control_coordinates {α : Type} (p : Parameters) (s : State α)
    (hm : signedCeil p.M (BM p) = (blocksM p : Int))
    (hn : signedCeil p.N (BN p) = (blocksN p : Int))
    (hf : firstM p s ≤ blocksM p) :
    (control p s).pidM = (pidM p s : Int) ∧
    (control p s).pidN = (pidN p s : Int) := by
  have hwidth : (p.GROUP_SIZE_M : Int) * (blocksN p : Int) = (perGroup p : Int) := by
    simp [perGroup]
  have hfirst : (s.pids 0 : Int).tdiv (perGroup p : Int) * (p.GROUP_SIZE_M : Int) =
      (firstM p s : Int) := by
    simp [firstM]
  have hsize : min ((blocksM p : Int) - (firstM p s : Int)) (p.GROUP_SIZE_M : Int) =
      (groupSize p s : Int) := by
    rw [← Int.natCast_sub hf]
    simp [groupSize]
  simp only [control,hm,hn,hwidth,hfirst,hsize,Id.run_pure]
  simp only [pidM,pidN,← Int.ofNat_tmod,← Int.ofNat_tdiv,Int.natCast_add]
  trivial

theorem maskM_model {α : Type} (p : Parameters) (s : State α)
    (h : (control p s).pidM = (pidM p s : Int)) (i : Fin (BM p)) :
    maskM p s i = decide (row p s i < p.M) := by
  have hs : (control p s).startM = (pidM p s : Int) * (BM p : Int) := by
    change (control p s).pidM * (BM p : Int) = _
    rw [h]
  have hi : ((i.val : Int) < (p.M : Int) - (pidM p s : Int) * (BM p : Int)) ↔
      row p s i < p.M := by
    simp only [row,← Int.natCast_mul]
    omega
  simp only [maskM,hs,hi]

theorem maskN_model {α : Type} (p : Parameters) (s : State α)
    (h : (control p s).pidN = (pidN p s : Int)) (i : Fin (BN p)) :
    maskN p s i = decide (col p s i < p.N) := by
  have hs : (control p s).startN = (pidN p s : Int) * (BN p : Int) := by
    change (control p s).pidN * (BN p : Int) = _
    rw [h]
  have hi : ((i.val : Int) < (p.N : Int) - (pidN p s : Int) * (BN p : Int)) ↔
      col p s i < p.N := by
    simp only [col,← Int.natCast_mul]
    omega
  simp only [maskN,hs,hi]

theorem offsetM_model {α : Type} (p : Parameters) (s : State α)
    (h : (control p s).pidM = (pidM p s : Int)) (i : Fin (BM p)) :
    offsetM p s i = (aRow p s i : Int) := by
  have hs : (control p s).startM = (pidM p s : Int) * (BM p : Int) := by
    change (control p s).pidM * (BM p : Int) = _
    rw [h]
  by_cases hi : pidM p s * BM p + i.val < p.M <;>
    simp [offsetM,maskM_model p s h,hs,aRow,hi,row]

theorem offsetN_model {α : Type} (p : Parameters) (s : State α)
    (h : (control p s).pidN = (pidN p s : Int)) (i : Fin (BN p)) :
    offsetN p s i = (bCol p s i : Int) := by
  have hs : (control p s).startN = (pidN p s : Int) * (BN p : Int) := by
    change (control p s).pidN * (BN p : Int) = _
    rw [h]
  by_cases hi : pidN p s * BN p + i.val < p.N <;>
    simp [offsetN,maskN_model p s h,hs,bCol,hi,col]

#axiomsClean control_coordinates
#axiomsClean maskM_model
#axiomsClean maskN_model
#axiomsClean offsetM_model
#axiomsClean offsetN_model
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
