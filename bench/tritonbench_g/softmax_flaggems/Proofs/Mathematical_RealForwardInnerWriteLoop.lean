import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealForwardInnerWriteback
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealForwardInnerFullStore

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
open VeriTile Triton

 theorem Pass2Inv.setStart {sb st : BlockState} {Y X : RegionName}
    {row N B c start : Nat} {hB : 0 < B} (h : Pass2Inv sb Y X row N B hB c st) :
    Pass2Inv sb Y X row N B hB c (st.setReg "start_n" .nat [] (Tile.scalar start)) := by
  refine ⟨h.frame, h.input, h.pids, h.numPids, h.undef, ?_, ?_, ?_, ?_, ?_, ?_, h.values⟩
  · simpa using h.rowReg
  · simpa using h.inputReg
  · simpa using h.outputReg
  · simpa using h.previousReg
  · simpa using h.maxReg
  · simpa using h.sumReg

 theorem Pass2Inv.storeMid {sb st : BlockState} {Y X : RegionName}
    {row N B c base : Nat} {hB : 0 < B} (mx : WithBot ℝ) (z : ℝ)
    (h : Pass2Inv sb Y X row N B hB c st) :
    Pass2Inv sb Y X row N B hB c (storeMid st X row N base B mx z) := by
  refine ⟨h.frame, h.input, h.pids, h.numPids, h.undef, ?_, ?_, ?_, ?_, ?_, ?_, h.values⟩
  · simpa [RealForwardInner.storeMid] using h.rowReg
  · simpa [RealForwardInner.storeMid] using h.inputReg
  · simpa [RealForwardInner.storeMid] using h.outputReg
  · simpa [RealForwardInner.storeMid] using h.previousReg
  · simpa [RealForwardInner.storeMid] using h.maxReg
  · simpa [RealForwardInner.storeMid] using h.sumReg

 theorem Pass2Inv.fullStoreMid {sb st : BlockState} {Y X : RegionName}
    {row N B c base : Nat} {hB : 0 < B} (mx : WithBot ℝ) (z : ℝ)
    (h : Pass2Inv sb Y X row N B hB c st) :
    Pass2Inv sb Y X row N B hB c (fullStoreMid st X row N base B mx z) := by
  refine ⟨h.frame, h.input, h.pids, h.numPids, h.undef, ?_, ?_, ?_, ?_, ?_, ?_, h.values⟩
  · simpa [RealForwardInner.fullStoreMid] using h.rowReg
  · simpa [RealForwardInner.fullStoreMid] using h.inputReg
  · simpa [RealForwardInner.fullStoreMid] using h.outputReg
  · simpa [RealForwardInner.fullStoreMid] using h.previousReg
  · simpa [RealForwardInner.fullStoreMid] using h.maxReg
  · simpa [RealForwardInner.fullStoreMid] using h.sumReg

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
