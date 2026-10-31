import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealBackwardInnerStore
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealBackwardInnerPrepare

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
open VeriTile Triton FullSource
open scoped BigOperators
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

noncomputable def oneSum (sb : BlockState) (P G : RegionName)
    (M N rbase B C : Nat) (k : Fin C) : ℝ :=
  ∑ i : Fin B, sample sb P M N (rbase+k.val) i.val *
    sample sb G M N (rbase+k.val) i.val

structure OneInv (sb : BlockState) (P G : RegionName)
    (M N rbase B C : Nat) (st : BlockState) : Prop where
  memory : st.mem=sb.mem
  pids : st.pids=sb.pids
  numPids : st.numPids=sb.numPids
  undef : st.undef=sb.undef
  offsetReg : st.regs .nat [C,B] "offsets" = some ⟨tileOffset N rbase 0⟩
  maskReg : st.regs .bool [C,B] "mask" = some ⟨tileActive M N rbase 0⟩
  pReg : st.regs .real [C,B] "out_tile" =
    some ⟨fun i ↦ some (tileSample sb P M N rbase 0 i)⟩
  gReg : st.regs .real [C,B] "out_grad_tile" =
    some ⟨fun i ↦ some (tileSample sb G M N rbase 0 i)⟩
  zReg : st.regs .real [C] "scale" =
    some ⟨fun i ↦ some (oneSum sb P G M N rbase B C i.1)⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
