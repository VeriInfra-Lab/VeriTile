import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealBackwardNonInnerStore
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealBackwardNonInnerPrepare

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
open VeriTile Triton FullSource
open scoped BigOperators
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

noncomputable def oneSum (sb : BlockState) (P G : RegionName)
    (row N K kbase B C : Nat) (k : Fin C) : ℝ :=
  ∑ i : Fin B, sample sb P row N K i.val (kbase+k.val) *
    sample sb G row N K i.val (kbase+k.val)

structure OneInv (sb : BlockState) (P G : RegionName)
    (row N K kbase B C : Nat) (st : BlockState) : Prop where
  memory : st.mem=sb.mem
  pids : st.pids=sb.pids
  numPids : st.numPids=sb.numPids
  undef : st.undef=sb.undef
  offsetReg : st.regs .nat [B,C] "offsets" = some ⟨tileOffset row N K kbase 0⟩
  maskReg : st.regs .bool [B,C] "mask" = some ⟨tileActive N K kbase 0⟩
  pReg : st.regs .real [B,C] "out_tile" =
    some ⟨fun i ↦ some (tileSample sb P row N K kbase 0 i)⟩
  gReg : st.regs .real [B,C] "out_grad_tile" =
    some ⟨fun i ↦ some (tileSample sb G row N K kbase 0 i)⟩
  zReg : st.regs .real [C] "scale" =
    some ⟨fun i ↦ some (oneSum sb P G row N K kbase B C i.1)⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
