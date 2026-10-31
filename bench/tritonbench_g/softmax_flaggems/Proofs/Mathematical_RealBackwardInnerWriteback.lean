import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealBackwardInnerScatter
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealBackwardInnerPrepare

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
open VeriTile Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def outputFootprint (Y : RegionName) (M N rbase C : Nat) : WriteFootprint :=
  fun a ↦ a.1=Y ∧ ∃ r n, r<C ∧ rbase+r<M ∧ n<N ∧ a.2=offset N (rbase+r) n

noncomputable def expected (s : BlockState) (P G : RegionName)
    (M N padded r n : Nat) : ℝ :=
  RealMath.backwardPadded padded (fun j ↦ sample s P M N r j)
    (fun j ↦ sample s G M N r j) n

theorem storeValue_expected {B C : Nat} (sb st : BlockState) (P G : RegionName)
    (M N rbase start : Nat) (hB : 0<B)
    (hp : ∀ o, st.readMem P o=sb.readMem P o)
    (hg : ∀ o, st.readMem G o=sb.readMem G o)
    (i : TileIndex [C,B]) (hi : tileActive M N rbase start i=Bool.true) :
    storeValue st P G M N rbase start (finalSum sb P G M N rbase B C) i =
      expected sb P G M N (RealAddress.rounds N B*B) (rbase+i.1.val) (start+i.2.1.val) := by
  have hi' : rbase+i.1.val<M ∧ start+i.2.1.val<N := of_decide_eq_true hi
  simp only [storeValue, tileSample, finalSum_eq sb P G M N rbase B C hB,
    expected, RealMath.backwardPadded]
  simp only [sample, if_pos hi', hp, hg]

structure Pass2Inv (sb : BlockState) (P G Y : RegionName)
    (M N rbase B C c : Nat) (st : BlockState) : Prop where
  frame : BlockState.WriteWithin (outputFootprint Y M N rbase C) sb st
  inputP : ∀ o, st.mem P o=sb.mem P o
  inputG : ∀ o, st.mem G o=sb.mem G o
  pids : st.pids=sb.pids
  numPids : st.numPids=sb.numPids
  undef : st.undef=sb.undef
  columnReg : st.regs .nat [C] "m_offsets" = some ⟨fun i ↦ rbase+i.1.val⟩
  nReg : st.regs .nat [B] "n_offsets" = some ⟨fun i ↦ c*B+i.1.val⟩
  offsetReg : st.regs .nat [C,B] "offsets" = some ⟨tileOffset N rbase (c*B)⟩
  sumReg : st.regs .real [C] "scale" = some ⟨fun i ↦ some (finalSum sb P G M N rbase B C i.1)⟩
  values : ∀ r n, r<C → rbase+r<M → n<N → n/B<c →
    st.readMem Y (offset N (rbase+r) n) =
      expected sb P G M N (RealAddress.rounds N B*B) (rbase+r) n

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
