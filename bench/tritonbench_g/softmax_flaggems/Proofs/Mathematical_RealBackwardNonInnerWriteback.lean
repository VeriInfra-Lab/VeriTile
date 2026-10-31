import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealBackwardNonInnerStore
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealBackwardNonInnerPrepare
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealForwardNonInnerWriteback

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
open VeriTile Triton
open scoped BigOperators
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

abbrev outputFootprint := RealForwardNonInner.outputFootprint

noncomputable def expected (sb : BlockState) (P G : RegionName)
    (row N K padded n k : Nat) : ℝ :=
  RealMath.backwardPadded padded
    (fun j ↦ sample sb P row N K j k) (fun j ↦ sample sb G row N K j k) n

theorem storeValue_expected {B C : Nat} (sb st : BlockState) (P G : RegionName)
    (row N K kbase start : Nat) (hB : 0 < B)
    (hp : ∀ o, st.readMem P o = sb.readMem P o)
    (hg : ∀ o, st.readMem G o = sb.readMem G o)
    (i : TileIndex [B,C]) (hi : tileActive N K kbase start i = Bool.true) :
    storeValue st P G row N K kbase start (finalSum sb P G row N K kbase B C) i =
      expected sb P G row N K (RealAddress.rounds N B*B) (start+i.1.val)
        (kbase+i.2.1.val) := by
  have hi' : start+i.1.val<N ∧ kbase+i.2.1.val<K := of_decide_eq_true hi
  simp only [storeValue, tileSample, finalSum_eq sb P G row N K kbase B C hB,
    expected, RealMath.backwardPadded]
  simp only [sample, if_pos hi', hp, hg]

structure Pass2Inv (sb : BlockState) (P G Y : RegionName)
    (row N K kbase B C c : Nat) (st : BlockState) : Prop where
  frame : BlockState.WriteWithin (outputFootprint Y row N K kbase C) sb st
  inputP : ∀ o, st.mem P o = sb.mem P o
  inputG : ∀ o, st.mem G o = sb.mem G o
  pids : st.pids = sb.pids
  numPids : st.numPids = sb.numPids
  undef : st.undef = sb.undef
  columnReg : st.regs .nat [C] "offsets_k" = some ⟨fun i ↦ kbase+i.1.val⟩
  nReg : st.regs .nat [B] "offsets_n" = some ⟨fun i ↦ c*B+i.1.val⟩
  offsetReg : st.regs .nat [B,C] "offsets" = some ⟨tileOffset row N K kbase (c*B)⟩
  sumReg : st.regs .real [C] "scale" =
    some ⟨fun i ↦ some (finalSum sb P G row N K kbase B C i.1)⟩
  values : ∀ n k, n < N → k < C → kbase+k<K → n / B < c →
    st.readMem Y (offset row N K n (kbase+k)) =
      expected sb P G row N K (RealAddress.rounds N B*B) n (kbase+k)

theorem scatter_preserve_processed {B C : Nat} (s : BlockState) (Y : RegionName)
    (row N K kbase c n k : Nat) (value : TileIndex [B,C] → ℝ)
    (hB : 0 < B) (hold : n/B<c) (hk : k<K) :
    (scatter s Y (tileOffset row N K kbase (c*B)) value
      (tileActive N K kbase (c*B))).mem Y (offset row N K n k) =
      s.mem Y (offset row N K n k) := by
  apply foldl_store_preserve_cell _ value (fun i ↦ tileActive N K kbase (c*B) i = Bool.true)
  intro i _ hi heq
  have hi' : c*B+i.1.val<N ∧ kbase+i.2.1.val<K := of_decide_eq_true hi
  have ha := heq.2
  unfold tileOffset offset at ha
  have hcanc : (c*B+i.1.val)*K+(kbase+i.2.1.val)=n*K+k := by omega
  have he := (RealAddress.row_column_injective hi'.2 hk hcanc).1
  have hsmall := (Nat.div_lt_iff_lt_mul hB).mp hold
  omega

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
