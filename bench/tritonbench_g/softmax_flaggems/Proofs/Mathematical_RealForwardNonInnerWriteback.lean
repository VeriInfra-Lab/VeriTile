import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealForwardNonInnerScatter

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
open VeriTile Triton

def outputFootprint (Y : RegionName) (row N K kbase C : Nat) : WriteFootprint :=
  fun a ↦ a.1 = Y ∧ ∃ n k : Nat,
    n < N ∧ k < C ∧ kbase + k < K ∧
      a.2 = row * N * K + n * K + (kbase + k)

structure Pass2Inv (sb : BlockState) (Y X : RegionName)
    (row N K kbase B C : Nat) (hB : 0 < B) (c : Nat) (st : BlockState) : Prop where
  frame : BlockState.WriteWithin (outputFootprint Y row N K kbase C) sb st
  input : ∀ o, st.mem X o = sb.mem X o
  pids : st.pids = sb.pids
  numPids : st.numPids = sb.numPids
  undef : st.undef = sb.undef
  rowReg : st.regs .nat [] "pid_m" = some (Tile.scalar row)
  columnReg : st.regs .nat [C] "k_offsets" = some ⟨fun i ↦ kbase + i.1.val⟩
  previousReg : st.regs .nat [] "previous_multiple" = some (Tile.scalar (RealAddress.previous N B))
  maxReg : st.regs .real [C] "m" = some ⟨fun i ↦ finalMax sb X row N K kbase B C hB i.1⟩
  sumReg : st.regs .real [C] "z" = some ⟨fun i ↦ some (finalSum sb X row N K kbase B C hB i.1)⟩
  values : ∀ n k, n < N → k < C → kbase + k < K →
    RealAddress.rounds N B - c ≤ n / B →
    st.readMem Y (row * N * K + n * K + (kbase + k)) =
      expected sb X row N K n (kbase + k)

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
