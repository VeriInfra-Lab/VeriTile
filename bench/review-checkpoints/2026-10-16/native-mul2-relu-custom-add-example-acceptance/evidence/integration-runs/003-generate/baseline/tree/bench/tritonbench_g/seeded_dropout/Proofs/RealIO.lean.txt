import bench.tritonbench_g.seeded_dropout.Proofs.Safety
import bench.tritonbench_g.seeded_dropout.Proofs.Candidate
namespace VeriTile.Bench.TritonBenchG.SeededDropout.RealIO
open VeriTile.Triton TypedSource Execution
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 3000

/-- Per-launched-program finite execution with original runtime parameters,
canonical random values, exact output cell tags, and every other cell framed.
The zero undef premise is the current generic flatten bridge's explicit
restriction; omitted source padding has not been changed. -/
def Implements (kernel : Kernel) (kind : Random.IntegerType) (x y : RegionName)
    (N : Nat) (p : ℝ) (seed : SeedCarrier kind) (B : Nat) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ u : BlockState, N < 2^31 → u.pid < SourceDomain.grid N B →
    kind.inRange (seedInt kind seed) = Bool.true → u.undef = (fun _ _ => 0) →
    FlatMemory.Bounds x y N B u A.extent →
    ∃ v, exec (A.flattenKernel kernel) (A.flattenState u) = some v ∧
      (∀ j : Fin B, Index.offset u.pid B j < N →
        v.mem A.flat (A.addr y (Index.offset u.pid B j)) =
          MemCell.real (Formula.expected (u.readMem x) p (seedInt kind seed) (Index.offset u.pid B j))) ∧
      (∀ r o, FlatMemory.Outside A u y N B r o →
        v.mem r o = (A.flattenState u).mem r o)

theorem resultBase_flat_mem (A : FlatAlloc) (kind : Random.IntegerType) (x : RegionName)
    (N : Nat) (p : ℝ) (seed : SeedCarrier kind) (B : Nat) (u : BlockState) :
    (A.flattenState (resultBase kind x N p seed B u)).mem = (A.flattenState u).mem := by
  simp only [FlatAlloc.flattenState, FlatAlloc.readFlat, resultBase_mem]

theorem original_correct (storage : Storage) (kind : Random.IntegerType) (x y : RegionName)
    (N : Nat) (p : ℝ) (seed : SeedCarrier kind) (B : Nat) (hB : SourceDomain.LegalBlock B) :
    Implements (source storage kind Bool.false x y N p seed B hB).eraseDType.toAlgKernel
      kind x y N p seed B := by
  intro A hd hc u hN hpid hseed hu hb
  let v := Memory.storeBlock u (resultBase kind x N p seed B u) x y N B p (seedInt kind seed)
  have he := exec_storeBlock kind x y N p seed B hB u hN hpid hseed
  have hs := Safety.trace_safe kind x y N p seed B hB u hN hpid hseed A.extent hb
  have hok := Safety.flatten_ok kind x y N p seed B hB
  have hf := A.exec_flatten hd hc (original kind x y N p seed B hB) u hs hok hu
  refine ⟨A.flattenState v, ?_, ?_, ?_⟩
  · rw [Erasure.kernel]
    change exec (A.flattenKernel (original kind x y N p seed B hB)) (A.flattenState u) = _
    rw [hf, he, Option.map_some]
  · intro j hj
    exact FlatMemory.storeBlock_active_cell A hd hc u (resultBase kind x N p seed B u)
      x y N B p (seedInt kind seed) hb j hj
  · intro r o ho
    exact (FlatMemory.storeBlock_frame A u (resultBase kind x N p seed B u)
      x y N B p (seedInt kind seed) r o ho).trans
      (congrFun (congrFun (resultBase_flat_mem A kind x N p seed B u) r) o)

theorem candidate_correct (storage : Storage) (kind : Random.IntegerType) (x y : RegionName)
    (N : Nat) (p : ℝ) (seed : SeedCarrier kind) (B : Nat) (hB : SourceDomain.LegalBlock B) :
    Implements (source storage kind Bool.true x y N p seed B hB).eraseDType.toAlgKernel
      kind x y N p seed B := by
  intro A hd hc u hN hpid hseed hu hb
  obtain ⟨v, he, hv, hf⟩ := original_correct storage kind x y N p seed B hB
    A hd hc u hN hpid hseed hu hb
  refine ⟨v, ?_, hv, hf⟩
  rw [← Candidate.flat_exec]
  exact he

end
end VeriTile.Bench.TritonBenchG.SeededDropout.RealIO
