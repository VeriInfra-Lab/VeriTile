import bench.tritonbench_g.uniform_sampling.Proofs.SafetyPrefix
import bench.tritonbench_g.uniform_sampling.Proofs.SafetyStores

/-! Successful finite flat-memory execution of the complete source for every
launched program. All four streams have their independently specified values,
with exact cell tags and a frame at every other flat address. Both original
signed64 arguments and the launched i32 index domain remain explicit. Zero
undef is an explicit requirement of the generic whole-state flatten bridge.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.RealIO
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 10000

theorem trace_safe (swapped : Bool) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) (hB : 0 < B) (u : BlockState)
    (hseed : Random.IntegerType.i64.inRange seed = Bool.true) (bounds : RegionBounds)
    (hb : FlatMemory.Bounds y N B u.pid bounds) :
    (Algorithm.kernel swapped y N seed offset lower upper B).TraceSafe bounds u := by
  unfold Kernel.TraceSafe
  rw [← List.take_append_drop 19 (Algorithm.kernel swapped y N seed offset lower upper B).body,
    SafetyPrefix.safe_append]
  refine ⟨SafetyPrefix.prefix_safe bounds swapped y N seed offset lower upper B u, ?_⟩
  intro t ht
  rw [Execution.prefix_exec swapped y N seed offset lower upper B u hseed] at ht
  cases ht
  rw [StorePhase.source_slice]
  exact SafetyStores.stores_safe hB bounds u.pid y N seed offset lower upper _
    (Execution.result_values seed offset lower upper B u)
    (Execution.result_offsets seed offset lower upper B u) hb

def Implements (kernel : Kernel) (y : RegionName) (N B : Nat)
    (seed offset : Int) (lower upper : ℝ) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ u : BlockState, N < 2^31 → u.pid < SourceDomain.grid N B →
    Random.IntegerType.i64.inRange seed = Bool.true →
    Random.IntegerType.i64.inRange offset = Bool.true →
    u.undef = (fun _ _ => 0) → FlatMemory.Bounds y N B u.pid A.extent →
    ∃ v, exec (A.flattenKernel kernel) (A.flattenState u) = some v ∧
      (∀ f : Fin 4, ∀ j : Fin B, SourceDomain.outputIndex u.pid B f j < N →
        v.mem A.flat (A.addr y (SourceDomain.outputIndex u.pid B f j)) =
          MemCell.real (Formula.expected B seed offset lower upper (SourceDomain.outputIndex u.pid B f j))) ∧
      (∀ r o, FlatMemory.Outside A u.pid y N B r o →
        v.mem r o = (A.flattenState u).mem r o)

theorem resultBase_flat_mem (A : FlatAlloc) (seed offset : Int) (lower upper : ℝ)
    (B : Nat) (u : BlockState) :
    (A.flattenState (Execution.resultBase seed offset lower upper B u)).mem =
      (A.flattenState u).mem := by
  simp only [FlatAlloc.flattenState, FlatAlloc.readFlat,
    (Execution.resultBase_frame seed offset lower upper B u).1]

theorem source_correct (storage : TypedSource.Storage) (swapped : Bool) (y : RegionName)
    (N : Nat) (seed offset : Int) (lower upper : ℝ) (B : Nat)
    (hB : SourceDomain.LegalBlock B) :
    Implements (TypedSource.source storage swapped y N seed offset lower upper B hB).eraseDType.toAlgKernel
      y N B seed offset lower upper := by
  intro A hd hc u hN hpid hseed hoffset hu hb
  let v := Memory.storeBlock u.pid (Execution.resultBase seed offset lower upper B u)
    y N B seed offset lower upper
  have hk : (TypedSource.source storage swapped y N seed offset lower upper B hB).eraseDType.toAlgKernel =
      Algorithm.kernel swapped y N seed offset lower upper B := by
    simp only [ComputeKernel.toAlgKernel, Erasure.complete_projection, Algorithm.source_eq]
  have he := Execution.exec_storeBlock swapped y N seed offset lower upper B
    (SourceDomain.legalBlock_fields hB).1 u hseed
  have hs := trace_safe swapped y N seed offset lower upper B
    (SourceDomain.legalBlock_fields hB).1 u hseed A.extent hb
  have hf := A.exec_flatten hd hc (Algorithm.kernel swapped y N seed offset lower upper B) u hs
    (SafetyPrefix.flatten_ok swapped y N seed offset lower upper B) hu
  refine ⟨A.flattenState v, ?_, ?_, ?_⟩
  · rw [hk, hf, he, Option.map_some]
  · intro f j hj
    exact FlatMemory.storeBlock_active_cell A hd hc u.pid
      (Execution.resultBase seed offset lower upper B u) y N B seed offset lower upper hb f j hj
  · intro r o ho
    exact (FlatMemory.storeBlock_frame A u.pid (Execution.resultBase seed offset lower upper B u)
      y N B seed offset lower upper r o ho).trans
      (congrFun (congrFun (resultBase_flat_mem A seed offset lower upper B u) r) o)

#axiomsClean trace_safe
#axiomsClean resultBase_flat_mem
#axiomsClean source_correct

end
end VeriTile.Bench.TritonBenchG.UniformSampling.RealIO
