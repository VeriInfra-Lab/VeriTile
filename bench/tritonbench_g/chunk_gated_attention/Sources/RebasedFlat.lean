import bench.tritonbench_g.chunk_gated_attention.Sources.RebasedSafety
import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealHFlat

/-! Complete actual repaired H execution in finite disjoint flat allocations.
The shared flattening semantics fixes its oracle to zero. Choosing a region
representative with that oracle changes neither flat input nor any physical
input coordinate, so no caller oracle restriction is needed. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedFlat
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealHMath RealHProgram
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
abbrev footprint := RealHFlat.footprint
abbrev Outputs := RealHFlat.Outputs

theorem source_executes (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (p : Layout) (r : Regions) (dot gate history : NativeSource.Storage)
    (seed final : FloatDType) (base : Nat) (gateK hasSeed storeFinal swapped : Bool) (a : Inputs) (u : BlockState)
    (hfull : ∀ c, c < p.NT → RebasedAccess.Full p (RebasedInit.decoded p base u) c)
    (hkey : (RebasedInit.decoded p base u).key*p.BK+p.BK ≤ p.K)
    (hvalue : (RebasedInit.decoded p base u).value*p.BV+p.BV ≤ p.V)
    (hs : RealHStep.Separate r)
    (hfsep : storeFinal = Bool.true → r.history ≠ r.final)
    (hhi : RealHLoop.HistoryNoCollision p (RebasedInit.decoded p base u))
    (hfi : storeFinal = Bool.true → RealHStore.NoCollision p (RebasedInit.decoded p base u) (RealHFinal.base p (RebasedInit.decoded p base u)) p.V 1)
    (hk : ∀ off, u.readMem r.key off = a.key off)
    (hv : ∀ off, u.readMem r.value off = a.value off)
    (hg : ∀ off, u.readMem r.gate off = a.gate off)
    (hseed : ∀ off, u.readMem r.seed off = a.seed off)
    (hb : RebasedSafety.Bounds p (RebasedInit.decoded p base u) r gateK hasSeed storeFinal A.extent) :
    ∃ v, exec (A.flattenKernel (RebasedProgram.source p r dot gate history seed final base gateK hasSeed storeFinal swapped).eraseDType.toAlgKernel)
        (A.flattenState u) = some v ∧ Outputs A p (RebasedInit.decoded p base u) r a gateK hasSeed storeFinal v ∧
      BlockState.WriteWithin (footprint A p (RebasedInit.decoded p base u) r storeFinal) (A.flattenState u) v := by
  let u0 : BlockState := {u with undef := fun _ _ => 0}
  have hrep : A.flattenState u0 = A.flattenState u := rfl
  obtain ⟨v,he,ho,hf⟩ := RebasedExecution.executes p r dot gate history seed final base gateK hasSeed storeFinal swapped a u0
    hfull hkey hvalue hs hfsep hhi hfi hk hv hg hseed
  have hflat := A.exec_flatten hd hcov _ u0
    (RebasedSafety.source_safe p r dot gate history seed final base gateK hasSeed storeFinal swapped u0
      hfull hkey hvalue hs hhi A.extent hb)
    (RebasedSafety.source_flatten_ok p r dot gate history seed final base gateK hasSeed storeFinal swapped) rfl
  rw [he,Option.map_some,hrep] at hflat
  have hmem (rg : RegionName) (off : Nat) (hbound : off < A.extent rg) :
      (A.flattenState v).readMem A.flat (A.addr rg off) = v.readMem rg off := by
    have hr : rg ∈ A.regions := by
      by_contra hn
      rw [hcov rg hn] at hbound
      exact Nat.not_lt_zero _ hbound
    exact A.flattenState_readMem hd v hr hbound
  refine ⟨A.flattenState v,hflat,⟨?_,?_⟩,?_⟩
  · intro c e j hc ha
    have hoff := (hb.loop c hc).history e j (by simpa using And.intro e.isLt j.isLt)
    have heq : RealHMath.historyAddress p (RebasedInit.decoded p base u) c
        (keyIndex p (RebasedInit.decoded p base u) e.val) (valueIndex p (RebasedInit.decoded p base u) j.val) < A.extent r.history := by
      convert hoff using 1 <;> simp only [historyAddress,keyIndex,valueIndex,Nat.zero_add] <;> ring
    rw [hmem _ _ heq]
    exact ho.history c e j hc ha
  · intro hon e j ha
    have hoff := hb.final hon e j (by simpa using And.intro e.isLt j.isLt)
    have heq : stateAddress p (RebasedInit.decoded p base u)
        (keyIndex p (RebasedInit.decoded p base u) e.val) (valueIndex p (RebasedInit.decoded p base u) j.val) < A.extent r.final := by
      convert hoff using 1 <;> simp only [stateAddress,keyIndex,valueIndex,Nat.zero_add,Nat.mul_one] <;> ring
    rw [hmem _ _ heq]
    exact ho.final hon e j ha
  · simpa only [hrep] using RealHFlat.flatten_frame A p (RebasedInit.decoded p base u) r storeFinal u0 v hf

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedFlat
