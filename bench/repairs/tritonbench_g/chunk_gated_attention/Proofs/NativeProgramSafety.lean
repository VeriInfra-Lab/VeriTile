import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.NativeInnerAccessSafety

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.NativeSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaSimple
open NativeValue
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

theorem inner_range_safe (c : NativeValue.Shape) (base : Nat) (gateK hasSeed storeFinal : Bool)
    (initial current : State NativeGraph.Expr) (cur : Nat) (acc : HIndex c → NativeGraph.Expr)
    (ctx : Context c base initial current) (ha : HasAcc c current acc)
    (bounds : RegionName → Nat) (hb : InnerBounds c base initial gateK hasSeed storeFinal bounds) :
    Safety.rangeSafe (engine algebra) bounds "i_t" cur c.NT 1
      (parts c base gateK hasSeed storeFinal).body current := by
  rw [Safety.rangeSafe_unfold]
  simp only [one_ne_zero,↓reduceIte]
  by_cases active : cur<c.NT
  · simp only [active,if_true]
    refine ⟨inner_body_safe c base gateK hasSeed storeFinal initial current cur acc ctx ha bounds hb active,?_⟩
    let next := bodyExecution c base gateK hasSeed storeFinal initial current cur acc ctx ha
    rw [next.property]
    exact inner_range_safe c base gateK hasSeed storeFinal initial next.val (cur+1) _
      (bodyContext c base gateK hasSeed storeFinal initial current cur acc ctx ha)
      (bodyAcc c base gateK hasSeed storeFinal initial current cur acc ctx ha) bounds hb
  · simp [active]
termination_by c.NT-cur
decreasing_by omega

theorem inner_safe (c : NativeValue.Shape) (base : Nat) (gateK hasSeed storeFinal : Bool)
    (initial : State NativeGraph.Expr) (bounds : RegionName → Nat)
    (hb : InnerBounds c base initial gateK hasSeed storeFinal bounds) :
    TraceSafe (engine algebra) bounds (NativeValue.source c base gateK hasSeed storeFinal) initial := by
  let start := entryExecution algebra c base gateK hasSeed storeFinal initial
  have ctx := entryContext algebra c base gateK hasSeed storeFinal initial
  obtain ⟨mid,ran,context,acc,memory⟩ := loopExecution c base gateK hasSeed storeFinal initial start.val
    ctx.1 ctx.2 (entryMemory c base gateK hasSeed storeFinal initial)
  unfold TraceSafe
  rw [source_parts,List.append_assoc,Safety.traceSafeList_append]
  refine ⟨inner_entry_safe c base gateK hasSeed storeFinal initial bounds hb,?_⟩
  intro s hs
  have equal := Option.some.inj (hs.symm.trans start.property)
  subst s
  change Safety.traceSafeList (engine algebra) bounds
    (.forRange "i_t" 0 c.NT 1 (parts c base gateK hasSeed storeFinal).body ::
      (parts c base gateK hasSeed storeFinal).after) start.val
  apply Safety.traceSafeList_cons_of_step (t:=mid)
  · rw [Safety.stepSafe_unfold]
    exact inner_range_safe c base gateK hasSeed storeFinal initial start.val 0 _ ctx.1 ctx.2 bounds hb
  · simpa only [step] using ran
  · exact inner_finish_safe c base gateK hasSeed storeFinal initial mid _ context acc bounds hb

#axiomsClean inner_safe
#axiomsClean prefix_safe
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.NativeSafety
