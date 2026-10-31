import bench.tritonbench_g.bmm_chunk_fwd.Proofs.NativePhaseSafety

namespace VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO.Native.SourceSafety
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open KernelInterface NativeGraph FPSuccess SourcePointers
noncomputable section
open Classical
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false


theorem iterations {α : Type} [Inhabited α] (o : Ops α) (h : Host.Call) (c : Host.Config)
    (d : Coordinates) (cur : Nat) (bounds : RegionName → Nat) (s : State α)
    (hs : StateAt h c d cur s) (hbounds : Bounds h c d bounds)
    (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) :
    Safety.rangeSafe (engine o.algebra) bounds "k" cur (Host.cdiv h.shape.K c.K) 1 (parts h c).body s := by
  by_cases less : cur<Host.cdiv h.shape.K c.K
  · obtain ⟨mid,hm,hr,hmem⟩ := SourcePointers.body o h c d cur s hs ha hb
    rw [Safety.rangeSafe,if_neg (by decide),if_pos less,hm]
    exact ⟨body o h c d cur bounds s hs (hbounds.a cur less) (hbounds.b cur less) ha hb,
      iterations o h c d (cur+1) bounds mid hr hbounds ha hb⟩
  · rw [Safety.rangeSafe,if_neg (by decide),if_neg less]
    trivial
termination_by Host.cdiv h.shape.K c.K-cur

/-- The actual source trace is safe at each intermediate state returned by the
already-proved source phases. No hypothetical execution is substituted. -/
theorem program {α : Type} [Inhabited α] (o : Ops α) (h : Host.Call) (c : Host.Config)
    (bounds : RegionName → Nat) (s : State α)
    (hbounds : NativeReference.activeBlock h c (NativeReference.coordinatesFor h c s.pids)=Bool.true →
      Bounds h c (NativeReference.coordinatesFor h c s.pids) bounds)
    (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) :
    TraceSafe (engine o.algebra) bounds (Kernels.source h c) s := by
  let d := NativeReference.coordinatesFor h c s.pids
  obtain ⟨entry,hpre,hbase,hmem,hactive,hcontrols⟩ := FPSuccess.prologue o h c s
  have controls : SourcePointers.Controls d entry := hcontrols
  change Safety.traceSafeList (engine o.algebra) bounds (Kernels.source h c).surfaceBody s
  rw [source_split,Safety.traceSafeList_append]
  refine ⟨prologue o h c bounds s ha hb,?_⟩
  intro entry' runEntry
  have same : entry'=entry := Option.some.inj (runEntry.symm.trans hpre)
  subst entry'
  apply Safety.traceSafeList_cons_intro
  · rw [Safety.stepSafe]
    constructor
    · cases causal : h.causal <;>
        simp [branch,Kernels.source,Host.Call.kernel,ha,hb,TypedSource.specialize,VeriTile.Bench.Repairs.BmmChunkFwd.Kernels.mainSource,causal,
          Source.conditional?,Source.dynamicRange?,ComputeKernel.surfaceBody,Safety.safeOp.eq_def]
    · rw [hactive]
      cases act : FPSuccess.active h c s
      · trivial
      · have bounded := hbounds (show NativeReference.activeBlock h c d=Bool.true from act)
        obtain ⟨prepared,hsetup,hstate,hm0⟩ := SourcePointers.setup o h c d entry controls ha hb
        obtain ⟨terminal,hrange,hterminal,hm1⟩ := SourcePointers.iterations o h c d
          0 (Host.cdiv h.shape.K c.K) (Nat.zero_le _) prepared hstate ha hb
        have hs : evalOp (engine o.algebra) none (parts h c).start prepared=some (fun _=>(0 : Nat)) := by
          simp only [parts,branch,Kernels.source,Host.Call.kernel,ha,hb,TypedSource.specialize]; rfl
        have he : evalOp (engine o.algebra) none (parts h c).stop prepared=some (fun _=>Host.cdiv h.shape.K c.K) := by
          simp only [parts,branch,Kernels.source,Host.Call.kernel,ha,hb,TypedSource.specialize]; rfl
        have hd : evalOp (engine o.algebra) none (parts h c).stride prepared=some (fun _=>(1 : Nat)) := by
          simp only [parts,branch,Kernels.source,Host.Call.kernel,ha,hb,TypedSource.specialize]; rfl
        have hloop : step (engine o.algebra)
            (.forRangeDyn "k" (parts h c).start (parts h c).stop (parts h c).stride (parts h c).body)
            prepared=some terminal := by
          simpa only [step,hs,he,hd,Option.bind_some] using hrange
        change Safety.traceSafeList (engine o.algebra) bounds (branch h c).yes entry
        rw [branch_split,List.append_assoc,Safety.traceSafeList_append]
        refine ⟨setup o h c d bounds entry controls ha hb,?_⟩
        intro prepared' runPrepared
        have same : prepared'=prepared := Option.some.inj (runPrepared.symm.trans hsetup)
        subst prepared'
        apply Safety.traceSafeList_cons_of_step (t:=terminal) (executed:=hloop)
        · rw [Safety.stepSafe,hs,he,hd]
          refine ⟨?_,?_,?_,iterations o h c d 0 bounds prepared hstate bounded ha hb⟩
          all_goals
            cases causal : h.causal <;>
              simp [parts,branch,Kernels.source,Host.Call.kernel,ha,hb,TypedSource.specialize,VeriTile.Bench.Repairs.BmmChunkFwd.Kernels.mainSource,causal,
                Source.conditional?,Source.dynamicRange?,ComputeKernel.surfaceBody,Safety.safeOp.eq_def]
        · exact epilogue o h c d _ bounds terminal hterminal bounded.rowSeq bounded.colSeq bounded.out ha hb
  · intro final _
    exact Safety.traceSafeList_nil _ _ final

end
end VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO.Native.SourceSafety
