import bench.tritonbench_g.context_attn_nopad.Proofs.IOSourceSafety

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.IOSourceSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Formula Program
open FP.ProfiledRewrite (engine)
open FPReference
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

theorem range_safe (p : Layout) (r : Regions) (scale : ℝ) (o : FPBackend.Ops α)
    (hBN : 0<p.BN) (initialState s : State α) (old : Acc p α)
    (live : Matches p r o.algebra (FPReference.inputs r initialState) old s)
    (memory : s.mem=initialState.mem) (cur : Int) (hc : 0 ≤ cur) (aligned : ∃ n : Nat,cur=(n*p.BN : Nat))
    (bounds : RegionName → Nat) (hb : Bounds p r (initialState.pids 0) (FPReference.inputs r initialState) bounds) :
    Safety.rangeIntSafe (engine o.algebra) bounds "start_n" cur
      ((blockMask p (FPReference.inputs r initialState)*((FPReference.inputs r initialState).block+1)*p.BM : Nat) : Int) p.BN (FPSymbolic.parts p r scale).body s := by
  apply Safety.rangeIntSafe_of_invariant (engine o.algebra) bounds "start_n"
    ((blockMask p (FPReference.inputs r initialState)*((FPReference.inputs r initialState).block+1)*p.BM : Nat) : Int) p.BN _
    (fun q t => 0 ≤ q ∧ (∃ n : Nat,q=(n*p.BN : Nat)) ∧ ∃ acc,Matches p r o.algebra (FPReference.inputs r initialState) acc t ∧ t.mem=initialState.mem)
  · rintro q hstop t ⟨hq,ha,acc,live,hm⟩
    exact body_safe p r scale o hBN initialState t acc live q hq ha hstop bounds hb
  · rintro q _ t ⟨hq,ha,acc,live,hm⟩ u ran
    obtain ⟨v,hv,hl,hm'⟩ := body p r scale o hBN initialState t acc live hm q
    rw [hv] at ran
    cases Option.some.inj ran
    refine ⟨by omega,?_,_,hl,hm'⟩
    obtain ⟨n,rfl⟩ := ha
    refine ⟨n+1,?_⟩
    push_cast
    ring
  · exact ⟨hc,aligned,old,live,memory⟩

theorem loop_safe (p : Layout) (r : Regions) (scale : ℝ) (o : FPBackend.Ops α)
    (hBN : 0<p.BN) (initialState s : State α)
    (live : Matches p r o.algebra (FPReference.inputs r initialState) (initial p o.algebra) s)
    (memory : s.mem=initialState.mem)
    (bounds : RegionName → Nat) (hb : Bounds p r (initialState.pids 0) (FPReference.inputs r initialState) bounds) :
    Safety.stepSafe (engine o.algebra) bounds
      (.forRangeInt "start_n" (FPSymbolic.parts p r scale).start (FPSymbolic.parts p r scale).stop
        (FPSymbolic.parts p r scale).stride (FPSymbolic.parts p r scale).body) s := by
  have hblock := live.2.1
  have hmask := live.2.2.2.2.2.2.2.2.2.2.2.2.2
  have safe := range_safe p r scale o hBN initialState s _ live memory 0 (by omega) ⟨0,by simp⟩ bounds hb
  simpa (config := {maxSteps := 500000}) [FPSymbolic.parts,Program.parts,Program.mathematical,Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnNopad.original,VeriTile.Bench.Repairs.ContextAttnNopad.SelectedSource.original,Source.intRange?,ComputeKernel.surfaceBody,
    Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,
    evalOp_unfold,Option.bind,numeric,bop,State.setReg,hblock,hmask] using safe

theorem program_safe (p : Layout) (r : Regions) (scale : ℝ) (o : FPBackend.Ops α)
    (hBN : 0<p.BN) (s : State α) (bounds : RegionName → Nat)
    (hb : Bounds p r (s.pids 0) (FPReference.inputs r s) bounds) :
    TraceSafe (engine o.algebra) bounds (FPSymbolic.program p r scale) s := by
  obtain ⟨pre,hpre,live,memory⟩ := prelude p r scale o s
  obtain ⟨post,hloop,live',memory'⟩ := loop p r scale o hBN s pre live memory
  change Safety.traceSafeList _ _ _ _
  rw [FPSymbolic.source_split,List.append_assoc,Safety.traceSafeList_append]
  refine ⟨prelude_safe p r scale o s bounds hb,?_⟩
  intro t ht
  rw [hpre] at ht
  cases Option.some.inj ht
  simp only [List.singleton_append]
  apply Safety.traceSafeList_cons_of_step
  · exact loop_safe p r scale o hBN s pre live memory bounds hb
  · exact hloop
  · exact epilogue_safe p r scale o s post _ live' bounds hb

#axiomsClean program_safe
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.IOSourceSafety
