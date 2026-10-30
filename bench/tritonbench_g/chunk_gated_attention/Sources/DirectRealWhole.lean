import bench.tritonbench_g.chunk_gated_attention.Sources.DirectRealSequence
import bench.tritonbench_g.chunk_gated_attention.Sources.LogicalCoverage

/-! The whole recurrent host grid, with actual sequential legal launches,
ordinary materialized input snapshots, independent logical readback for every
returned coordinate, optional final readback and the complete allocation frame. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.DirectRealWhole
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealHMath RealHProgram
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000

def historyOffset (p : Layout) (h c k v : Nat) := h*p.historyHead+c*p.K*p.V+k*p.historyKey+v*p.historyValue
def finalOffset (p : Layout) (h k v : Nat) := h*p.K*p.V+k*p.V+v

structure Outputs (p : Layout) (heads T K V : Nat) (r : Regions)
    (x : Nat → Host.InnerInputs) (gateK hasSeed storeFinal : Bool) (v : BlockState) : Prop where
  history : ∀ h c k j, h<heads → c<p.NT → k<K → j<V →
    v.readMem r.history (historyOffset p h c k j) = Host.innerState (x h) T p.BT gateK hasSeed c k j
  final : storeFinal=Bool.true → ∀ h k j, h<heads → k<K → j<V →
    v.readMem r.final (finalOffset p h k j) = Host.innerState (x h) T p.BT gateK hasSeed p.NT k j

theorem executes (p : Layout) (heads T K V : Nat) (r : Regions) (dot gate history : NativeSource.Storage)
    (seed final : FloatDType) (gateK hasSeed storeFinal swapped : Bool) (x : Nat → Host.InnerInputs)
    (hp : StagedLayout.Canonical p) (hk : K≤p.K) (hv : V≤p.V)
    (hs : DirectRealSequence.InputSeparate r storeFinal)
    (hfsep : storeFinal=Bool.true → r.history≠r.final)
    (u : BlockState)
    (hkey : ∀ off, u.readMem r.key off=(Materialize.inputs p T K V gateK x).key off)
    (hvalue : ∀ off, u.readMem r.value off=(Materialize.inputs p T K V gateK x).value off)
    (hgate : ∀ off, u.readMem r.gate off=(Materialize.inputs p T K V gateK x).gate off)
    (hseed : ∀ off, u.readMem r.seed off=(Materialize.inputs p T K V gateK x).seed off) :
    ∃ v, DirectRealSequence.Runs p r dot gate history seed final gateK hasSeed storeFinal swapped
        (LaunchPlan.plans (heads*(p.K/p.BK)*(p.V/p.BV))) {u with undef := fun _ _ => 0} v ∧
      Outputs p heads T K V r x gateK hasSeed storeFinal v ∧
      BlockState.WriteWithin
        (DirectRealSequence.allWrites p r
          (LaunchPlan.plans (heads*(p.K/p.BK)*(p.V/p.BV))) storeFinal u) u v := by
  let u0 : BlockState := {u with undef := fun _ _ => 0}
  let a := Materialize.inputs p T K V gateK x
  let total := heads*(p.K/p.BK)*(p.V/p.BV)
  obtain ⟨v,hrun,ho,hf,hu⟩ := DirectRealSequence.executes p heads r dot gate history seed final
    gateK hasSeed storeFinal swapped a hp hs hfsep (LaunchPlan.plans total)
    (LaunchPlan.plans_valid total) (LaunchPlan.plans_ordered total) u0 rfl hkey hvalue hgate hseed
  refine ⟨v,hrun,⟨?_,?_⟩,hf⟩
  · intro h c k j hhh hc hkk hjj
    have hkp := lt_of_lt_of_le hkk hk
    have hjp := lt_of_lt_of_le hjj hv
    obtain ⟨plan,hplan,idx,e,f,hhead,hek,hjv⟩ := LogicalCoverage.coverage p heads h k j u0 hp hhh hkp hjp
    let l := RebasedInit.decoded p plan.base (u0.withGridIndex idx)
    have hr := (ho plan hplan idx).history c e f hc ⟨by simpa [hek] using hkp,by simpa [hjv] using hjp⟩
    have hcform := HostStaging.closed_eq p l a (x l.head) T K V gateK hasSeed
      (Materialize.staged p l T K V gateK x hp hk hv) c _ _ hc.le
      (by simpa [l,hek] using hkk) (by simpa [l,hjv] using hjj)
    rw [hek,hjv] at hr
    have hvlog := hr.trans hcform
    simpa [historyAddress,historyOffset,l,hhead,hek,hjv] using hvlog
  · intro hon h k j hhh hkk hjj
    have hkp := lt_of_lt_of_le hkk hk
    have hjp := lt_of_lt_of_le hjj hv
    obtain ⟨plan,hplan,idx,e,f,hhead,hek,hjv⟩ := LogicalCoverage.coverage p heads h k j u0 hp hhh hkp hjp
    let l := RebasedInit.decoded p plan.base (u0.withGridIndex idx)
    have hr := (ho plan hplan idx).final hon e f ⟨by simpa [hek] using hkp,by simpa [hjv] using hjp⟩
    have hcform := HostStaging.closed_eq p l a (x l.head) T K V gateK hasSeed
      (Materialize.staged p l T K V gateK x hp hk hv) p.NT _ _ le_rfl
      (by simpa [l,hek] using hkk) (by simpa [l,hjv] using hjj)
    rw [hek,hjv] at hr
    have hvlog := hr.trans hcform
    simpa [stateAddress,finalOffset,l,hhead,hek,hjv] using hvlog

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.DirectRealWhole
