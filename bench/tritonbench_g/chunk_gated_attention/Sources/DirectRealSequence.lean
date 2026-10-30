import bench.tritonbench_g.chunk_gated_attention.Sources.DirectRealBatch
import bench.tritonbench_g.chunk_gated_attention.Sources.LaunchPlan

/-! Actual sequential host launches over finite allocations. Each later batch
starts with the preceding batch's memory. Input snapshots are preserved because
all kernel output allocations are separate from staged input allocations. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.DirectRealSequence
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealHMath RealHProgram
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000

structure InputSeparate (r : Regions) (storeFinal : Bool) : Prop where
  key : r.key ≠ r.history ∧ (storeFinal = Bool.true → r.key ≠ r.final)
  value : r.value ≠ r.history ∧ (storeFinal = Bool.true → r.value ≠ r.final)
  gate : r.gate ≠ r.history ∧ (storeFinal = Bool.true → r.gate ≠ r.final)
  seed : r.seed ≠ r.history ∧ (storeFinal = Bool.true → r.seed ≠ r.final)

def allWrites (p : Layout) (r : Regions) (plans : List LaunchPlan.Item) (storeFinal : Bool) (u : BlockState) : WriteFootprint :=
  fun cell => ∃ plan ∈ plans, DirectRealBatch.writes p r plan.base plan.size storeFinal u cell

/-- A sequence of actual ordinary flat-grid launches with successive memory. -/
inductive Runs (p : Layout) (r : Regions) (dot gate history : NativeSource.Storage)
    (seed final : FloatDType) (gateK hasSeed storeFinal swapped : Bool) :
    List LaunchPlan.Item → BlockState → BlockState → Prop
  | nil (u) : Runs p r dot gate history seed final gateK hasSeed storeFinal swapped [] u u
  | cons (plan rest u mid v)
      (head : Nonempty (Kernel.GridLaunchedOrdinary
        (RebasedProgram.source p r dot gate history seed final plan.base gateK hasSeed storeFinal swapped).eraseDType.toAlgKernel
        (Batch.grid plan.size) u mid))
      (tail : Runs p r dot gate history seed final gateK hasSeed storeFinal swapped rest mid v) :
      Runs p r dot gate history seed final gateK hasSeed storeFinal swapped (plan::rest) u v

theorem input_frame (p : Layout) (r : Regions) (plan : LaunchPlan.Item) (storeFinal : Bool)
    (u v : BlockState) (hf : BlockState.WriteWithin (DirectRealBatch.writes p r plan.base plan.size storeFinal u) u v)
    (rg : RegionName) (hr : rg≠r.history ∧ (storeFinal=Bool.true → rg≠r.final)) :
    ∀ off, v.readMem rg off = u.readMem rg off := by
  intro off
  unfold BlockState.readMem
  rw [← hf rg off (by
    rintro ⟨idx,hi|hi⟩
    · exact hr.1 hi.1
    · exact hr.2 hi.1 hi.2.1)]

theorem separated_writes (p : Layout) (r : Regions) (storeFinal : Bool) (hp : StagedLayout.Canonical p)
    (hr : storeFinal=Bool.true → r.history≠r.final) (heads : Nat)
    (a b : LaunchPlan.Item) (ha : LaunchPlan.Valid (heads*(p.K/p.BK)*(p.V/p.BV)) a)
    (hb : LaunchPlan.Valid (heads*(p.K/p.BK)*(p.V/p.BV)) b) (hab : a.base+a.size ≤ b.base)
    (u v : BlockState) :
    WriteFootprint.disjoint (DirectRealBatch.writes p r a.base a.size storeFinal u)
      (DirectRealBatch.writes p r b.base b.size storeFinal v) := by
  intro cell hwa hwb
  rcases hwa with ⟨i,hi⟩
  rcases hwb with ⟨j,hj⟩
  have he := Ownership.region_owner p _ _ r storeFinal hp hr cell hi hj
  have htotal : 0 < heads*(p.K/p.BK)*(p.V/p.BV) := lt_of_lt_of_le (lt_of_lt_of_le ha.1 (Nat.le_add_left _ _)) ha.2.2
  have hx : 0 < p.V/p.BV := by by_contra hn; simp [Nat.eq_zero_of_not_pos hn] at htotal
  have hy : 0 < p.K/p.BK := by by_contra hn; simp [Nat.eq_zero_of_not_pos hn] at htotal
  have heq := congrArg (fun l : Launch => Grid.encode (p.V/p.BV) (p.K/p.BK) l.value l.key l.head) he
  change Grid.encode _ _ (Grid.decodeX _ _) (Grid.decodeY _ _ _) (Grid.decodeZ _ _ _) =
    Grid.encode _ _ (Grid.decodeX _ _) (Grid.decodeY _ _ _) (Grid.decodeZ _ _ _) at heq
  rw [Grid.decode_encode_identity hx hy,Grid.decode_encode_identity hx hy,Batch.pid_lane,Batch.pid_lane] at heq
  have hia := Batch.lane_lt i
  have hjb := Batch.lane_lt j
  omega

theorem outputs_frame (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (gateK hasSeed storeFinal : Bool) (u v : BlockState)
    (ho : RealHExecution.Outputs p l r a gateK hasSeed storeFinal u)
    (hf : ∀ rg off, RebasedExecution.footprint p l r storeFinal (rg,off) → u.mem rg off=v.mem rg off) :
    RealHExecution.Outputs p l r a gateK hasSeed storeFinal v := by
  constructor
  · intro c e j hc hact
    have hm := hf r.history _ (Or.inl ⟨rfl,c,e,j,hc,hact,rfl⟩)
    exact (congrArg (fun cell : MemCell => match cell.readAs .real with | some (some x) => x | _ => 0) hm).symm.trans (ho.history c e j hc hact)
  · intro hon e j hact
    have hm := hf r.final (stateAddress p l (keyIndex p l e.val) (valueIndex p l j.val))
      (Or.inr ⟨hon,rfl,e,j,hact,by simp [RealHStore.address,RealHFinal.base,stateAddress]⟩)
    exact (congrArg (fun cell : MemCell => match cell.readAs .real with | some (some x) => x | _ => 0) hm).symm.trans (ho.final hon e j hact)

theorem executes (p : Layout) (heads : Nat) (r : Regions) (dot gate history : NativeSource.Storage)
    (seed final : FloatDType) (gateK hasSeed storeFinal swapped : Bool) (a : Inputs)
    (hp : StagedLayout.Canonical p) (hs : InputSeparate r storeFinal)
    (hfsep : storeFinal=Bool.true → r.history≠r.final)
    (plans : List LaunchPlan.Item)
    (hv : ∀ plan ∈ plans, LaunchPlan.Valid (heads*(p.K/p.BK)*(p.V/p.BV)) plan)
    (hord : plans.Pairwise (fun p q => p.base+p.size ≤ q.base))
    (u : BlockState) (hu : u.undef=fun _ _ => 0)
    (hkey : ∀ off, u.readMem r.key off=a.key off)
    (hvalue : ∀ off, u.readMem r.value off=a.value off)
    (hgate : ∀ off, u.readMem r.gate off=a.gate off)
    (hseed : ∀ off, u.readMem r.seed off=a.seed off) :
    ∃ v, Runs p r dot gate history seed final gateK hasSeed storeFinal swapped plans u v ∧
      (∀ plan ∈ plans, ∀ idx : GridIndex (Batch.grid plan.size),
        RealHExecution.Outputs p (RebasedInit.decoded p plan.base (u.withGridIndex idx)) r a gateK hasSeed storeFinal v) ∧
      BlockState.WriteWithin (allWrites p r plans storeFinal u) u v ∧ v.undef=fun _ _ => 0 := by
  induction plans generalizing u with
  | nil =>
    exact ⟨u,.nil u,by simp,by intro _ _ _; rfl,hu⟩
  | cons plan rest ih =>
    have hplan := hv plan List.mem_cons_self
    have hrest := fun q hq => hv q (List.mem_cons_of_mem _ hq)
    obtain ⟨hbefore,hordered⟩ := List.pairwise_cons.mp hord
    obtain ⟨mid,hrun,hom,hfm,hum⟩ := DirectRealBatch.executes p heads r dot gate history seed final plan.base plan.size
      gateK hasSeed storeFinal swapped a u hp hplan.1 hplan.2.1 hplan.2.2
      ⟨hs.key.1,hs.value.1,hs.gate.1⟩ hfsep hkey hvalue hgate hseed
    obtain ⟨v,hruns,hov,hfv,huv⟩ := ih hrest hordered mid (hum.trans hu)
      (fun off => (input_frame p r plan storeFinal u mid hfm r.key hs.key off).trans (hkey off))
      (fun off => (input_frame p r plan storeFinal u mid hfm r.value hs.value off).trans (hvalue off))
      (fun off => (input_frame p r plan storeFinal u mid hfm r.gate hs.gate off).trans (hgate off))
      (fun off => (input_frame p r plan storeFinal u mid hfm r.seed hs.seed off).trans (hseed off))
    refine ⟨v,.cons plan rest u mid v hrun hruns,?_,?_,huv⟩
    · intro q hq idx
      rcases List.mem_cons.mp hq with heq|hqr
      · subst q
        apply outputs_frame p _ r a gateK hasSeed storeFinal mid v (hom idx)
        intro rg off hw
        apply hfv rg off
        rintro ⟨other,hm,hwo⟩
        exact separated_writes p r storeFinal hp hfsep heads plan other hplan (hrest other hm)
          (hbefore other hm) u mid (rg,off) ⟨idx,hw⟩ hwo
      · exact hov q hqr idx
    · intro rg off hn
      apply (hfm rg off (fun h => hn ⟨plan,List.mem_cons_self,h⟩)).trans
      apply hfv rg off
      rintro ⟨q,hq,hw⟩
      exact hn ⟨q,List.mem_cons_of_mem _ hq,hw⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.DirectRealSequence
