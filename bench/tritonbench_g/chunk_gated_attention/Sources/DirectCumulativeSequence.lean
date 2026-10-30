import bench.tritonbench_g.chunk_gated_attention.Sources.DirectCumulativeBatch
import bench.tritonbench_g.chunk_gated_attention.Sources.LaunchPlan

/-! Sequential cumulative autotune batches, with actual successive flat
memories, preserved staged raw gates, readback and complete memory frame. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.DirectCumulativeSequence
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealCumProgram RebasedCumExecution RebasedCumAccess
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 16000

def allWrites (p : Layout) (heads : Nat) (plans : List LaunchPlan.Item) (o : RegionName) (u : BlockState) : WriteFootprint :=
  fun cell => ∃ plan ∈ plans, DirectCumulativeBatch.writes p heads plan.base plan.size o u cell

inductive Runs (p : Layout) (heads : Nat) (s o : RegionName)
    (swapped : Bool) (dtype out_dtype : FloatDType) : List LaunchPlan.Item → BlockState → BlockState → Prop
  | nil (u) : Runs p heads s o swapped dtype out_dtype [] u u
  | cons (plan rest u mid v)
      (head : Nonempty (Kernel.GridLaunchedOrdinary
        (RebasedCumProgram.source p s o heads plan.base swapped dtype out_dtype).eraseDType.toAlgKernel
        (Batch.grid plan.size) u mid))
      (tail : Runs p heads s o swapped dtype out_dtype rest mid v) :
      Runs p heads s o swapped dtype out_dtype (plan::rest) u v

theorem input_frame (p : Layout) (heads : Nat) (s o : RegionName) (plan : LaunchPlan.Item)
    (u v : BlockState) (hsep : s≠o)
    (hf : BlockState.WriteWithin (DirectCumulativeBatch.writes p heads plan.base plan.size o u) u v) :
    ∀ off, v.readMem s off=u.readMem s off := by
  intro off
  unfold BlockState.readMem
  rw [← hf s off (by rintro ⟨idx,ha,hr,_⟩; exact hsep hr)]

theorem separated_writes (p : Layout) (heads : Nat) (o : RegionName)
    (hp : CumulativeHost.Canonical p) (a b : LaunchPlan.Item) (hab : a.base+a.size≤b.base) (u v : BlockState) :
    WriteFootprint.disjoint (DirectCumulativeBatch.writes p heads a.base a.size o u)
      (DirectCumulativeBatch.writes p heads b.base b.size o v) := by
  intro cell ha hb
  rcases ha with ⟨i,hi⟩
  rcases hb with ⟨j,hj⟩
  have he := CumulativeOwnership.owner p heads a.base b.base o (u.withGridIndex i) (v.withGridIndex j) hp cell hi hj
  simp only [linear,Batch.pid_lane] at he
  have hia := Batch.lane_lt i
  have hjb := Batch.lane_lt j
  omega

theorem executes (p : Layout) (heads : Nat) (s o : RegionName) (swapped : Bool) (dtype out_dtype : FloatDType) (a : Nat → ℝ)
    (hp : CumulativeHost.Canonical p) (hsep : s≠o)
    (plans : List LaunchPlan.Item) (hv : ∀ plan ∈ plans, 0<plan.size ∧ plan.size≤Grid.limit)
    (hord : plans.Pairwise (fun p q => p.base+p.size≤q.base))
    (u : BlockState) (hu : u.undef=fun _ _ => 0) (hx : ∀ off, u.readMem s off=a off) :
    ∃ v, Runs p heads s o swapped dtype out_dtype plans u v ∧
      (∀ plan ∈ plans, DirectCumulativeBatch.Outputs p heads plan.base plan.size o a u v) ∧
      BlockState.WriteWithin (allWrites p heads plans o u) u v ∧ v.undef=fun _ _ => 0 := by
  induction plans generalizing u with
  | nil => exact ⟨u,.nil u,by simp,by intro _ _ _; rfl,hu⟩
  | cons plan rest ih =>
    have hplan := hv plan List.mem_cons_self
    have hrest := fun q hq => hv q (List.mem_cons_of_mem _ hq)
    obtain ⟨hbefore,hordered⟩ := List.pairwise_cons.mp hord
    obtain ⟨mid,hrun,hom,hfm,hum⟩ := DirectCumulativeBatch.executes p heads plan.base plan.size s o swapped dtype out_dtype
      a u hp hplan.1 hplan.2 hx
    obtain ⟨v,hruns,hov,hfv,huv⟩ := ih hrest hordered mid (hum.trans hu)
      (fun off => (input_frame p heads s o plan u mid hsep hfm off).trans (hx off))
    refine ⟨v,.cons plan rest u mid v hrun hruns,?_,?_,huv⟩
    · intro q hq
      rcases List.mem_cons.mp hq with heq|hqr
      · subst q
        intro idx hactive i
        have hw : DirectCumulativeBatch.writes p heads plan.base plan.size o u
            (o,RealCumExecution.offset (q p plan.base (u.withGridIndex idx)) localLaunch 0 i) :=
          ⟨idx,hactive,rfl,i,all_active p _ _ i,rfl⟩
        have hm := hfv o _ (by
          rintro ⟨other,hother,hwrite⟩
          exact separated_writes p heads o hp plan other (hbefore other hother) u mid _ hw hwrite)
        exact (congrArg (fun cell : MemCell => match cell.readAs .real with | some (some x) => x | _ => 0) hm).symm.trans (hom idx hactive i)
      · exact hov q hqr
    · intro rg off hn
      apply (hfm rg off (fun h => hn ⟨plan,List.mem_cons_self,h⟩)).trans
      apply hfv rg off
      rintro ⟨q,hq,hw⟩
      exact hn ⟨q,List.mem_cons_of_mem _ hq,hw⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.DirectCumulativeSequence
