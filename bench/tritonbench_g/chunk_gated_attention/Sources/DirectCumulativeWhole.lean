import bench.tritonbench_g.chunk_gated_attention.Sources.DirectCumulativeSequence
import bench.tritonbench_g.chunk_gated_attention.Sources.AutotunePlan
import bench.tritonbench_g.chunk_gated_attention.Sources.LogicalCoverage

/-! Complete fwd_pre grid: actual BS16 outer batching, selected BS16/32/64
sizes and guarded dummies, successive finite memories, all logical prefixes,
and complete output-allocation frame. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.DirectCumulativeWhole
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealCumProgram RebasedCumExecution RebasedCumAccess
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

def total (p : Layout) (heads : Nat) := heads*(p.T/p.BT)*(p.S/p.BS)
def maximum (p : Layout) (heads : Nat) := heads*(p.T/p.BT)*(p.S/16)
def plans (p : Layout) (heads : Nat) := AutotunePlan.plans (maximum p heads) (total p heads)

theorem total_le_maximum (p : Layout) (heads : Nat) (hbs : p.BS=16 ∨ p.BS=32 ∨ p.BS=64) :
    total p heads ≤ maximum p heads := by
  have h : p.S/p.BS≤p.S/16 := by rcases hbs with h|h|h <;> rw [h] <;> omega
  exact Nat.mul_le_mul_left _ h

theorem coverage (p : Layout) (heads h t f : Nat) (u : BlockState)
    (hp : CumulativeHost.Canonical p) (hbs : p.BS=16 ∨ p.BS=32 ∨ p.BS=64)
    (hh : h<heads) (ht : t<p.T) (hf : f<p.S) :
    ∃plan∈plans p heads, ∃idx : GridIndex (Batch.grid plan.size), ∃i : TileIndex [p.BT,p.BS],
      enabled p heads plan.base (u.withGridIndex idx) ∧
      (decoded p plan.base (u.withGridIndex idx)).head=h ∧
      chunk p plan.base (u.withGridIndex idx)*p.BT+i.1.val=t ∧
      (decoded p plan.base (u.withGridIndex idx)).feature*p.BS+i.2.1.val=f := by
  have hft : f/p.BS<p.S/p.BS := (Nat.div_lt_iff_lt_mul hp.bs).2 (by rwa [hp.feature_full])
  have htt : t/p.BT<p.T/p.BT := (Nat.div_lt_iff_lt_mul hp.bt).2 (by rwa [hp.time_full])
  let n := Grid.encode (p.S/p.BS) (p.T/p.BT) (f/p.BS) (t/p.BT) h
  have hn : n<total p heads := Grid.encode_bounded hft htt hh
  obtain ⟨plan,hplan,hb,hsz⟩ := AutotunePlan.covers _ _ n (total_le_maximum p heads hbs) hn
  let li : Fin plan.size := ⟨n-plan.base,by omega⟩
  let idx := LogicalCoverage.index li
  have hlin : linear plan.base (u.withGridIndex idx)=n := by
    change n-plan.base+plan.base=n
    omega
  have hd := Grid.decode_encode (z := h) hft htt
  have hfd : (decoded p plan.base (u.withGridIndex idx)).feature=f/p.BS := by
    simpa only [decoded,hlin,Grid.decodeX] using hd.1
  have htd : chunk p plan.base (u.withGridIndex idx)=t/p.BT := by
    simpa only [RebasedCumExecution.chunk,hlin,Grid.decodeY] using hd.2.1
  have hhd : (decoded p plan.base (u.withGridIndex idx)).head=h := by
    simpa only [decoded,hlin,Grid.decodeZ] using hd.2.2
  refine ⟨plan,hplan,idx,(⟨t%p.BT,Nat.mod_lt _ hp.bt⟩,⟨f%p.BS,Nat.mod_lt _ hp.bs⟩,PUnit.unit),?_,hhd,?_,?_⟩
  · simpa only [enabled,hlin,total] using hn
  · simp only [htd]
    simpa [Nat.mul_comm] using Nat.div_add_mod t p.BT
  · simp only [hfd]
    simpa [Nat.mul_comm] using Nat.div_add_mod f p.BS

theorem executes (p : Layout) (heads T S : Nat) (s o : RegionName) (swapped : Bool) (dtype out_dtype : FloatDType)
    (x : Nat → Nat → Nat → ℝ) (hp : CumulativeHost.Canonical p)
    (hbs : p.BS=16 ∨ p.BS=32 ∨ p.BS=64) (hT : T≤p.T) (hS : S≤p.S)
    (hsep : s≠o)
    (u : BlockState) (hx : ∀ off, u.readMem s off=CumulativeHost.materialized p T S x off) :
    ∃v, DirectCumulativeSequence.Runs p heads s o swapped dtype out_dtype (plans p heads)
        {u with undef := fun _ _ => 0} v ∧
      (∀h t f, h<heads → t<T → f<S →
        v.readMem o (h*p.headStride+t*p.timeStride+f*p.channelStride) =
          Host.chunkPrefix p.BT (x h) t f) ∧
      BlockState.WriteWithin (DirectCumulativeSequence.allWrites p heads (plans p heads) o u) u v := by
  let u0 : BlockState := {u with undef := fun _ _ => 0}
  obtain ⟨v,hrun,hout,hframe,hu⟩ := DirectCumulativeSequence.executes p heads s o swapped dtype out_dtype
    (CumulativeHost.materialized p T S x) hp hsep (plans p heads)
    (AutotunePlan.valid _ _) (AutotunePlan.ordered _ _ (total_le_maximum p heads hbs)) u0 rfl hx
  refine ⟨v,hrun,?_,hframe⟩
  intro h t f hh ht hf
  have htp := lt_of_lt_of_le ht hT
  have hfp := lt_of_lt_of_le hf hS
  obtain ⟨plan,hplan,idx,i,ha,hhead,hti,hfi⟩ := coverage p heads h t f u0 hp hbs hh htp hfp
  have hr := hout plan hplan idx ha i
  have hex := CumulativeHost.expected_eq p heads T S (decoded p plan.base (u0.withGridIndex idx))
    (chunk p plan.base (u0.withGridIndex idx)) hp (CumulativeHost.decoded_valid p heads plan.base (u0.withGridIndex idx) ha)
    x i (by simpa [hti] using ht) (by simpa [hfi] using hf)
  have hlog := hr.trans hex
  simpa [q,CumulativeHost.local_offset,hhead,hti,hfi] using hlog

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.DirectCumulativeWhole
