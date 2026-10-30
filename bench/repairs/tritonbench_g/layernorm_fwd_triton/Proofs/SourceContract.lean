import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.SourceComplete
namespace VeriTile.Bench.Repairs.LayernormFwdTriton.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 22000

theorem fold_preserves {β : Type} (update : Nat → β → β) (P : β → Prop)
    (start stop stride : Nat) (a : β)
    (hstep : ∀ off b, start ≤ off → P b → P (update off b)) (h : P a) :
    P (foldRange update start stop stride a) := by
  by_cases hz : stride=0
  · simpa [foldRange,hz] using h
  · by_cases hc : start<stop
    · rw [foldRange]
      simp only [hz,hc,if_false,if_true]
      apply fold_preserves update P (start+stride) stop stride (update start a)
      · intro off b ho hb
        exact hstep off b (by omega) hb
      · exact hstep start a (by omega) h
    · simpa [foldRange,hz,hc] using h
termination_by stop-start
decreasing_by omega

/-- Only active lanes need distinct addresses. This admits zero/huge strides
on singleton dimensions without adding a production guard. -/
theorem masked_readback {α : Type} (r : RegionName)
    (off : TileIndex shape → Nat) (val : TileIndex shape → Cell α)
    (active : TileIndex shape → Prop) [DecidablePred active] (s : State α)
    (i : TileIndex shape) (hi : active i)
    (hinj : ∀ a b, active a → active b → off a=off b → a=b) :
    ((TileShape.allIndices shape).foldl
      (fun t k => if active k then t.write r (off k) (val k) else t) s).mem r (off i)=val i := by
  rw [State.masked_scatter_eq_filter]
  have mem : i∈(TileShape.allIndices shape).filter (fun i => decide (active i)) := by
    simp [TileShape.mem_allIndices,hi]
  obtain ⟨before,after,hl⟩ := List.append_of_mem mem
  have nd := (TileShape.allIndices_nodup shape).filter (fun i => decide (active i))
  rw [hl,List.nodup_append,List.nodup_cons] at nd
  rw [hl,List.foldl_append,List.foldl_cons,State.scatter_frame]
  · exact State.write_same ..
  · right
    intro k hk he
    have kmem : k∈(TileShape.allIndices shape).filter (fun i => decide (active i)) := by
      rw [hl]; simp only [List.mem_append,List.mem_cons]; exact .inr (.inr hk)
    have ha : active k := of_decide_eq_true (List.mem_filter.mp kmem).2
    exact nd.2.1.1 ((hinj i k hi ha he) ▸ hk)

theorem outputStep_readback {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (mem : RegionName → Nat → Cell α) (start n : Nat)
    (hs : start ≤ n) (hb : n < start+p.block) (hn : n<p.columns)
    (hinj : ∀ a b, a<p.columns → b<p.columns → a*p.yDimStride=b*p.yDimStride → a=b) :
    outputStep M p s start mem p.y (yBase p s+n*p.yDimStride) = .mk .fp32 (yValue M p s n) := by
  rw [← scatter_memory M p s {s with mem:=mem} start]
  let i : TileIndex [p.block] := (⟨n-start,by omega⟩,PUnit.unit)
  have hi : start+i.1.val=n := by dsimp [i]; omega
  have h := masked_readback p.y
    (fun j : TileIndex [p.block] => yBase p s+(start+j.1.val)*p.yDimStride)
    (fun j => Cell.mk .fp32 (yValue M p s (start+j.1.val)))
    (fun j => start+j.1.val<p.columns) {s with mem:=mem} i (by simpa [hi] using hn) ?_
  · simpa only [scatter,hi] using h
  · intro a b ha hb hab
    have hval : a.1.val=b.1.val := Nat.add_left_cancel (hinj _ _ ha hb (Nat.add_left_cancel hab))
    rcases a with ⟨a,⟨⟩⟩
    rcases b with ⟨b,⟨⟩⟩
    simpa [Fin.ext_iff] using hval

theorem outputStep_before {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (mem : RegionName → Nat → Cell α) (start n : Nat) (h : n<start) (hn : n<p.columns)
    (hinj : ∀ a b, a<p.columns → b<p.columns → a*p.yDimStride=b*p.yDimStride → a=b) :
    outputStep M p s start mem p.y (yBase p s+n*p.yDimStride) = mem p.y (yBase p s+n*p.yDimStride) := by
  rw [← scatter_memory M p s {s with mem:=mem} start]
  apply State.masked_scatter_frame
  right
  intro i _ hi heq
  have := hinj n (start+i.1.val) hn hi (Nat.add_left_cancel heq)
  omega

theorem fold_readback {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (mem : RegionName → Nat → Cell α) (start n : Nat)
    (hb : 0<p.block) (hs : start≤n) (hn : n<p.columns)
    (hinj : ∀ a b, a<p.columns → b<p.columns → a*p.yDimStride=b*p.yDimStride → a=b) :
    foldRange (outputStep M p s) start p.columns p.block mem p.y (yBase p s+n*p.yDimStride) =
      .mk .fp32 (yValue M p s n) := by
  have hz : p.block≠0 := by omega
  have hc : start<p.columns := by omega
  rw [foldRange]
  simp only [hz,hc,if_false,if_true]
  by_cases hnext : n<start+p.block
  · apply fold_preserves (outputStep M p s)
      (fun mem => mem p.y (yBase p s+n*p.yDimStride)=Cell.mk .fp32 (yValue M p s n))
      (start+p.block) p.columns p.block (outputStep M p s start mem)
    · intro off b ho hprev
      rw [outputStep_before M p s b off n (by omega) hn hinj]
      exact hprev
    · exact outputStep_readback M p s mem start n hs hnext hn hinj
  · exact fold_readback M p s (outputStep M p s start mem) (start+p.block) n hb (by omega) hn hinj
termination_by p.columns-start
decreasing_by omega

theorem reference_readback {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (hb : 0<p.block) (n : Nat) (hn : n<p.columns)
    (hinj : ∀ a b, a<p.columns → b<p.columns → a*p.yDimStride=b*p.yDimStride → a=b) :
    referenceMem M p s p.y (yBase p s+n*p.yDimStride) = .mk .fp32 (yValue M p s n) :=
  fold_readback M p s s.mem 0 n hb (Nat.zero_le _) hn hinj

theorem reference_frame {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (r : RegionName) (o : Nat) (h : Outside p s r o) :
    referenceMem M p s r o=s.mem r o := by
  apply fold_preserves (outputStep M p s) (fun mem => mem r o=s.mem r o)
  · intro off b _ hb
    rw [outputStep_frame M p s b off r o h]
    exact hb
  · rfl

#axiomsClean fold_preserves
#axiomsClean outputStep_readback
#axiomsClean outputStep_before
#axiomsClean fold_readback
#axiomsClean reference_readback
#axiomsClean reference_frame
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.Source
