import bench.tritonbench_g.layernorm_fwd_triton.Proofs.Mathematical_FPComplete
namespace VeriTile.Bench.TritonBenchG.LayernormFwdTriton.FPStages
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

theorem outputStep_readback {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (mem : RegionName → Nat → Cell α) (start n : Nat)
    (hs : start ≤ n) (hb : n < start+p.block) (hn : n<p.columns) :
    outputStep M p s start mem p.y (yBase p s+n) = .mk .fp32 (yValue M p s n) := by
  rw [← scatter_memory M p s {s with mem:=mem} start]
  let i : TileIndex [p.block] := (⟨n-start,by omega⟩,PUnit.unit)
  have hi : start+i.1.val=n := by dsimp [i]; omega
  have h := State.masked_scatter_readback p.y
    (fun j : TileIndex [p.block] => yBase p s+(start+j.1.val))
    (fun j => Cell.mk .fp32 (yValue M p s (start+j.1.val)))
    (fun j => start+j.1.val<p.columns) {s with mem:=mem} i (by simpa [hi] using hn) ?_
  · simpa only [scatter,hi] using h
  · intro a b hab
    have hval : a.1.val=b.1.val := Nat.add_left_cancel (Nat.add_left_cancel hab)
    rcases a with ⟨a,⟨⟩⟩
    rcases b with ⟨b,⟨⟩⟩
    simpa [Fin.ext_iff] using hval

theorem outputStep_before {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (mem : RegionName → Nat → Cell α) (start n : Nat) (h : n<start) :
    outputStep M p s start mem p.y (yBase p s+n) = mem p.y (yBase p s+n) := by
  rw [← scatter_memory M p s {s with mem:=mem} start]
  apply State.masked_scatter_frame
  right
  intro i _ _ heq
  have := Nat.add_left_cancel heq
  omega

theorem fold_readback {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (mem : RegionName → Nat → Cell α) (start n : Nat)
    (hb : 0<p.block) (hs : start≤n) (hn : n<p.columns) :
    foldRange (outputStep M p s) start p.columns p.block mem p.y (yBase p s+n) =
      .mk .fp32 (yValue M p s n) := by
  have hz : p.block≠0 := by omega
  have hc : start<p.columns := by omega
  rw [foldRange]
  simp only [hz,hc,if_false,if_true]
  by_cases hnext : n<start+p.block
  · apply fold_preserves (outputStep M p s)
      (fun mem => mem p.y (yBase p s+n)=Cell.mk .fp32 (yValue M p s n))
      (start+p.block) p.columns p.block (outputStep M p s start mem)
    · intro off b ho hprev
      rw [outputStep_before M p s b off n (by omega)]
      exact hprev
    · exact outputStep_readback M p s mem start n hs hnext hn
  · exact fold_readback M p s (outputStep M p s start mem) (start+p.block) n hb (by omega) hn
termination_by p.columns-start
decreasing_by omega

theorem reference_readback {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (hb : 0<p.block) (n : Nat) (hn : n<p.columns) :
    referenceMem M p s p.y (yBase p s+n) = .mk .fp32 (yValue M p s n) :=
  fold_readback M p s s.mem 0 n hb (Nat.zero_le _) hn

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
end VeriTile.Bench.TritonBenchG.LayernormFwdTriton.FPStages
