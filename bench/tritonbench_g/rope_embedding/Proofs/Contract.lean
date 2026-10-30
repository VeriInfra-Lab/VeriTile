import bench.tritonbench_g.rope_embedding.Proofs.Group
import bench.tritonbench_g.rope_embedding.Proofs.Schedule
namespace VeriTile.Bench.Repairs.RopeEmbedding.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.RotaryEmbNopad
noncomputable section
set_option maxHeartbeats 3000000

theorem offset_column_bound (p : Parameters) (side : Bool) (i : Lane p) (hi : active p i) :
    i.1.val+(if side then p.dimension/2 else 0)<p.dimension := by
  unfold active at hi
  cases side <;> simp only [Bool.false_eq_true,↓reduceIte] <;> omega

theorem offset_head_injective (p : Parameters) (row a b : Nat) (side other : Bool)
    (i j : Lane p) (hi : active p i) (hj : active p j)
    (he : offset p row a side i=offset p row b other j) : a=b := by
  have hd : 0<p.dimension := by have := offset_column_bound p side i hi; omega
  have hc := offset_column_bound p side i hi
  have hc' := offset_column_bound p other j hj
  have eq : a*p.dimension+(i.1.val+if side then p.dimension/2 else 0)=
      b*p.dimension+(j.1.val+if other then p.dimension/2 else 0) := by
    apply Nat.add_left_cancel (n:=row*p.rowStride)
    simpa only [offset,Nat.add_assoc] using he
  have div (k z : Nat) (hz : z<p.dimension) : (k*p.dimension+z)/p.dimension=k := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hd,Nat.div_eq_of_lt hz]
    simp
  have h := congrArg (fun n => n/p.dimension) eq
  simpa only [div _ _ hc,div _ _ hc'] using h

theorem writesApart (p : Parameters) (row head : Nat) : Pair.WritesApart (active p) (offset p row head) := by
  constructor
  · intro side a b _ _ he
    have hi : a.1.val=b.1.val := by simp only [offset] at he; omega
    exact Prod.ext (Fin.ext hi) (Subsingleton.elim _ _)
  · intro a b ha hb he
    simp only [offset,Bool.false_eq_true,↓reduceIte] at he
    unfold active at ha hb
    omega

theorem written_readback {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (row head : Nat) (cs ss : Lane p→α) (s : State α) (side : Bool) (i : Lane p) (hi : active p i) :
    (written M p row head cs ss s).mem p.input (offset p row head side i)=
      .mk .real (value M p row head cs ss s side i) :=
  Pair.readback _ _ _ _ _ (writesApart p row head) side i hi

theorem written_frame {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (row head : Nat) (cs ss : Lane p→α) (s : State α) (region : RegionName) (o : Nat)
    (outside : region≠p.input ∨ ∀ side i,active p i → o≠offset p row head side i) :
    (written M p row head cs ss s).mem region o=s.mem region o :=
  Pair.frame _ _ _ _ _ _ _ outside

theorem heads_frame {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (row : Nat) (cs ss : Lane p→α) (start count : Nat) (s : State α) (region : RegionName) (o : Nat)
    (outside : region≠p.input ∨ ∀ h,start≤h → h<start+count →
      ∀ side i,active p i → o≠offset p row h side i) :
    (heads M p row cs ss start count s).mem region o=s.mem region o := by
  induction count generalizing start s with
  | zero => rfl
  | succ count ih =>
    change (heads M p row cs ss (start+1) count (written M p row start cs ss s)).mem region o=_
    rw [ih (start+1) _ (outside.imp id (fun h k hlo hhi => h k (by omega) (by omega)))]
    exact written_frame M p row start cs ss s region o
      (outside.imp id (fun h => h start (by omega) (by omega)))

theorem value_mem_congr {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (row head : Nat) (cs ss : Lane p→α) (s t : State α) (side : Bool) (i : Lane p)
    (same : ∀ part,s.mem p.input (offset p row head part i)=t.mem p.input (offset p row head part i)) :
    value M p row head cs ss s side i=value M p row head cs ss t side i := by
  simp only [value,same]

theorem heads_readback {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (row : Nat) (cs ss : Lane p→α) (start count : Nat) (s : State α)
    (h : Nat) (hlo : start≤h) (hhi : h<start+count) (side : Bool) (i : Lane p) (hi : active p i) :
    (heads M p row cs ss start count s).mem p.input (offset p row h side i)=
      .mk .real (value M p row h cs ss s side i) := by
  induction count generalizing start s with
  | zero => omega
  | succ count ih =>
    change (heads M p row cs ss (start+1) count (written M p row start cs ss s)).mem p.input
      (offset p row h side i)=_
    by_cases hfirst : h=start
    · subst h
      rw [heads_frame M p row cs ss (start+1) count _ p.input _ (Or.inr ?_)]
      · exact written_readback M p row start cs ss s side i hi
      · intro k hk _ other j hj equal
        have := offset_head_injective p row start k side other i j hi hj equal
        omega
    · rw [ih (start+1) _ (by omega) (by omega)]
      congr 1
      apply value_mem_congr
      intro part
      apply written_frame
      right
      intro other j hj equal
      exact hfirst (offset_head_injective p row h start part other i j hi hj equal)

theorem complete_correctness {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (backward : Bool) (s : State α) (group : s.pids 1*4≤p.heads) :
    ∃ final,FP.Structural.exec M (source p backward) s=some final ∧
      (∀ h,s.pids 1*4≤h → h<min (s.pids 1*4+4) p.heads → ∀ side i,active p i →
        final.mem p.input (offset p (s.pids 0) h side i)=.mk .real
          (value M p (s.pids 0) h (coefficient M p s p.cosine p.cosineStride)
            (fun j => if backward then M.unary (some .fp32) (.neg .real) (coefficient M p s p.sine p.sineStride j)
              else coefficient M p s p.sine p.sineStride j) s side i)) ∧
      (∀ region o,(region≠p.input ∨ ∀ h,s.pids 1*4≤h → h<min (s.pids 1*4+4) p.heads →
        ∀ side i,active p i → o≠offset p (s.pids 0) h side i) → final.mem region o=s.mem region o) := by
  obtain ⟨final,he,hm⟩ := complete_execution M p backward s group
  have hs : s.pids 1*4≤min (s.pids 1*4+4) p.heads := le_min (by omega) group
  refine ⟨final,he,?_,?_⟩
  · intro h hlo hhi side i hi
    rw [hm]
    exact heads_readback M p (s.pids 0) _ _ _ _ s h hlo (by simpa [Nat.add_sub_of_le hs] using hhi) side i hi
  · intro region o outside
    rw [hm]
    exact heads_frame M p (s.pids 0) _ _ _ _ s region o
      (outside.imp id (fun h k hlo hhi => h k hlo (by simpa [Nat.add_sub_of_le hs] using hhi)))

#axiomsClean complete_correctness
#axiomsClean heads_readback
#axiomsClean heads_frame
end
end VeriTile.Bench.Repairs.RopeEmbedding.Source
