import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.OrderedReference
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.IOBindings

namespace VeriTile.Bench.Repairs.LayerNormLiger.Ordered
open VeriTile.Triton FP.Structural LayerIO
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (SumPrimitive)
noncomputable section
variable {α : Type} [Inhabited α]
set_option linter.unusedSectionVars false

def Match (initial actual : State α) (w : Writes α) : Prop :=
  ∀ r o, actual.mem r o=match w r o with | none => initial.mem r o | some x => .mk .fp32 x

theorem match_initial (s : State α) : Match s s (fun _ _ => none) := by intro r o; rfl

theorem match_put {s t : State α} {w : Writes α} (h : Match s t w)
    (r : RegionName) (o : Nat) (x : α) : Match s (t.write r o (.mk .fp32 x)) (put w r o x) := by
  intro q i
  by_cases hit : q=r ∧ i=o
  · simp [put,State.write,hit]
  · simpa [put,State.write,hit] using h q i

theorem match_fold {B : Nat} (s : State α) (r : RegionName) (offset : TileIndex [B] → Nat)
    (value : TileIndex [B] → α) (n : Nat) (l : List (TileIndex [B])) (t : State α) (w : Writes α)
    (h : Match s t w) : Match s
      (l.foldl (fun t i => if i.1.val<n then t.write r (offset i) (.mk .fp32 (value i)) else t) t)
      (l.foldl (fun t i => if i.1.val<n then put t r (offset i) (value i) else t) w) := by
  induction l generalizing t w with
  | nil => exact h
  | cons i rest ih =>
    simp only [List.foldl_cons]
    by_cases hi : i.1.val<n
    · simp only [if_pos hi]
      exact ih _ _ (match_put h r (offset i) (value i))
    · simp only [if_neg hi]
      exact ih _ _ h

theorem match_scatter {B : Nat} {s t : State α} {w : Writes α} (h : Match s t w)
    (r : RegionName) (offset : TileIndex [B] → Nat) (value : TileIndex [B] → α) (n : Nat) :
    Match s ((TileShape.allIndices [B]).foldl
      (fun t i => if i.1.val<n then t.write r (offset i) (.mk .fp32 (value i)) else t) t)
      (scatter B n r offset value w) := match_fold s r offset value n _ t w h

theorem match_read {s t : State α} {w : Writes α} (h : Match s t w)
    (base : RegionName → Nat → α) (r : RegionName) (o : Nat)
    (bound : (s.mem r o).read .fp32=base r o) :
    (t.mem r o).read .fp32=read base w r o := by
  rw [h]
  cases hw : w r o <;> simp [read,hw,bound]

def Present (w : Writes α) (r : RegionName) (o : Nat) : Prop := ∃ x,w r o=some x

theorem put_present (w : Writes α) (r : RegionName) (o : Nat) (x : α) :
    Present (put w r o x) r o := ⟨x,by simp [put]⟩
theorem put_preserves (w : Writes α) (r : RegionName) (o : Nat) (x : α)
    (q : RegionName) (i : Nat) (h : Present w q i) : Present (put w r o x) q i := by
  by_cases hit : q=r ∧ i=o
  · exact ⟨x,by simp [put,hit]⟩
  · simpa [Present,put,hit] using h

theorem fold_preserves {B : Nat} (n : Nat) (r : RegionName) (offset : TileIndex [B] → Nat)
    (value : TileIndex [B] → α) (l : List (TileIndex [B])) (w : Writes α)
    (q : RegionName) (i : Nat) (h : Present w q i) :
    Present (l.foldl (fun t j => if j.1.val<n then put t r (offset j) (value j) else t) w) q i := by
  induction l generalizing w with
  | nil => exact h
  | cons j rest ih =>
    simp only [List.foldl_cons]
    apply ih
    by_cases hj : j.1.val<n
    · simp only [if_pos hj]
      exact put_preserves _ _ _ _ _ _ h
    · simpa only [if_neg hj] using h

theorem scatter_preserves {B : Nat} (n : Nat) (r : RegionName) (offset : TileIndex [B] → Nat)
    (value : TileIndex [B] → α) (w : Writes α) (q : RegionName) (i : Nat) (h : Present w q i) :
    Present (scatter B n r offset value w) q i := fold_preserves n r offset value _ w q i h

theorem scatter_present {B : Nat} (n : Nat) (r : RegionName) (offset : TileIndex [B] → Nat)
    (value : TileIndex [B] → α) (w : Writes α) (i : TileIndex [B]) (hi : i.1.val<n) :
    Present (scatter B n r offset value w) r (offset i) := by
  obtain ⟨before,after,eq⟩ := List.append_of_mem (TileShape.mem_allIndices [B] i)
  unfold scatter
  rw [eq,List.foldl_append]
  simp only [List.foldl_cons,if_pos hi]
  exact fold_preserves n r offset value after _ r (offset i) (put_present _ _ _ _)

theorem match_output {s t : State α} {w : Writes α} (h : Match s t w)
    (r : RegionName) (o : Nat) (hp : Present w r o) :
    t.mem r o=.mk .fp32 ((w r o).getD default) := by
  obtain ⟨x,hx⟩ := hp
  simp [h r o,hx]

theorem b_initial_eq (M : Algebra α) (R : SumPrimitive α) (p : Backward.Parameters) (pid : Nat)
    (v : ChannelValues α (bInputs p)) (s : State α)
    (hin : (backwardIO M R p pid).InputBinding v s)
    (r : RegionName) (o : Nat) (slot : BSlot p) (hit : bAt p pid v r o slot) :
    (s.mem r o).read .fp32=bInitial p pid v r o := by
  have existsSlot : ∃ slot,bAt p pid v r o slot := ⟨slot,hit⟩
  rw [bInitial,dif_pos existsSlot]
  obtain ⟨active,hr,ho⟩ := Classical.choose_spec existsSlot
  apply (congrArg₂ (fun r o => (s.mem r o).read .fp32) hr ho).trans
  have bound : ∀ (slot : BSlot p), (bMemory α p pid).inputMask slot.1 v slot.2 →
      (s.mem (bInputs p)[slot.1].buf ((bMemory α p pid).read slot.1 v slot.2)).read .fp32 =
        bScalar p v slot.1 slot.2 := by
    intro ⟨k,i⟩ ha
    have h := hin k i ha
    fin_cases k <;> simpa [backwardIO,bMemory,bInputs,bScalar,mapValue] using h
  exact bound _ active

end
end VeriTile.Bench.Repairs.LayerNormLiger.Ordered
