import bench.tritonbench_g.int8_dequant_matmul.Proofs.QuoteExecution

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOStoreFold
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000

/-- A finite mathematical fold used only to analyze the quotation's actual
epilogue writes. Each update reads its own current cell. -/
def updates {α ι : Type} (active : ι→Prop) [DecidablePred active]
    (offset : ι→Nat) (combine : Cell α→ι→Cell α) (region : RegionName)
    (lanes : List ι) (memory : RegionName→Nat→Cell α) :=
  lanes.foldl (fun mem i=>if active i then
    Quote.writeCell mem region (offset i) (combine (mem region (offset i)) i) else mem) memory

theorem frame {α ι : Type} (active : ι→Prop) [DecidablePred active]
    (offset : ι→Nat) (combine : Cell α→ι→Cell α) (region : RegionName)
    (lanes : List ι) (memory : RegionName→Nat→Cell α) (r : RegionName) (o : Nat)
    (miss : r≠region ∨ ∀ i∈lanes,active i → o≠offset i) :
    updates active offset combine region lanes memory r o=memory r o := by
  induction lanes generalizing memory with
  | nil=>rfl
  | cons i rest ih=>
    have tail : r≠region ∨ ∀ j∈rest,active j → o≠offset j := by
      rcases miss with h|h
      · exact Or.inl h
      · exact Or.inr (fun j hj=>h j (List.mem_cons_of_mem _ hj))
    simp only [updates,List.foldl_cons]
    change updates active offset combine region rest _ r o=_
    rw [ih _ tail]
    by_cases enabled : active i
    · simp only [enabled,ite_true,Quote.writeCell]
      apply if_neg
      rintro ⟨same,eq⟩
      rcases miss with h|h
      · exact h same
      · exact h i (by simp) enabled eq
    · simp only [enabled,ite_false]

theorem read {α ι : Type} (active : ι→Prop) [DecidablePred active]
    (offset : ι→Nat) (combine : Cell α→ι→Cell α) (region : RegionName)
    (lanes : List ι) (memory : RegionName→Nat→Cell α)
    (nodup : lanes.Nodup) (i : ι) (present : i∈lanes) (enabled : active i)
    (unique : ∀ j∈lanes,active j → offset i=offset j → i=j) :
    updates active offset combine region lanes memory region (offset i)=
      combine (memory region (offset i)) i := by
  obtain ⟨before,after,equal⟩:=List.append_of_mem present
  have nd:=nodup
  rw [equal,List.nodup_append,List.nodup_cons] at nd
  have noAfter : i∉after := nd.2.1.1
  have noBefore : i∉before := by
    intro member
    exact nd.2.2 i member i (by simp) rfl
  have missBefore : ∀ j∈before,active j → offset i≠offset j := by
    intro j member enabledJ same
    have all : j∈lanes := by rw [equal]; simp [member]
    exact noBefore (unique j all enabledJ same ▸ member)
  have missAfter : ∀ j∈after,active j → offset i≠offset j := by
    intro j member enabledJ same
    have all : j∈lanes := by rw [equal]; simp [member]
    exact noAfter (unique j all enabledJ same ▸ member)
  rw [equal]
  simp only [updates,List.foldl_append,List.foldl_cons,enabled,ite_true]
  change updates active offset combine region after
    (Quote.writeCell (updates active offset combine region before memory) region (offset i)
      (combine (updates active offset combine region before memory region (offset i)) i))
      region (offset i)=_
  rw [frame _ _ _ _ _ _ _ _ (Or.inr missAfter)]
  simp only [Quote.writeCell,ite_true,and_self]
  rw [frame _ _ _ _ _ _ _ _ (Or.inr missBefore)]

#axiomsClean frame
#axiomsClean read
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOStoreFold
