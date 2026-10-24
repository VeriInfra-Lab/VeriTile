import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.Source

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Source
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1000000

variable {α ι : Type}

def writeMemory (m : RegionName→Nat→Cell α) (r : RegionName) (o : Nat) (v : Cell α) :
    RegionName→Nat→Cell α := fun r' o' => if r'=r ∧ o'=o then v else m r' o'

theorem masked_memory (r : RegionName) (off : ι→Nat) (val : ι→Cell α)
    (active : ι→Prop) [DecidablePred active] (l : List ι) (s : State α) :
    (l.foldl (fun t i => if active i then t.write r (off i) (val i) else t) s).mem=
      l.foldl (fun m i => if active i then writeMemory m r (off i) (val i) else m) s.mem := by
  induction l generalizing s with
  | nil => rfl
  | cons i rest ih =>
    by_cases hi : active i <;> simp only [List.foldl_cons,hi,↓reduceIte,ih]
    rfl

theorem masked_stable (r : RegionName) (off : ι→Nat) (val : ι→Cell α)
    (active : ι→Prop) [DecidablePred active] (l : List ι) (s : State α) (i : ι)
    (before : s.mem r (off i)=val i)
    (same : ∀ j∈l, active j → off j=off i → val j=val i) :
    (l.foldl (fun t j=>if active j then t.write r (off j) (val j) else t) s).mem r (off i)=val i := by
  induction l generalizing s with
  | nil => exact before
  | cons j rest ih =>
    simp only [List.foldl_cons]
    apply ih
    · by_cases hj : active j
      · rw [if_pos hj]
        by_cases he : off j=off i
        · rw [he,State.write_same,same j (by simp) hj he]
        · rw [State.write_other _ _ _ _ _ _ (Or.inr (Ne.symm he))]
          exact before
      · simpa only [if_neg hj] using before
    · intro k hk
      exact same k (List.mem_cons_of_mem _ hk)

theorem masked_readback (r : RegionName) (off : ι→Nat) (val : ι→Cell α)
    (active : ι→Prop) [DecidablePred active] (l : List ι) (s : State α) (i : ι)
    (hi : i∈l) (ha : active i)
    (hinj : ∀ j k, active j → active k → off j=off k → j=k) :
    (l.foldl (fun t j=>if active j then t.write r (off j) (val j) else t) s).mem r (off i)=val i := by
  obtain ⟨before,after,he⟩ := List.append_of_mem hi
  rw [he,List.foldl_append,List.foldl_cons,if_pos ha]
  apply masked_stable
  · exact State.write_same _ _ _ _
  · intro j _ hj he
    exact congrArg val (hinj j i hj ha he)

#axiomsClean masked_memory
#axiomsClean masked_readback
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Source
