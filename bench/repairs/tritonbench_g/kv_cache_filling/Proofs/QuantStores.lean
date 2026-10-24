import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.QuantExecution

namespace VeriTile.Bench.Repairs.KvCacheFilling.QuantStores
open VeriTile.Triton FP.Structural QuantProfile CopySource CopyExecution QuantExecution
noncomputable section

theorem scatter_read {α : Type} (p : Parameters) (dst : Nat) (key : Bool)
    (values : Nat → BitVec 8) (s : State α) (i : Nat) (hi : i < capacity p key)
    (covered : capacity p key ≤ lanes p key) :
    (QuantExecution.scatter p dst key values s).mem (output key) (dst*capacity p key+i) =
      .mk .nat (values i).toNat := by
  let ix : TileIndex [lanes p key] := (⟨i,by omega⟩,PUnit.unit)
  unfold QuantExecution.scatter
  apply State.masked_scatter_readback (i := ix)
    (active := fun j : TileIndex [lanes p key] => j.1.val < capacity p key)
  · exact hi
  · intro a b he
    have h := Nat.add_left_cancel he
    cases a with | mk a au =>
      cases b with | mk b bu =>
        have hab : a = b := Fin.ext h
        cases hab
        rfl

theorem scatter_frame {α : Type} (p : Parameters) (dst : Nat) (key : Bool)
    (values : Nat → BitVec 8) (s : State α) (reg : RegionName) (i : Nat)
    (h : reg ≠ output key ∨ i < dst*capacity p key ∨ dst*capacity p key+capacity p key ≤ i) :
    (QuantExecution.scatter p dst key values s).mem reg i = s.mem reg i := by
  unfold QuantExecution.scatter
  apply State.masked_scatter_frame
  rcases h with hr | hlo | hhi
  · exact Or.inl hr
  · right; intro j _ hj; omega
  · right; intro j _ hj; omega

theorem metadata_cache {α : Type} (dst : Nat) (k v : Row α) (s : State α)
    (key : Bool) (i : Nat) :
    (writeMetadata dst k v s).mem (output key) i = s.mem (output key) i := by
  cases key <;> simp [writeMetadata,output]

theorem metadata_read {α : Type} (dst : Nat) (k v : Row α) (s : State α)
    (key : Bool) (i : Nat) (hi : i < 2) :
    (writeMetadata dst k v s).mem (metaRegion key) (dst*2+i) =
      .mk .real (if i = 0 then (if key then k else v).scale else (if key then k else v).zero) := by
  cases key <;> interval_cases i <;> simp [writeMetadata,metaRegion]

theorem metadata_frame {α : Type} (dst : Nat) (k v : Row α) (s : State α)
    (reg : RegionName) (i : Nat)
    (h : ∀ key, reg ≠ metaRegion key ∨ i < dst*2 ∨ dst*2+2 ≤ i) :
    (writeMetadata dst k v s).mem reg i = s.mem reg i := by
  have hk0 : reg ≠ "keyMeta" ∨ i ≠ dst*2 := by
    rcases h Bool.true with h | h | h
    · exact Or.inl h
    · exact Or.inr (by omega)
    · exact Or.inr (by omega)
  have hk1 : reg ≠ "keyMeta" ∨ i ≠ dst*2+1 := by
    rcases h Bool.true with h | h | h
    · exact Or.inl h
    · exact Or.inr (by omega)
    · exact Or.inr (by omega)
  have hv0 : reg ≠ "valueMeta" ∨ i ≠ dst*2 := by
    rcases h Bool.false with h | h | h
    · exact Or.inl h
    · exact Or.inr (by omega)
    · exact Or.inr (by omega)
  have hv1 : reg ≠ "valueMeta" ∨ i ≠ dst*2+1 := by
    rcases h Bool.false with h | h | h
    · exact Or.inl h
    · exact Or.inr (by omega)
    · exact Or.inr (by omega)
  simp only [writeMetadata,State.write_other _ _ reg _ i _ hv1,
    State.write_other _ _ reg _ i _ hv0,State.write_other _ _ reg _ i _ hk1,
    State.write_other _ _ reg _ i _ hk0]

#axiomsClean scatter_read
#axiomsClean scatter_frame
#axiomsClean metadata_read
#axiomsClean metadata_frame
end
end VeriTile.Bench.Repairs.KvCacheFilling.QuantStores
