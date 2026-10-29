import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.API
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section

/-- Raw-word write traces are a structural load/store boundary. Their payloads
are unconstrained; proving a frame never requires comparing NaN payloads. -/
structure RawStore (α : Type) where
  address : Address
  word : α

def applyRawStores (stores : List (RawStore α)) (before : Memory α) : Memory α :=
  stores.foldl (fun current st => writeWord current st.address st.word) before

theorem raw_store_frame (stores : List (RawStore α)) (before : Memory α) (address : Address)
    (off : ∀ st∈stores,st.address≠address) : applyRawStores stores before address=before address := by
  induction stores generalizing before with
  | nil => rfl
  | cons st rest ih =>
    change applyRawStores rest (writeWord before st.address st.word) address=before address
    rw [ih _ (fun x hx => off x (by simp [hx]))]
    exact writeWord_frame before st.address address st.word (Ne.symm (off st (by simp)))

theorem raw_allocation_frame (alloc : Allocation n caller) (stores : List (RawStore (BitVec 32)))
    (targets : ∀ st∈stores,∃ i,st.address.1=alloc.storage i)
    (before : Memory (BitVec 32)) (address : Address) (belongs : address.1∈caller) :
    applyRawStores stores before address=before address := by
  apply raw_store_frame
  intro st hs he
  obtain ⟨i,hi⟩ := targets st hs
  apply alloc.fresh i
  rw [he] at hi
  exact hi ▸ belongs

#axiomsClean raw_store_frame
#axiomsClean raw_allocation_frame
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
