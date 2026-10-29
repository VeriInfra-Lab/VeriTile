import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.Metadata

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.HostMetadata

/-- Python zip over starts.tolist(), lengths.tolist()[:batches], and
adapters.tolist(), in their original batch order. -/
def rawSegments (batches : Nat) (starts lengths adapters : List Int) : List RawSegment :=
  (starts.zip ((lengths.take batches).zip adapters)).map (fun t => ⟨t.1,t.2.1,t.2.2⟩)

theorem projections (batches : Nat) (starts lengths adapters : List Int)
    (hs : starts.length = batches) (hl : batches ≤ lengths.length) (ha : adapters.length = batches) :
    (rawSegments batches starts lengths adapters).map RawSegment.start = starts ∧
    (rawSegments batches starts lengths adapters).map RawSegment.length = lengths.take batches ∧
    (rawSegments batches starts lengths adapters).map RawSegment.adapter = adapters := by
  have htake : (lengths.take batches).length = batches := by simp [List.length_take,Nat.min_eq_left hl]
  have hzip : ((lengths.take batches).zip adapters).length = batches := by simp [htake,ha]
  have hfst := List.map_fst_zip (l₁ := starts) (l₂ := (lengths.take batches).zip adapters) (by omega)
  have hsnd := List.map_snd_zip (l₁ := starts) (l₂ := (lengths.take batches).zip adapters) (by omega)
  refine ⟨?_,?_,?_⟩
  · simpa only [rawSegments,List.map_map,Function.comp_def] using hfst
  · calc
      (rawSegments batches starts lengths adapters).map RawSegment.length =
          ((starts.zip ((lengths.take batches).zip adapters)).map Prod.snd).map Prod.fst := by
            simp only [rawSegments,List.map_map,Function.comp_def]
      _ = ((lengths.take batches).zip adapters).map Prod.fst := by rw [hsnd]
      _ = lengths.take batches := List.map_fst_zip (by omega)
  · calc
      (rawSegments batches starts lengths adapters).map RawSegment.adapter =
          ((starts.zip ((lengths.take batches).zip adapters)).map Prod.snd).map Prod.snd := by
            simp only [rawSegments,List.map_map,Function.comp_def]
      _ = ((lengths.take batches).zip adapters).map Prod.snd := by rw [hsnd]
      _ = adapters := List.map_snd_zip (by omega)

theorem full_lengths (batches : Nat) (starts lengths adapters : List Int)
    (hs : starts.length = batches) (hl : batches ≤ lengths.length) (ha : adapters.length = batches) :
    (rawSegments batches starts lengths adapters).map RawSegment.length ++ lengths.drop batches = lengths := by
  rw [(projections batches starts lengths adapters hs hl ha).2.1]
  exact List.take_append_drop batches lengths

theorem batch_count (batches : Nat) (starts lengths adapters : List Int)
    (hs : starts.length = batches) (hl : batches ≤ lengths.length) (ha : adapters.length = batches) :
    (rawSegments batches starts lengths adapters).length = batches := by
  have h := congrArg List.length (projections batches starts lengths adapters hs hl ha).1
  simpa only [List.length_map,hs] using h

#axiomsClean projections
#axiomsClean full_lengths
#axiomsClean batch_count
end VeriTile.Bench.Repairs.SgmvExpandSlice.HostMetadata
