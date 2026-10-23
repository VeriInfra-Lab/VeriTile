import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.Execution
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.KvCacheFilling

theorem runStores_permutation {ν : Type} (xs ys : List (Store ν)) (perm : xs.Perm ys)
    (same : ∀ a∈ys, ∀ b∈ys,a.address=b.address → a.value=b.value)
    (m : Address → ν) : runStores xs m=runStores ys m := by
  classical
  funext p
  by_cases present : ∃ s∈ys,s.address=p
  · obtain ⟨s,hs,hp⟩ := present
    have right := runStores_readback ys m p s.value ⟨s,hs,hp⟩
      (by intro a ha he; exact same a ha s hs (he.trans hp.symm))
    have left := runStores_readback xs m p s.value ⟨s,perm.mem_iff.mpr hs,hp⟩
      (by intro a ha he; exact same a (perm.mem_iff.mp ha) s hs (he.trans hp.symm))
    exact left.trans right.symm
  · have right := runStores_frame ys m p (by intro s hs hp; exact present ⟨s,hs,hp⟩)
    have left := runStores_frame xs m p
      (by intro s hs hp; exact present ⟨s,perm.mem_iff.mp hs,hp⟩)
    exact left.trans right.symm

theorem byteStores_value {α : Type} (r : Request) (d : Domain r) (l : Layout r)
    (ops : Operations α Byte) (kp : Plan r.dk) (vp : Plan r.dv) (k v : Nat → Nat → α)
    (encode : α → Nat → Byte) (m : Nat → Byte) (s : Store Byte)
    (hs : s∈byteStores r l ops kp vp k v encode) :
    s.value=executeBytes r l ops kp vp k v encode m s.address.row := by
  obtain ⟨a,ha,hs'⟩ := List.mem_flatMap.mp hs
  obtain ⟨j,hj,ha'⟩ := List.mem_flatMap.mp ha
  obtain ⟨b,hb,hptr,hvalue⟩ := (mem_expandStore ..).mp hs'
  obtain ⟨haddress,havalue⟩ := (mem_taskStores ..).mp ha'
  rw [hptr,hvalue,havalue]
  exact (whole_bytes_readback r d l ops kp vp k v encode m j hj a.address haddress b hb).symm

/-- Any complete byte-store interleaving has the same result. This includes
partial program interleavings, not only permutations of whole rows. -/
theorem byte_events_permutation {α : Type} (r : Request) (d : Domain r) (l : Layout r)
    (ops : Operations α Byte) (kp : Plan r.dk) (vp : Plan r.dv) (k v : Nat → Nat → α)
    (encode : α → Nat → Byte) (m : Nat → Byte) (events : List (Store Byte))
    (complete : events.Perm (byteStores r l ops kp vp k v encode)) :
    runStores events (fun a => m a.row)=
      runStores (byteStores r l ops kp vp k v encode) (fun a => m a.row) := by
  apply runStores_permutation _ _ complete
  intro a ha b hb equal
  rw [byteStores_value r d l ops kp vp k v encode m a ha,
    byteStores_value r d l ops kp vp k v encode m b hb,equal]

#axiomsClean byte_events_permutation
end VeriTile.Bench.Repairs.KvCacheFilling
