import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.CopyBack

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.CallerLogical

/-- The ordinary local logical-view read/write contract, including the
supplied view's lazy-negative interpretation. This concerns one element; it
contains no program execution, copy footprint, indexing, or final-array law. -/
structure Codec (β : Type) (view : OutputView) where
  encode : β → StoredWord view.width
  decode : StoredWord view.width → β
  read_write : ∀ word, decode (encode word) = word

def read {β : Type} (view : OutputView) (codec : Codec β view) (bytes : Bytes) (r c : Nat) : β :=
  codec.decode (fun b => bytes view.backing (view.byte r c b.val))

theorem active_read {β : Type} (shape : Shape) (view : OutputView) (codec : Codec β view)
    (segments : List Segment) (capacity : shape.slice+shape.columns ≤ view.columns)
    (words : Nat → Nat → β) (entry : Bytes) (segment : Segment) (hs : segment ∈ segments)
    (r c : Nat) (active : ActiveCell shape segment r c) :
    read view codec (Host.returned view
      (copyBackWrites shape view segments (fun r c => codec.encode (words r c))) entry).memory r c = words r c := by
  have hbytes : (fun b => (Host.returned view
      (copyBackWrites shape view segments (fun r c => codec.encode (words r c))) entry).memory
        view.backing (view.byte r c b.val)) = codec.encode (words r c) := by
    funext b
    exact actual_copyback_readback shape view segments capacity _ entry segment hs r c active b
  unfold read
  rw [hbytes,codec.read_write]

theorem inactive_byte (shape : Shape) (view : OutputView) (segments : List Segment)
    (capacity : shape.slice+shape.columns ≤ view.columns) (r c : Nat) (hc : c < view.columns)
    (outside : ∀ segment ∈ segments, ¬ActiveCell shape segment r c) (byte : Fin view.width) :
    ¬AllowedByte shape segments view view.backing (view.byte r c byte.val) := by
  rintro ⟨_,segment,hs,r',c',b',active,hb',he⟩
  have hc' : c' < view.columns := by have h := active.2.2.2.2; omega
  obtain ⟨hr,hc,_⟩ := output_byte_injective view hc hc' byte.isLt hb' he
  subst r'
  subst c'
  exact outside segment hs active

theorem inactive_read {β : Type} (shape : Shape) (view : OutputView) (codec : Codec β view)
    (segments : List Segment) (capacity : shape.slice+shape.columns ≤ view.columns)
    (words : Nat → Nat → β) (entry : Bytes) (r c : Nat) (hc : c < view.columns)
    (outside : ∀ segment ∈ segments, ¬ActiveCell shape segment r c) :
    read view codec (Host.returned view
      (copyBackWrites shape view segments (fun r c => codec.encode (words r c))) entry).memory r c =
        read view codec entry r c := by
  unfold read
  congr 1
  funext byte
  exact actual_copyback_frame shape view segments _ entry _ _
    (inactive_byte shape view segments capacity r c hc outside byte)

#axiomsClean active_read
#axiomsClean inactive_byte
#axiomsClean inactive_read
end VeriTile.Bench.Repairs.SgmvExpandSlice.CallerLogical
