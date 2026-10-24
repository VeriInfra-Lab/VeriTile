import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.Geometry

namespace VeriTile.Bench.Repairs.SgmvExpandSlice

abbrev Bytes := Nat → Nat → BitVec 8

/-- Object identity is separate from shared backing identity. Offsets and strides
are in elements. The repair's output contiguity requirement fixes row stride. -/
structure OutputView where
  (object backing offset rows columns width : Nat)
  negative : Bool
  widthAllowed : width = 2 ∨ width = 4

def OutputView.byte (v : OutputView) (row column byte : Nat) : Nat :=
  (v.offset + row*v.columns + column)*v.width + byte

/-- Every byte in an active supplied-output element is authorized. No other
backing, unused row, slice gap, metadata alias or input alias is excluded. -/
def AllowedByte (p : Shape) (segments : List Segment) (v : OutputView)
    (backing offset : Nat) : Prop :=
  backing = v.backing ∧ ∃ s ∈ segments, ∃ r c b,
    ActiveCell p s r c ∧ b < v.width ∧ offset = v.byte r c b

structure ByteWrite where
  (backing offset : Nat)
  value : BitVec 8

def ByteWrite.apply (w : ByteWrite) (memory : Bytes) : Bytes :=
  fun a b => if a = w.backing ∧ b = w.offset then w.value else memory a b

def writeAll (writes : List ByteWrite) (memory : Bytes) :=
  writes.foldl (fun m w => w.apply m) memory

theorem byteWrite_frame (w : ByteWrite) (memory : Bytes) (a b : Nat)
    (h : ¬ (a = w.backing ∧ b = w.offset)) : w.apply memory a b = memory a b := by
  simp [ByteWrite.apply,h]

theorem writeAll_frame (writes : List ByteWrite) (memory : Bytes) (a b : Nat)
    (h : ∀ w ∈ writes, ¬ (a = w.backing ∧ b = w.offset)) :
    writeAll writes memory a b = memory a b := by
  induction writes generalizing memory with
  | nil => rfl
  | cons w ws ih =>
      change writeAll ws (w.apply memory) a b = memory a b
      rw [ih _ (by intro x hx; exact h x (by simp [hx]))]
      exact byteWrite_frame w memory a b (h w (by simp))

theorem authorized_copy_frame (p : Shape) (segments : List Segment) (v : OutputView)
    (writes : List ByteWrite) (memory : Bytes)
    (legal : ∀ w ∈ writes, AllowedByte p segments v w.backing w.offset)
    (a b : Nat) (outside : ¬ AllowedByte p segments v a b) :
    writeAll writes memory a b = memory a b := by
  apply writeAll_frame
  intro w hw he
  exact outside (he.1 ▸ he.2 ▸ legal w hw)

/-- This is a concrete inhabited byte-copy implementation. A word is exactly
width bytes, so half/bfloat views never pretend to store arbitrary 32-bit words. -/
abbrev StoredWord (width : Nat) := Fin width → BitVec 8

def cellWrites (v : OutputView) (r c : Nat) (value : StoredWord v.width) : List ByteWrite :=
  (List.finRange v.width).map (fun b => ⟨v.backing,v.byte r c b.val,value b⟩)

theorem cellWrites_authorized (p : Shape) (segments : List Segment) (v : OutputView)
    (s : Segment) (hs : s ∈ segments) (r c : Nat) (active : ActiveCell p s r c)
    (value : StoredWord v.width) (w : ByteWrite) (hw : w ∈ cellWrites v r c value) :
    AllowedByte p segments v w.backing w.offset := by
  obtain ⟨b,_,rfl⟩ := List.mem_map.mp hw
  exact ⟨rfl,s,hs,r,c,b.val,active,b.isLt,rfl⟩

/-- Even an arbitrary partially committed copy is framed when each committed
byte belongs to the local output slice. Prefix order is not assumed. -/
theorem partial_copy_frame (p : Shape) (segments : List Segment) (v : OutputView)
    (s : Segment) (hs : s ∈ segments) (r c : Nat) (active : ActiveCell p s r c)
    (value : StoredWord v.width) (committed : List ByteWrite)
    (subset : ∀ w ∈ committed, w ∈ cellWrites v r c value)
    (memory : Bytes) (a b : Nat) (outside : ¬ AllowedByte p segments v a b) :
    writeAll committed memory a b = memory a b :=
  authorized_copy_frame p segments v committed memory
    (fun w hw => cellWrites_authorized p segments v s hs r c active value w (subset w hw))
    a b outside

theorem empty_write_identity (memory : Bytes) : writeAll [] memory = memory := rfl

end VeriTile.Bench.Repairs.SgmvExpandSlice
