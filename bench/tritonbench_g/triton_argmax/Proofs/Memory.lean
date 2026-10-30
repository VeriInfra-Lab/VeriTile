import bench.tritonbench_g.triton_argmax.Proofs.Tree

/-! Byte-addressed primitive memory. Device ownership is part of every address;
word bounds and a complete byte frame are separate from numerical operations.
The laws below are single-read/single-write contracts, never kernel laws. -/
namespace VeriTile.Bench.Repairs.TritonArgmax.Memory

abbrev Device := Nat
abbrev Address := Device × Nat
abbrev Byte := BitVec 8

structure Allocation where
  owner : Device
  base : Nat
  bytes : Nat
  deriving DecidableEq,Repr

def Allocation.contains (a : Allocation) (p : Address) : Prop :=
  p.1=a.owner ∧ a.base ≤ p.2 ∧ p.2 < a.base+a.bytes

def address (a : Allocation) (width index : Nat) : Address := (a.owner,a.base+width*index)

def Disjoint (p : Address) (n : Nat) (q : Address) (m : Nat) : Prop :=
  p.1 ≠ q.1 ∨ p.2+n ≤ q.2 ∨ q.2+m ≤ p.2

def Allocation.Disjoint (a b : Allocation) : Prop :=
  Memory.Disjoint (a.owner,a.base) a.bytes (b.owner,b.base) b.bytes

structure Primitives (M : Type) where
  byte : M → Address → Byte
  read : (width : Nat) → M → Address → BitVec (8*width)
  write : (width : Nat) → M → Address → BitVec (8*width) → M
  readSame : ∀ width s p value,read width (write width s p value) p=value
  readFrame : ∀ width otherWidth s p q value, Disjoint p width q otherWidth →
    read otherWidth (write width s p value) q=read otherWidth s q
  byteFrame : ∀ width s p q value, Disjoint p width q 1 →
    byte (write width s p value) q=byte s q

def stores {M ι : Type} (ops : Primitives M) (a : Allocation) (width : Nat)
    (key : ι → Nat) (value : ι → BitVec (8*width)) (jobs : List ι) (s : M) : M :=
  jobs.foldl (fun t j => ops.write width t (address a width (key j)) (value j)) s

theorem distinct {a : Allocation} {width i j : Nat} (positive : 0 < width) (different : i ≠ j) :
    Disjoint (address a width i) width (address a width j) width := by
  dsimp [Disjoint,address]
  by_cases order : i < j
  · right; left
    have h := Nat.mul_le_mul_left width (show i+1 ≤ j by omega)
    nlinarith
  · right; right
    have h := Nat.mul_le_mul_left width (show j+1 ≤ i by omega)
    nlinarith

theorem read_frame {M ι : Type} (ops : Primitives M) (a : Allocation) (width : Nat)
    (key : ι → Nat) (value : ι → BitVec (8*width)) (jobs : List ι) (s : M)
    (otherWidth : Nat) (p : Address)
    (apart : ∀ j,j∈jobs → Disjoint (address a width (key j)) width p otherWidth) :
    ops.read otherWidth (stores ops a width key value jobs s) p=ops.read otherWidth s p := by
  induction jobs generalizing s with
  | nil => rfl
  | cons j rest ih =>
    rw [stores,List.foldl_cons]
    change ops.read otherWidth (stores ops a width key value rest
      (ops.write width s (address a width (key j)) (value j))) p = _
    rw [ih _ (fun k hk => apart k (by simp [hk])),ops.readFrame]
    exact apart j (by simp)

theorem bytes_frame {M ι : Type} (ops : Primitives M) (a : Allocation) (width : Nat)
    (key : ι → Nat) (value : ι → BitVec (8*width)) (jobs : List ι) (s : M)
    (p : Address) (apart : ∀ j,j∈jobs → Disjoint (address a width (key j)) width p 1) :
    ops.byte (stores ops a width key value jobs s) p=ops.byte s p := by
  induction jobs generalizing s with
  | nil => rfl
  | cons j rest ih =>
    rw [stores,List.foldl_cons]
    change ops.byte (stores ops a width key value rest
      (ops.write width s (address a width (key j)) (value j))) p = _
    rw [ih _ (fun k hk => apart k (by simp [hk])),ops.byteFrame]
    exact apart j (by simp)

theorem stable_readback {M ι : Type} (ops : Primitives M) (a : Allocation) (width : Nat)
    (positive : 0 < width) (key : ι → Nat) (value : ι → BitVec (8*width))
    (jobs : List ι) (s : M) (j : ι)
    (initial : ops.read width s (address a width (key j))=value j)
    (consistent : ∀ i,i∈jobs → key i=key j → value i=value j) :
    ops.read width (stores ops a width key value jobs s) (address a width (key j))=value j := by
  induction jobs generalizing s with
  | nil => exact initial
  | cons i rest ih =>
    change ops.read width (stores ops a width key value rest
      (ops.write width s (address a width (key i)) (value i))) _ = _
    apply ih
    · by_cases same : key i=key j
      · rw [same,ops.readSame,consistent i (by simp) same]
      · rw [ops.readFrame _ _ _ _ _ _ (distinct positive same)]
        exact initial
    · intro k hk
      exact consistent k (by simp [hk])

theorem readback {M ι : Type} (ops : Primitives M) (a : Allocation) (width : Nat)
    (positive : 0 < width) (key : ι → Nat) (value : ι → BitVec (8*width))
    (jobs : List ι) (s : M) (j : ι) (member : j∈jobs)
    (consistent : ∀ i,i∈jobs → key i=key j → value i=value j) :
    ops.read width (stores ops a width key value jobs s) (address a width (key j))=value j := by
  induction jobs generalizing s with
  | nil => simp at member
  | cons i rest ih =>
    change ops.read width (stores ops a width key value rest
      (ops.write width s (address a width (key i)) (value i))) _ = _
    rcases List.mem_cons.mp member with equal | member
    · subst i
      apply stable_readback ops a width positive key value rest _ j (ops.readSame _ _ _ _)
      intro k hk
      exact consistent k (by simp [hk])
    · exact ih _ member (fun k hk => consistent k (by simp [hk]))

theorem subword_disjoint (a b : Allocation) (apart : a.Disjoint b)
    (width index otherWidth otherIndex : Nat)
    (inside : width*(index+1) ≤ a.bytes) (otherInside : otherWidth*(otherIndex+1) ≤ b.bytes) :
    Disjoint (address a width index) width (address b otherWidth otherIndex) otherWidth := by
  rcases apart with owners | left | right
  · exact Or.inl owners
  · right; left
    dsimp [address] at *
    nlinarith
  · right; right
    dsimp [address] at *
    nlinarith

#axiomsClean distinct
#axiomsClean read_frame
#axiomsClean bytes_frame
#axiomsClean stable_readback
#axiomsClean readback
#axiomsClean subword_disjoint
end VeriTile.Bench.Repairs.TritonArgmax.Memory
