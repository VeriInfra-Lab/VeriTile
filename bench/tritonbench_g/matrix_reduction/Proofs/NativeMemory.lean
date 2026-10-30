import bench.tritonbench_g.matrix_reduction.Proofs.NativeSource

namespace VeriTile.Bench.TritonBenchG.MatrixReduction.Native
open VeriTile.Triton TypedSource MachineMax ConcreteCorrect
noncomputable section
open Classical
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

abbrev Byte := Fin 256
def wordByte (d : Storage) (v : Bits d) (k : Nat) : Byte :=
  ⟨(v.toNat / 2^(8*k))%256,Nat.mod_lt _ (by decide)⟩
def readBytes : Nat → (Nat → Option Byte) → Option Nat
  | 0,_ => some 0
  | n+1,b => do
    let low ← b 0
    let high ← readBytes n (fun k => b (k+1))
    pure (low.val+256*high)
def decode (d : Storage) (b : Nat → Option Byte) : Option (Bits d) :=
  (readBytes (bytes d) b).map (BitVec.ofNat (width d))
def Disjoint (a b width : Nat) : Prop := a+width ≤ b ∨ b+width ≤ a

/-- Ordinary raw typed memory, with absent bytes preserved. Local primitive
laws only; these cannot return a reduction result or invent an input matrix. -/
structure Memory (S : Type) (d : Storage) where
  load : S → Nat → Option (Bits d)
  store : S → Nat → Bits d → S
  byte : S → Nat → Option Byte
  raw_load : ∀ s a, load s a = decode d (fun k => byte s (a+k))
  load_store : ∀ s a v, load (store s a v) a = some v
  load_disjoint : ∀ s a v b, Disjoint a b (bytes d) → load (store s a v) b = load s b
  store_byte : ∀ s a v k, k < bytes d → byte (store s a v) (a+k) = some (wordByte d v k)
  byte_frame : ∀ s a v b, b<a ∨ a+bytes d≤b → byte (store s a v) b = byte s b

structure Layout (d : Storage) (M N : Nat) where
  input : Nat
  output : Nat

structure Layout.Valid (p : Layout d M N) : Prop where
  input_end : p.input+bytes d*(M*N) ≤ 2^64
  output_end : p.output+bytes d*M ≤ 2^64
  fresh : p.input+bytes d*(M*N) ≤ p.output ∨ p.output+bytes d*M ≤ p.input

def Layout.xAddress (p : Layout d M N) (i : Fin M) (j : Fin N) : Nat := p.input+bytes d*(i.val*N+j.val)
def Layout.yAddress (p : Layout d M N) (i : Fin M) : Nat := p.output+bytes d*i.val

theorem input_inside (p : Layout d M N) (i : Fin M) (j : Fin N) :
    p.input ≤ p.xAddress i j ∧ p.xAddress i j+bytes d ≤ p.input+bytes d*(M*N) := by
  have h := Nat.mul_le_mul_left (bytes d) (show i.val*N+j.val+1 ≤ M*N from linear_index_lt i j)
  unfold Layout.xAddress
  constructor
  · omega
  · nlinarith

theorem output_inside (p : Layout d M N) (i : Fin M) :
    p.output ≤ p.yAddress i ∧ p.yAddress i+bytes d ≤ p.output+bytes d*M := by
  have h := Nat.mul_le_mul_left (bytes d) (show i.val+1 ≤ M from i.isLt)
  unfold Layout.yAddress
  constructor
  · omega
  · nlinarith

theorem output_disjoint (p : Layout d M N) (i k : Fin M) (hne : i ≠ k) :
    Disjoint (p.yAddress i) (p.yAddress k) (bytes d) := by
  have hne' : i.val ≠ k.val := by intro h; exact hne (Fin.ext h)
  rcases lt_or_gt_of_ne hne' with h | h
  · left
    have := Nat.mul_le_mul_left (bytes d) (show i.val+1 ≤ k.val from h)
    unfold Layout.yAddress
    nlinarith
  · right
    have := Nat.mul_le_mul_left (bytes d) (show k.val+1 ≤ i.val from h)
    unfold Layout.yAddress
    nlinarith

theorem input_output_disjoint (p : Layout d M N) (valid : p.Valid) (i k : Fin M) (j : Fin N) :
    Disjoint (p.yAddress k) (p.xAddress i j) (bytes d) := by
  have hx := input_inside p i j
  have hy := output_inside p k
  rcases valid.fresh with hf | hf
  · exact Or.inr (by omega)
  · exact Or.inl (by omega)

/-- A successful runtime chooses complete row stores and complete per-row
reduction trees. This is an ordinary source/compiler execution boundary.
The coverage theorem below derives every output from those primitive events. -/
structure Schedule (M N : Nat) where
  rows : List (Fin M)
  covers_rows : ∀ i, i ∈ rows
  tree : Fin M → Tree N
  covers_columns : ∀ i, (tree i).Complete

def stores (ops : Memory S d) (address : Fin M → Nat) (value : Fin M → Bits d) : List (Fin M) → S → S
  | [],s => s
  | i::rest,s => stores ops address value rest (ops.store s (address i) (value i))

theorem stores_load_frame (ops : Memory S d) (address : Fin M → Nat) (value : Fin M → Bits d)
    (rows : List (Fin M)) (s : S) (a : Nat) (sep : ∀ i ∈ rows, Disjoint (address i) a (bytes d)) :
    ops.load (stores ops address value rows s) a = ops.load s a := by
  induction rows generalizing s with
  | nil => rfl
  | cons i rest ih =>
    rw [stores,ih _ (fun k hk => sep k (by simp [hk])),ops.load_disjoint _ _ _ _ (sep i (by simp))]

theorem stores_readback (ops : Memory S d) (p : Layout d M N) (value : Fin M → Bits d)
    (rows : List (Fin M)) (s : S) (i : Fin M) (covered : i ∈ rows) :
    ops.load (stores ops p.yAddress value rows s) (p.yAddress i) = some (value i) := by
  induction rows generalizing s with
  | nil => simp at covered
  | cons k rest ih =>
    by_cases hr : i ∈ rest
    · exact ih (ops.store s (p.yAddress k) (value k)) hr
    · have he : i=k := by simpa [hr] using covered
      subst k
      rw [stores,stores_load_frame,ops.load_store]
      intro k hk
      exact output_disjoint p k i (by intro he; subst k; exact hr hk)

theorem stores_byte_frame (ops : Memory S d) (address : Fin M → Nat) (value : Fin M → Bits d)
    (rows : List (Fin M)) (s : S) (a : Nat) (outside : ∀ i ∈ rows, a < address i ∨ address i+bytes d ≤ a) :
    ops.byte (stores ops address value rows s) a = ops.byte s a := by
  induction rows generalizing s with
  | nil => rfl
  | cons i rest ih =>
    rw [stores,ih _ (fun k hk => outside k (by simp [hk])),ops.byte_frame _ _ _ _ (outside i (by simp))]

theorem stores_byte_readback (ops : Memory S d) (p : Layout d M N) (value : Fin M → Bits d)
    (rows : List (Fin M)) (s : S) (i : Fin M) (covered : i ∈ rows) (k : Nat) (hk : k < bytes d) :
    ops.byte (stores ops p.yAddress value rows s) (p.yAddress i+k) = some (wordByte d (value i) k) := by
  induction rows generalizing s with
  | nil => simp at covered
  | cons j rest ih =>
    by_cases hr : i ∈ rest
    · exact ih (ops.store s (p.yAddress j) (value j)) hr
    · have he : i=j := by simpa [hr] using covered
      subst j
      rw [stores,stores_byte_frame,ops.store_byte _ _ _ _ hk]
      intro j hj
      have hs := output_disjoint p j i (by intro he; subst j; exact hr hj)
      unfold Disjoint at hs
      omega

def sourceObserved (ops : Memory S d) (p : Layout d M N) (sm sn : Nat) (s : S)
    (i : Fin M) (j : Fin N) : Bits d :=
  (ops.load s (inputPointer p.input (bytes d) sm sn i.val j.val)).getD 0

def observed (ops : Memory S d) (_admitted : Admitted M N) (p : Layout d M N) (s : S) :=
  sourceObserved ops p (max N 1) 1 s

/-- The actual launch strides enter make_block_ptr. The output stride argument
is intentionally unused, exactly as in the source's y_ptr + arange store. -/
def sourceKernel (ops : Memory S d) (num : Numerics) (_admitted : Admitted M N) (p : Layout d M N)
    (sm sn _sy : Nat) (schedule : Schedule M N) (s : S) : Except String S :=
  if ∀ i : Fin M, ∀ j : Fin N,
      (ops.load s (inputPointer p.input (bytes d) sm sn i.val j.val)).isSome then
    .ok (stores ops (fun i => outputPointer p.output (bytes d) i.val)
      (fun i => rowResult num d (sourceObserved ops p sm sn s i) (schedule.tree i)) schedule.rows s)
  else .error "input storage bytes absent"

def kernel (ops : Memory S d) (num : Numerics) (admitted : Admitted M N) (p : Layout d M N)
    (schedule : Schedule M N) (s : S) : Except String S :=
  sourceKernel ops num admitted p (max N 1) 1 1 schedule s

def Input (ops : Memory S d) (p : Layout d M N) (s : S) (xs : Fin M → Fin N → Bits d) : Prop :=
  ∀ i j, ops.load s (p.xAddress i j) = some (xs i j)

def KernelSpec (ops : Memory S d) (p : Layout d M N) (s t : S) (xs : Fin M → Fin N → Bits d) : Prop :=
  (∀ i, ∃ v, ops.load t (p.yAddress i) = some v ∧ IsMaximum d (xs i) v ∧
    isNaN d v = Bool.false ∧ ∀ k, k < bytes d → ops.byte t (p.yAddress i+k) = some (wordByte d v k)) ∧
  Input ops p t xs ∧
  (∀ a, a < p.output ∨ p.output+bytes d*M ≤ a → ops.byte t a = ops.byte s a)

theorem kernel_correct (ops : Memory S d) (num : Numerics) (laws : NumericalContracts num)
    (admitted : Admitted M N) (p : Layout d M N) (valid : p.Valid) (schedule : Schedule M N)
    (s : S) (xs : Fin M → Fin N → Bits d) (input : Input ops p s xs)
    (hn : ∀ i j, isNaN d (xs i j) = Bool.false) :
    ∃ t, kernel ops num admitted p schedule s = .ok t ∧ KernelSpec ops p s t xs := by
  have hx i j : ops.load s (inputPointer p.input (bytes d) (max N 1) 1 i.val j.val) = some (xs i j) := by
    rw [input_address_exact admitted d p.input valid.input_end i j]
    exact input i j
  have present : ∀ i : Fin M, ∀ j : Fin N,
      (ops.load s (inputPointer p.input (bytes d) (max N 1) 1 i.val j.val)).isSome := by
    intro i j; rw [hx]; rfl
  have he : observed ops admitted p s = xs := by
    funext i j
    simp [observed,sourceObserved,hx]
  have hy : (fun i : Fin M => outputPointer p.output (bytes d) i.val) = p.yAddress := by
    funext i
    exact output_address_exact admitted d p.output valid.output_end i
  let value := fun i => rowResult num d (xs i) (schedule.tree i)
  refine ⟨stores ops p.yAddress value schedule.rows s,?_,?_,?_,?_⟩
  · change sourceObserved ops p (max N 1) 1 s = xs at he
    simp [kernel, source16, source32,sourceKernel,present,he,hy,value]
  · intro i
    have hs := complete_tree_maximum num laws d (xs i) (hn i) (schedule.tree i) (schedule.covers_columns i)
    exact ⟨value i,stores_readback ops p value schedule.rows s i (schedule.covers_rows i),hs.1,hs.2,
      fun k hk => stores_byte_readback ops p value schedule.rows s i (schedule.covers_rows i) k hk⟩
  · intro i j
    rw [stores_load_frame]
    · exact input i j
    · intro k _; exact input_output_disjoint p valid i k j
  · intro a ha
    apply stores_byte_frame
    intro i _
    have hb := output_inside p i
    omega

#axiomsClean input_inside
#axiomsClean output_inside
#axiomsClean input_output_disjoint
#axiomsClean output_disjoint
#axiomsClean stores_readback
#axiomsClean stores_byte_readback
#axiomsClean kernel_correct
#specNonCircular KernelSpec avoiding [kernel, source16, source32,sourceKernel,stores,observed,sourceObserved]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatrixReduction.Native
