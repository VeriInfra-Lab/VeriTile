import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.rowwise_quantization_triton.Kernels
namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton
set_option autoImplicit false
set_option maxHeartbeats 2000000

theorem writeAll_append (a b : List Write) (heap : Heap) :
    writeAll (a ++ b) heap = writeAll b (writeAll a heap) := by
  simp [writeAll, List.foldl_append]

theorem writeAll_frame (events : List Write) (heap : Heap) (address : Nat)
    (outside : ∀ event ∈ events, event.address ≠ address) :
    writeAll events heap address = heap address := by
  induction events generalizing heap with
  | nil => rfl
  | cons event rest ih =>
    change writeAll rest (write heap event) address = heap address
    rw [ih _ (fun e he => outside e (by simp [he]))]
    simp [write, Ne.symm (outside event (by simp))]

theorem writeAll_preserves_value (events : List Write) (heap : Heap) (address : Nat) (value : Byte)
    (initial : heap address = value)
    (compatible : ∀ event ∈ events, event.address = address → event.value = value) :
    writeAll events heap address = value := by
  induction events generalizing heap with
  | nil => exact initial
  | cons event rest ih =>
    apply ih
    · by_cases h : address = event.address
      · simpa [write, h] using compatible event (by simp) h.symm
      · simp [write, h, initial]
    · exact fun e he => compatible e (by simp [he])

theorem writeAll_readback (events : List Write) (heap : Heap) (event : Write)
    (present : event ∈ events)
    (compatible : ∀ other ∈ events, other.address = event.address → other.value = event.value) :
    writeAll events heap event.address = event.value := by
  obtain ⟨before, after, he⟩ := List.mem_iff_append.mp present
  subst events
  rw [writeAll_append]
  change writeAll after (write (writeAll before heap) event) event.address = event.value
  apply writeAll_preserves_value
  · simp [write]
  · intro other ho
    exact compatible other (by simp [ho])

theorem linear_bound (rows columns row column : Nat) (hr : row < rows) (hc : column < columns) :
    row * columns + column < rows * columns := by
  have h := Nat.mul_le_mul_right columns (show row + 1 ≤ rows by omega)
  nlinarith

theorem linear_injective (columns r s c d : Nat) (hc : c < columns) (hd : d < columns)
    (h : r * columns + c = s * columns + d) : r = s ∧ c = d := by
  have positive : 0 < columns := by omega
  have hr := congrArg (fun n => n / columns) h
  have hh : r = s := by
    simpa [Nat.mul_comm, Nat.mul_add_div positive, Nat.div_eq_of_lt hc, Nat.div_eq_of_lt hd] using hr
  subst s
  exact ⟨rfl, Nat.add_left_cancel h⟩

theorem input_bound (p : Launch) (v : Layout) (row column byte : Nat)
    (hr : row < p.rows) (hc : column < p.columns) (hb : byte < 4) :
    InRange v.input (4 * (p.rows * p.columns)) (inputAddress p v row column + byte) := by
  have h := linear_bound p.rows p.columns row column hr hc
  unfold InRange inputAddress
  constructor <;> omega

theorem output_bound (p : Launch) (v : Layout) (row column : Nat)
    (hr : row < p.rows) (hc : column < p.columns) :
    InRange v.output (p.rows * p.columns) (outputAddress p v row column) := by
  have h := linear_bound p.rows p.columns row column hr hc
  unfold InRange outputAddress
  constructor <;> omega

theorem maximum_bound (p : Launch) (v : Layout) (row : Nat) (byte : Fin 2)
    (hr : row < p.rows) : InRange v.maximum (2 * p.rows) (maximumAddress v row byte) := by
  have hb := byte.isLt
  unfold InRange maximumAddress
  constructor <;> omega

def qValue (ops : Operations Word) (p : Launch) (v : Layout) (heap : Heap) (row column : Nat) : Byte :=
  rowQuantized ops (load32 heap (inputAddress p v row column))
    (rowMaximum ops p (fun c => load32 heap (inputAddress p v row c)))

def mValue (ops : Operations Word) (p : Launch) (v : Layout) (heap : Heap) (row : Nat)
    (byte : Fin 2) : Byte :=
  halfByte (ops.toHalf (rowMaximum ops p (fun c => load32 heap (inputAddress p v row c)))) byte

theorem sourceRow_form (ops : Operations Word) (p : Launch) (v : Layout) (heap : Heap) (row : Nat) :
    sourceRow ops p v heap row =
      (List.range p.width).filterMap (fun column => if column < p.columns then
        some ⟨outputAddress p v row column, qValue ops p v heap row column⟩ else none) ++
      (List.finRange 2).map (fun byte => ⟨maximumAddress v row byte, mValue ops p v heap row byte⟩) := rfl

theorem sourceRow_member (ops : Operations Word) (p : Launch) (v : Layout) (heap : Heap)
    (row : Nat) (event : Write) : event ∈ sourceRow ops p v heap row ↔
    (∃ column, column < p.columns ∧ event = ⟨outputAddress p v row column, qValue ops p v heap row column⟩) ∨
    (∃ byte : Fin 2, event = ⟨maximumAddress v row byte, mValue ops p v heap row byte⟩) := by
  rw [sourceRow_form]
  simp only [List.mem_append, List.mem_filterMap, List.mem_range, List.mem_map, List.mem_finRange, true_and]
  constructor
  · rintro (⟨column, hw, he⟩ | ⟨byte, he⟩)
    · by_cases hc : column < p.columns
      · simp only [if_pos hc, Option.some.injEq] at he
        exact Or.inl ⟨column, hc, he.symm⟩
      · simp [hc] at he
    · exact Or.inr ⟨byte, he.symm⟩
  · rintro (⟨column, hc, rfl⟩ | ⟨byte, rfl⟩)
    · exact Or.inl ⟨column, lt_of_lt_of_le hc p.covers, by simp [hc]⟩
    · exact Or.inr ⟨byte, rfl⟩

theorem sourceRow_footprint (ops : Operations Word) (p : Launch) (v : Layout) (heap : Heap)
    (row : Nat) (hr : row < p.rows) (event : Write) (he : event ∈ sourceRow ops p v heap row) :
    OutputCell p v event.address := by
  rcases (sourceRow_member ops p v heap row event).mp he with ⟨c, hc, rfl⟩ | ⟨b, rfl⟩
  · exact Or.inl (output_bound p v row c hr hc)
  · exact Or.inr (maximum_bound p v row b hr)

theorem runRow_frame (ops : Operations Word) (p : Launch) (v : Layout) (heap : Heap)
    (row : Nat) (hr : row < p.rows) (address : Nat) (outside : ¬ OutputCell p v address) :
    runRow ops p v heap row address = heap address := by
  apply writeAll_frame
  intro event he same
  exact outside (same ▸ sourceRow_footprint ops p v heap row hr event he)

def SameInput (p : Launch) (v : Layout) (before after : Heap) : Prop :=
  ∀ address, InRange v.input (4 * (p.rows * p.columns)) address → after address = before address

theorem runRow_input (ops : Operations Word) (p : Launch) (v : Layout) (heap : Heap)
    (sep : Separation p v) (row : Nat) (hr : row < p.rows) :
    SameInput p v heap (runRow ops p v heap row) := by
  intro address inside
  exact runRow_frame ops p v heap row hr address (sep.input address inside)

theorem load32_same (p : Launch) (v : Layout) (before after : Heap) (same : SameInput p v before after)
    (row column : Nat) (hr : row < p.rows) (hc : column < p.columns) :
    load32 after (inputAddress p v row column) = load32 before (inputAddress p v row column) := by
  unfold load32
  rw [same _ (by simpa using input_bound p v row column 0 hr hc (by decide)),
    same _ (input_bound p v row column 1 hr hc (by decide)),
    same _ (input_bound p v row column 2 hr hc (by decide)),
    same _ (input_bound p v row column 3 hr hc (by decide))]

theorem Tree.evaluate_congr {α : Type} (tree : Tree) (combine : α → α → α) (f g : Nat → α)
    (same : ∀ i ∈ tree.leaves, f i = g i) : tree.evaluate combine f = tree.evaluate combine g := by
  induction tree with
  | leaf i => exact same i (by simp [Tree.leaves])
  | node left right hl hr =>
    simp only [Tree.evaluate]
    rw [hl (fun i hi => same i (by simp [Tree.leaves, hi])), hr (fun i hi => same i (by simp [Tree.leaves, hi]))]

theorem rowMaximum_congr {α : Type} (ops : Operations α) (p : Launch) (f g : Nat → α)
    (same : ∀ c, c < p.columns → f c = g c) : rowMaximum ops p f = rowMaximum ops p g := by
  apply Tree.evaluate_congr
  intro c _
  by_cases hc : c < p.columns <;> simp [hc, same c]

theorem sourceRow_same (ops : Operations Word) (p : Launch) (v : Layout)
    (before after : Heap) (same : SameInput p v before after) (row : Nat) (hr : row < p.rows) :
    sourceRow ops p v after row = sourceRow ops p v before row := by
  have hm := rowMaximum_congr ops p _ _ (fun c hc => load32_same p v before after same row c hr hc)
  rw [sourceRow_form, sourceRow_form]
  congr 1
  · apply List.filterMap_congr
    intro c _
    by_cases hc : c < p.columns
    · simp [hc, qValue, hm, load32_same p v before after same row c hr hc]
    · simp [hc]
  · apply List.map_congr_left
    intro b _
    simp [mValue, hm]

#axiomsClean writeAll_readback
#axiomsClean linear_injective
#axiomsClean runRow_frame
#axiomsClean sourceRow_same
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton
