import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.Memory
namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton
set_option autoImplicit false
set_option maxHeartbeats 2000000

theorem runGrid_frame (ops : Operations Word) (p : Launch) (v : Layout) (order : List Nat)
    (valid : ∀ row ∈ order, row < p.rows) (heap : Heap) (address : Nat)
    (outside : ¬ OutputCell p v address) : runGrid ops p v order heap address = heap address := by
  induction order generalizing heap with
  | nil => rfl
  | cons row rest ih =>
    change runGrid ops p v rest (runRow ops p v heap row) address = heap address
    rw [ih (fun row hr => valid row (by simp [hr]))]
    exact runRow_frame ops p v heap row (valid row (by simp)) address outside

theorem runGrid_frozen (ops : Operations Word) (p : Launch) (v : Layout) (sep : Separation p v)
    (order : List Nat) (valid : ∀ row ∈ order, row < p.rows) (initial heap : Heap)
    (same : SameInput p v initial heap) :
    runGrid ops p v order heap = writeAll (order.flatMap (sourceRow ops p v initial)) heap := by
  induction order generalizing heap with
  | nil => rfl
  | cons row rest ih =>
    have hr := valid row (by simp)
    have hs : SameInput p v initial (runRow ops p v heap row) := by
      intro address inside
      exact (runRow_input ops p v heap sep row hr address inside).trans (same address inside)
    change runGrid ops p v rest (runRow ops p v heap row) =
      writeAll ((sourceRow ops p v initial row) ++ rest.flatMap (sourceRow ops p v initial)) heap
    rw [writeAll_append, ih (fun r he => valid r (by simp [he])) _ hs]
    simp only [runRow, sourceRow_same ops p v initial heap same row hr]

theorem output_address_injective (p : Launch) (v : Layout) (r s c d : Nat)
    (hc : c < p.columns) (hd : d < p.columns)
    (same : outputAddress p v r c = outputAddress p v s d) : r = s ∧ c = d := by
  apply linear_injective p.columns r s c d hc hd
  unfold outputAddress at same
  omega

theorem maximum_address_injective (v : Layout) (r s : Nat) (a b : Fin 2)
    (same : maximumAddress v r a = maximumAddress v s b) : r = s ∧ a = b := by
  have h : r * 2 + a.val = s * 2 + b.val := by
    unfold maximumAddress at same
    omega
  obtain ⟨hr, hb⟩ := linear_injective 2 r s a.val b.val a.isLt b.isLt h
  exact ⟨hr, Fin.ext hb⟩

theorem sourceRow_q_compatible (ops : Operations Word) (p : Launch) (v : Layout)
    (sep : Separation p v) (heap : Heap) (row column otherRow : Nat)
    (hr : row < p.rows) (hc : column < p.columns) (ho : otherRow < p.rows)
    (event : Write) (he : event ∈ sourceRow ops p v heap otherRow)
    (same : event.address = outputAddress p v row column) :
    event.value = qValue ops p v heap row column := by
  rcases (sourceRow_member ops p v heap otherRow event).mp he with ⟨c, hcc, rfl⟩ | ⟨b, rfl⟩
  · obtain ⟨rfl, rfl⟩ := output_address_injective p v otherRow row c column hcc hc same
    rfl
  · exact False.elim (sep.outputs _ (output_bound p v row column hr hc)
      (same ▸ maximum_bound p v otherRow b ho))

theorem sourceRow_m_compatible (ops : Operations Word) (p : Launch) (v : Layout)
    (sep : Separation p v) (heap : Heap) (row otherRow : Nat) (byte : Fin 2)
    (hr : row < p.rows) (ho : otherRow < p.rows)
    (event : Write) (he : event ∈ sourceRow ops p v heap otherRow)
    (same : event.address = maximumAddress v row byte) :
    event.value = mValue ops p v heap row byte := by
  rcases (sourceRow_member ops p v heap otherRow event).mp he with ⟨c, hc, rfl⟩ | ⟨b, rfl⟩
  · change outputAddress p v otherRow c = maximumAddress v row byte at same
    exact False.elim (sep.outputs _ (output_bound p v otherRow c ho hc)
      (by rw [same]; exact maximum_bound p v row byte hr))
  · obtain ⟨rfl, rfl⟩ := maximum_address_injective v otherRow row b byte same
    rfl

theorem runGrid_readback (ops : Operations Word) (p : Launch) (v : Layout)
    (sep : Separation p v) (schedule : Schedule p) (initial : Heap) :
    (∀ row, row < p.rows → ∀ column, column < p.columns →
      runGrid ops p v schedule.order initial (outputAddress p v row column) =
        qValue ops p v initial row column) ∧
    (∀ row, row < p.rows → ∀ byte : Fin 2,
      runGrid ops p v schedule.order initial (maximumAddress v row byte) =
        mValue ops p v initial row byte) := by
  have valid : ∀ row ∈ schedule.order, row < p.rows := by
    intro row hr
    exact List.mem_range.mp (schedule.complete.mem_iff.mp hr)
  have same : SameInput p v initial initial := by intro _ _; rfl
  rw [runGrid_frozen ops p v sep schedule.order valid initial initial same]
  constructor
  · intro row hr column hc
    apply writeAll_readback _ initial ⟨outputAddress p v row column, qValue ops p v initial row column⟩
    · apply List.mem_flatMap.mpr
      refine ⟨row, schedule.complete.mem_iff.mpr (List.mem_range.mpr hr), ?_⟩
      exact (sourceRow_member ops p v initial row _).mpr (Or.inl ⟨column, hc, rfl⟩)
    · intro event he sameAddress
      obtain ⟨other, ho, he⟩ := List.mem_flatMap.mp he
      exact sourceRow_q_compatible ops p v sep initial row column other hr hc (valid other ho) event he sameAddress
  · intro row hr byte
    apply writeAll_readback _ initial ⟨maximumAddress v row byte, mValue ops p v initial row byte⟩
    · apply List.mem_flatMap.mpr
      refine ⟨row, schedule.complete.mem_iff.mpr (List.mem_range.mpr hr), ?_⟩
      exact (sourceRow_member ops p v initial row _).mpr (Or.inr ⟨byte, rfl⟩)
    · intro event he sameAddress
      obtain ⟨other, ho, he⟩ := List.mem_flatMap.mp he
      exact sourceRow_m_compatible ops p v sep initial row other byte hr (valid other ho) event he sameAddress

/-- Any partial sequence of actual source stores preserves every byte outside
the two fresh output ranges. No rollback or successful-kernel result is used. -/
theorem partial_store_frame (ops : Operations Word) (p : Launch) (v : Layout)
    (initial : Heap) (events : List Write)
    (actual : ∀ event ∈ events, ∃ heap row, row < p.rows ∧ event ∈ sourceRow ops p v heap row)
    (address : Nat) (outside : ¬ OutputCell p v address) :
    writeAll events initial address = initial address := by
  apply writeAll_frame
  intro event he same
  obtain ⟨heap, row, hr, he⟩ := actual event he
  exact outside (same ▸ sourceRow_footprint ops p v heap row hr event he)

#axiomsClean runGrid_frozen
#axiomsClean runGrid_readback
#axiomsClean partial_store_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton
