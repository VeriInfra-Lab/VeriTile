import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.Memory

namespace VeriTile.Bench.Repairs.QuantizeCopyKv
set_option autoImplicit false
set_option maxHeartbeats 2000000

theorem runProgram_frame {α η : Type} (ops : Operations α η) (p : Request) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 2 → Byte) (v : Layout)
    (heap : Heap) (hmeta : Metadata p v heap) (s : Nat) (hs : s < p.tokens)
    (a : Nat) (outside : ¬ Touched p v a) :
    runProgram ops p plan decode encode v heap s a = heap a := by
  apply RW.writeAll_frame
  intro event he same
  exact outside (same ▸ programEvents_footprint ops p plan decode encode v heap hmeta s hs event he)

theorem runProgram_sameReads {α η : Type} (ops : Operations α η) (p : Request) (d : Domain p) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 2 → Byte) (v : Layout) (alloc : NativeAllocations p v)
    (heap : Heap) (hmeta : Metadata p v heap) (s : Nat) (hs : s < p.tokens) :
    SameReads v heap (runProgram ops p plan decode encode v heap s) := by
  intro r hr a inside
  exact runProgram_frame ops p plan decode encode v heap hmeta s hs a (input_untouched p d v alloc r hr a inside)

theorem runGrid_frozen {α η : Type} (ops : Operations α η) (p : Request) (d : Domain p) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 2 → Byte) (v : Layout) (alloc : NativeAllocations p v)
    (order : List Nat) (valid : ∀ s ∈ order, s < p.tokens) (initial heap : Heap)
    (hmeta : Metadata p v initial) (same : SameReads v initial heap) :
    runGrid ops p plan decode encode v order heap =
      writeAll (order.flatMap (programEvents ops p plan decode encode v initial)) heap := by
  induction order generalizing heap with
  | nil => rfl
  | cons s rest ih =>
    have hs := valid s (by simp)
    have hm := metadata_same p v alloc initial heap same hmeta
    have hkeep := runProgram_sameReads ops p d plan decode encode v alloc heap hm s hs
    have next : SameReads v initial (runProgram ops p plan decode encode v heap s) := by
      intro r hr a ha
      exact (hkeep r hr a ha).trans (same r hr a ha)
    change runGrid ops p plan decode encode v rest (runProgram ops p plan decode encode v heap s) =
      writeAll (programEvents ops p plan decode encode v initial s ++ rest.flatMap (programEvents ops p plan decode encode v initial)) heap
    rw [ih (fun t ht => valid t (by simp [ht])) _ next]
    simp only [runProgram, programEvents_same ops p d plan decode encode v alloc initial heap same s hs,
      writeAll, RowwiseQuantizationTriton.writeAll, List.foldl_append]

theorem output_injective (p : Request) (v : Layout) (s t h k c e : Nat)
    (hh : h < p.heads) (hk : k < p.heads) (hc : c < p.columns) (he : e < p.columns)
    (same : outputAddress p v s h c = outputAddress p v t k e) : s = t ∧ h = k ∧ c = e := by
  have hw : (s*p.heads+h)*p.columns+c = (t*p.heads+k)*p.columns+e := by
    exact Nat.add_left_cancel same
  obtain ⟨hr,hc'⟩ := RW.linear_injective p.columns _ _ c e hc he hw
  obtain ⟨hs,hh'⟩ := RW.linear_injective p.heads s t h k hh hk hr
  exact ⟨hs,hh',hc'⟩

theorem scale_injective (p : Request) (v : Layout) (s t h k : Nat) (a b : Fin 2)
    (hh : h < p.heads) (hk : k < p.heads)
    (same : scaleAddress p v s h a = scaleAddress p v t k b) : s = t ∧ h = k ∧ a = b := by
  have hw : (s*p.heads+h)*2+a.val = (t*p.heads+k)*2+b.val := by
    unfold scaleAddress at same
    omega
  obtain ⟨hr,hb⟩ := RW.linear_injective 2 _ _ a.val b.val a.isLt b.isLt hw
  obtain ⟨hs,hh'⟩ := RW.linear_injective p.heads s t h k hh hk hr
  exact ⟨hs,hh',Fin.ext hb⟩

theorem q_compatible {α η : Type} (ops : Operations α η) (p : Request) (d : Domain p) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 2 → Byte) (v : Layout) (alloc : NativeAllocations p v)
    (heap : Heap) (hmeta : Metadata p v heap) (s h c t : Nat)
    (hs : s < p.tokens) (hh : h < p.heads) (hc : c < p.columns) (ht : t < p.tokens)
    (event : Write) (he : event ∈ programEvents ops p plan decode encode v heap t)
    (same : event.address = outputAddress p v (p.destination s) h c) :
    event.value = codes ops plan decode v heap s h c := by
  rcases (programEvents_member ops p plan decode encode v heap t event).mp he with
    ⟨k,e,hk,hcol,rfl⟩ | ⟨k,b,hk,rfl⟩
  · simp only [hmeta t ht] at same
    obtain ⟨hd,rfl,rfl⟩ := output_injective p v _ _ k h e c hk hh hcol hc same
    have hts := d.destination_unique t s ht hs hd
    subst t
    rfl
  · have hscale := scale_bound p v alloc (p.destination t) k b (d.destination_bound t ht) hk
    simp only [hmeta t ht] at same
    rw [same] at hscale
    exact False.elim (alloc.separate .output .scale (by decide) _
      (output_bound p v alloc _ h c (d.destination_bound s hs) hh hc) hscale)

theorem m_compatible {α η : Type} (ops : Operations α η) (p : Request) (d : Domain p) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 2 → Byte) (v : Layout) (alloc : NativeAllocations p v)
    (heap : Heap) (hmeta : Metadata p v heap) (s h t : Nat) (b : Fin 2)
    (hs : s < p.tokens) (hh : h < p.heads) (ht : t < p.tokens)
    (event : Write) (he : event ∈ programEvents ops p plan decode encode v heap t)
    (same : event.address = scaleAddress p v (p.destination s) h b) :
    event.value = scales ops plan decode encode v heap s h b := by
  rcases (programEvents_member ops p plan decode encode v heap t event).mp he with
    ⟨k,c,hk,hc,rfl⟩ | ⟨k,a,hk,rfl⟩
  · have hq := output_bound p v alloc (p.destination t) k c (d.destination_bound t ht) hk hc
    simp only [hmeta t ht] at same
    rw [same] at hq
    exact False.elim (alloc.separate .scale .output (by decide) _
      (scale_bound p v alloc _ h b (d.destination_bound s hs) hh) hq)
  · simp only [hmeta t ht] at same
    obtain ⟨hd,rfl,rfl⟩ := scale_injective p v _ _ k h a b hk hh same
    have hts := d.destination_unique t s ht hs hd
    subst t
    rfl

theorem whole_readback {α η : Type} (ops : Operations α η) (p : Request) (d : Domain p) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 2 → Byte) (v : Layout) (alloc : NativeAllocations p v)
    (schedule : Schedule p) (heap : Heap) (hmeta : Metadata p v heap) :
    (∀ s h c, s < p.tokens → h < p.heads → c < p.columns →
      (call ops p plan decode encode v schedule heap).1 (outputAddress p v (p.destination s) h c) =
      codes ops plan decode v heap s h c) ∧
    (∀ s h (b : Fin 2), s < p.tokens → h < p.heads →
      (call ops p plan decode encode v schedule heap).1 (scaleAddress p v (p.destination s) h b) =
      scales ops plan decode encode v heap s h b) := by
  have valid : ∀ s ∈ schedule.order, s < p.tokens := by
    intro s hs; exact List.mem_range.mp (schedule.complete.mem_iff.mp hs)
  have frozen := runGrid_frozen ops p d plan decode encode v alloc schedule.order valid heap heap hmeta
    (by intro _ _ _ _; rfl)
  simp only [call]
  rw [frozen]
  constructor
  · intro s h c hs hh hc
    apply RW.writeAll_readback _ heap ⟨_, codes ops plan decode v heap s h c⟩
    · apply List.mem_flatMap.mpr
      refine ⟨s,schedule.complete.mem_iff.mpr (List.mem_range.mpr hs),?_⟩
      apply (programEvents_member ops p plan decode encode v heap s _).mpr
      exact Or.inl ⟨h,c,hh,hc,by simp only [hmeta s hs]; rfl⟩
    · intro event he same
      obtain ⟨t,ht,he⟩ := List.mem_flatMap.mp he
      exact q_compatible ops p d plan decode encode v alloc heap hmeta s h c t hs hh hc (valid t ht) event he same
  · intro s h b hs hh
    apply RW.writeAll_readback _ heap ⟨_, scales ops plan decode encode v heap s h b⟩
    · apply List.mem_flatMap.mpr
      refine ⟨s,schedule.complete.mem_iff.mpr (List.mem_range.mpr hs),?_⟩
      apply (programEvents_member ops p plan decode encode v heap s _).mpr
      exact Or.inr ⟨h,b,hh,by simp only [hmeta s hs]; rfl⟩
    · intro event he same
      obtain ⟨t,ht,he⟩ := List.mem_flatMap.mp he
      exact m_compatible ops p d plan decode encode v alloc heap hmeta s h t b hs hh (valid t ht) event he same

/-- Distinct programs cannot target the same physical store byte. This is
derived from logical destination uniqueness and native allocation separation. -/
theorem programs_disjoint {α η : Type} (ops : Operations α η) (p : Request) (d : Domain p) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 2 → Byte) (v : Layout) (alloc : NativeAllocations p v)
    (heap : Heap) (hmeta : Metadata p v heap) (s t : Nat) (hs : s < p.tokens) (ht : t < p.tokens) (different : s ≠ t)
    (a b : Write) (ha : a ∈ programEvents ops p plan decode encode v heap s)
    (hb : b ∈ programEvents ops p plan decode encode v heap t) : a.address ≠ b.address := by
  intro same
  rcases (programEvents_member ops p plan decode encode v heap s a).mp ha with
    ⟨h,c,hh,hc,rfl⟩ | ⟨h,byte,hh,rfl⟩ <;>
    rcases (programEvents_member ops p plan decode encode v heap t b).mp hb with
      ⟨k,e,hk,he,rfl⟩ | ⟨k,other,hk,rfl⟩
  · simp only [hmeta s hs,hmeta t ht] at same
    exact different (d.destination_unique s t hs ht (output_injective p v _ _ h k c e hh hk hc he same).1)
  · simp only [hmeta s hs,hmeta t ht] at same
    exact alloc.separate .output .scale (by decide) _
      (output_bound p v alloc _ h c (d.destination_bound s hs) hh hc)
      (by rw [same]; exact scale_bound p v alloc _ k other (d.destination_bound t ht) hk)
  · simp only [hmeta s hs,hmeta t ht] at same
    exact alloc.separate .scale .output (by decide) _
      (scale_bound p v alloc _ h byte (d.destination_bound s hs) hh)
      (by rw [same]; exact output_bound p v alloc _ k e (d.destination_bound t ht) hk he)
  · simp only [hmeta s hs,hmeta t ht] at same
    exact different (d.destination_unique s t hs ht (scale_injective p v _ _ h k byte other hh hk same).1)

theorem whole_frame {α η : Type} (ops : Operations α η) (p : Request) (d : Domain p) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 2 → Byte) (v : Layout) (alloc : NativeAllocations p v)
    (schedule : Schedule p) (heap : Heap) (hmeta : Metadata p v heap) (a : Nat) (outside : ¬ Touched p v a) :
    (call ops p plan decode encode v schedule heap).1 a = heap a := by
  have valid : ∀ s ∈ schedule.order, s < p.tokens := by
    intro s hs; exact List.mem_range.mp (schedule.complete.mem_iff.mp hs)
  rw [call, runGrid_frozen ops p d plan decode encode v alloc schedule.order valid heap heap hmeta (by intro _ _ _ _; rfl)]
  apply RW.writeAll_frame
  intro event he same
  obtain ⟨s,hs,he⟩ := List.mem_flatMap.mp he
  exact outside (same ▸ programEvents_footprint ops p plan decode encode v heap hmeta s (valid s hs) event he)

theorem returned_none {α η : Type} (ops : Operations α η) (p : Request) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 2 → Byte) (v : Layout) (schedule : Schedule p) (heap : Heap) :
    (call ops p plan decode encode v schedule heap).2 = none := rfl

theorem empty_call {α η : Type} (ops : Operations α η) (p : Request) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 2 → Byte) (v : Layout) (schedule : Schedule p) (heap : Heap)
    (empty : p.tokens = 0) : call ops p plan decode encode v schedule heap = (heap, none) := by
  have hs : schedule.order = [] := by simpa [empty] using schedule.complete
  simp [call, runGrid, hs]

end VeriTile.Bench.Repairs.QuantizeCopyKv

#guard_msgs (drop info) in
#auditModuleAxioms
