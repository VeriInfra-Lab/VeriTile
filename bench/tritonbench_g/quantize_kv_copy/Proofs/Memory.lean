import bench.tritonbench_g.rowwise_quantization_triton.Proofs.Memory
import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.quantize_kv_copy.Kernels

namespace VeriTile.Bench.Repairs.QuantizeKvCopy
namespace RW
export VeriTile.Bench.Repairs.RowwiseQuantizationTriton (linear_bound linear_injective writeAll_frame writeAll_readback writeAll_append)
end RW
set_option autoImplicit false
set_option maxHeartbeats 2000000

theorem groupBlock_covers (p : Request) : p.groups ≤ groupBlock p := Nat.le_pow_clog (by decide) _
theorem Domain.columns_positive {p : Request} (_d : Domain p) : 0 < groupWidth := by decide

theorem program_token_bound (p : Request) (s : Nat) (hs : s < p.programs) : s / p.headCount < p.tokens := by
  have hp : 0 < p.headCount := by
    by_contra h
    have hz : p.headCount = 0 := by omega
    simp [Request.programs,hz] at hs
  exact (Nat.div_lt_iff_lt_mul hp).mpr hs

theorem Domain.program_sources {p : Request} (d : Domain p) : p.programs ≤ p.sourcePrograms :=
  Nat.mul_le_mul_right p.headCount d.sources

theorem Domain.program_destination_bound {p : Request} (d : Domain p) (s : Nat) (hs : s < p.programs) :
    p.destination s < p.outputPrograms :=
  RW.linear_bound p.capacity p.headCount _ _ (d.destination_bound _ (program_token_bound p s hs))
    (Nat.mod_lt s d.heads_positive)

theorem Domain.program_destination_unique {p : Request} (d : Domain p) (s t : Nat)
    (hs : s < p.programs) (ht : t < p.programs) (same : p.destination s = p.destination t) : s = t := by
  obtain ⟨slots,heads⟩ := RW.linear_injective p.headCount _ _ _ _
    (Nat.mod_lt s d.heads_positive) (Nat.mod_lt t d.heads_positive) same
  have tokens := d.destination_unique _ _ (program_token_bound p s hs) (program_token_bound p t ht) slots
  have se := Nat.mod_add_div s p.headCount
  have te := Nat.mod_add_div t p.headCount
  rw [tokens,heads] at se
  omega

theorem tensor_bound (p : Request) (v : Layout) (alloc : NativeAllocations p v)
    (r : Role) (off : Nat) (h : off < elements p r) :
    (v.tensor r).Contains ((v.tensor r).data + off) := by
  have hc := alloc.capacity r
  unfold Tensor.Contains Tensor.data RW.InRange
  constructor <;> omega

theorem input_bound (p : Request) (v : Layout) (alloc : NativeAllocations p v)
    (s h c b : Nat) (hs : s < p.sourcePrograms) (hh : h < p.groups) (hc : c < groupWidth) (hb : b < 4) :
    (v.tensor .input).Contains (inputAddress p v s h c + b) := by
  have hr := RW.linear_bound p.sourcePrograms p.groups s h hs hh
  have hi := RW.linear_bound (p.sourcePrograms*p.groups) groupWidth (s*p.groups+h) c hr hc
  have bound : 4*((s*p.groups+h)*groupWidth+c)+b < elements p .input := by simp only [elements]; omega
  simpa [inputAddress, Nat.add_assoc] using tensor_bound p v alloc .input _ bound

theorem dest_bound (p : Request) (v : Layout) (alloc : NativeAllocations p v)
    (s b : Nat) (hs : s < p.programs) (hb : b < 4) :
    (v.tensor .destination).Contains (destAddress p v s + b) := by
  have hi := program_token_bound p s hs
  have ht : p.tokens ≠ 0 := Nat.ne_of_gt (lt_of_le_of_lt (Nat.zero_le _) hi)
  have hm := Nat.mul_le_mul_right p.destStride (Nat.le_pred_of_lt hi)
  simp only [Nat.pred_eq_sub_one] at hm
  have bound : 4*((s / p.headCount)*p.destStride)+b < elements p .destination := by
    simp only [elements, if_neg ht]; omega
  simpa [destAddress, Nat.add_assoc] using tensor_bound p v alloc .destination _ bound

theorem output_bound (p : Request) (v : Layout) (alloc : NativeAllocations p v)
    (slot h c : Nat) (hs : slot < p.outputPrograms) (hh : h < p.groups) (hc : c < groupWidth) :
    (v.tensor .output).Contains (outputAddress p v slot h c) := by
  have hr := RW.linear_bound p.outputPrograms p.groups slot h hs hh
  exact tensor_bound p v alloc .output _
    (RW.linear_bound (p.outputPrograms*p.groups) groupWidth (slot*p.groups+h) c hr hc)

theorem scale_bound (p : Request) (v : Layout) (alloc : NativeAllocations p v)
    (slot h : Nat) (b : Fin 4) (hs : slot < p.outputPrograms) (hh : h < p.groups) :
    (v.tensor .scale).Contains (scaleAddress p v slot h b) := by
  have hr := RW.linear_bound p.outputPrograms p.groups slot h hs hh
  have hb := b.isLt
  exact tensor_bound p v alloc .scale (4*(slot*p.groups+h)+b.val) (by simp only [elements]; omega)

theorem programEvents_member {α η : Type} (ops : Operations α η) (p : Request) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 4 → Byte)
    (v : Layout) (heap : Heap) (s : Nat) (event : Write) :
    event ∈ programEvents ops p plan decode encode v heap s ↔
    (∃ h c, h < p.groups ∧ c < groupWidth ∧ event =
      ⟨outputAddress p v (loadTarget p heap (destAddress p v s) s) h c, codes ops plan decode v heap s h c⟩) ∨
    (∃ h, ∃ b : Fin 4, h < p.groups ∧ event =
      ⟨scaleAddress p v (loadTarget p heap (destAddress p v s) s) h b, scales ops plan decode encode v heap s h b⟩) := by
  simp only [programEvents, List.mem_append, List.mem_flatMap, List.mem_range]
  constructor
  · rintro (⟨h, _, he⟩ | ⟨h, _, he⟩)
    · by_cases hh : h < p.groups
      · simp only [if_pos hh, List.mem_map, List.mem_range] at he
        obtain ⟨c, hc, rfl⟩ := he
        exact Or.inl ⟨h,c,hh,hc,rfl⟩
      · simp [hh] at he
    · by_cases hh : h < p.groups
      · simp only [if_pos hh, List.mem_map, List.mem_finRange, true_and] at he
        obtain ⟨b,rfl⟩ := he
        exact Or.inr ⟨h,b,hh,rfl⟩
      · simp [hh] at he
  · rintro (⟨h,c,hh,hc,rfl⟩ | ⟨h,b,hh,rfl⟩)
    · refine Or.inl ⟨h,lt_of_lt_of_le hh (groupBlock_covers p),?_⟩
      simp only [if_pos hh, List.mem_map, List.mem_range]
      exact ⟨c,hc,rfl⟩
    · refine Or.inr ⟨h,lt_of_lt_of_le hh (groupBlock_covers p),?_⟩
      simp only [if_pos hh, List.mem_map, List.mem_finRange, true_and]
      exact ⟨b,rfl⟩

theorem programEvents_footprint {α η : Type} (ops : Operations α η) (p : Request) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 4 → Byte)
    (v : Layout) (heap : Heap) (hmeta : Metadata p v heap) (s : Nat) (hs : s < p.programs)
    (event : Write) (he : event ∈ programEvents ops p plan decode encode v heap s) :
    Touched p v event.address := by
  rcases (programEvents_member ops p plan decode encode v heap s event).mp he with
    ⟨h,c,hh,hc,rfl⟩ | ⟨h,b,hh,rfl⟩
  · exact Or.inl ⟨s,h,c,hs,hh,hc,by simp [hmeta s hs]⟩
  · exact Or.inr ⟨s,h,b,hs,hh,by simp [hmeta s hs]⟩

theorem touched_output (p : Request) (d : Domain p) (v : Layout) (alloc : NativeAllocations p v)
    (a : Nat) (ha : Touched p v a) : (v.tensor .output).Contains a ∨ (v.tensor .scale).Contains a := by
  rcases ha with ⟨s,h,c,hs,hh,hc,rfl⟩ | ⟨s,h,b,hs,hh,rfl⟩
  · exact Or.inl (output_bound p v alloc _ h c (d.program_destination_bound s hs) hh hc)
  · exact Or.inr (scale_bound p v alloc _ h b (d.program_destination_bound s hs) hh)

theorem input_untouched (p : Request) (d : Domain p) (v : Layout) (alloc : NativeAllocations p v)
    (r : Role) (hr : r = .input ∨ r = .destination) (a : Nat) (inside : (v.tensor r).Contains a) :
    ¬ Touched p v a := by
  intro touched
  rcases touched_output p d v alloc a touched with hq | hm
  · exact alloc.separate r .output (by rcases hr with rfl | rfl <;> decide) a inside hq
  · exact alloc.separate r .scale (by rcases hr with rfl | rfl <;> decide) a inside hm

def SameReads (v : Layout) (before after : Heap) : Prop :=
  ∀ r, r = .input ∨ r = .destination → ∀ a, (v.tensor r).Contains a → after a = before a

theorem loadDest_same (p : Request) (v : Layout) (alloc : NativeAllocations p v)
    (before after : Heap) (same : SameReads v before after) (s : Nat) (hs : s < p.programs) :
    loadTarget p after (destAddress p v s) s = loadTarget p before (destAddress p v s) s := by
  have hd : loadDest after (destAddress p v s) = loadDest before (destAddress p v s) := by
    unfold loadDest
    congr 1
    apply List.map_congr_left
    intro b hb
    rw [same .destination (Or.inr rfl) _ (dest_bound p v alloc s b hs (List.mem_range.mp hb))]
  simp only [loadTarget, hd]

theorem metadata_same (p : Request) (v : Layout) (alloc : NativeAllocations p v)
    (before after : Heap) (same : SameReads v before after) (hmeta : Metadata p v before) : Metadata p v after := by
  intro s hs
  exact (loadDest_same p v alloc before after same s hs).trans (hmeta s hs)

theorem loadInput_same {α : Type} (p : Request) (d : Domain p) (v : Layout) (alloc : NativeAllocations p v)
    (decode : (Fin 4 → Byte) → α) (before after : Heap) (same : SameReads v before after)
    (s h c : Nat) (hs : s < p.programs) (hh : h < p.groups) (hc : c < groupWidth) :
    loadInput decode p v after s h c = loadInput decode p v before s h c := by
  unfold loadInput
  congr 1
  funext b
  exact same .input (Or.inl rfl) _ (input_bound p v alloc s h c b.val (lt_of_lt_of_le hs d.program_sources) hh hc b.isLt)

theorem sourceRegisters_congr {α η : Type} (ops : Operations α η) (p : Request) (plan : Plan p)
    (x y : Nat → α) (same : ∀ c, c < groupWidth → x c = y c) :
    (∀ c, c < groupWidth → (sourceRegisters ops plan.tree x).1 c = (sourceRegisters ops plan.tree y).1 c) ∧
    (sourceRegisters ops plan.tree x).2 = (sourceRegisters ops plan.tree y).2 := by
  have hm : rowMaximum ops plan.tree x = rowMaximum ops plan.tree y := by
    apply VeriTile.Bench.Repairs.RowwiseQuantizationTriton.Tree.evaluate_congr
    intro c hc
    rw [same c (List.mem_range.mp (plan.covers.mem_iff.mp hc))]
  change (∀ c, c < groupWidth → rowCode ops (x c) (rowMaximum ops plan.tree x) = rowCode ops (y c) (rowMaximum ops plan.tree y)) ∧
    rowScale ops (rowMaximum ops plan.tree x) = rowScale ops (rowMaximum ops plan.tree y)
  exact ⟨fun c hc => by rw [hm,same c hc], by rw [hm]⟩

theorem programEvents_same {α η : Type} (ops : Operations α η) (p : Request) (d : Domain p) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 4 → Byte)
    (v : Layout) (alloc : NativeAllocations p v) (before after : Heap) (same : SameReads v before after)
    (s : Nat) (hs : s < p.programs) :
    programEvents ops p plan decode encode v after s = programEvents ops p plan decode encode v before s := by
  have hd := loadDest_same p v alloc before after same s hs
  have hreg (h : Nat) (hh : h < p.groups) := sourceRegisters_congr ops p plan
    (loadInput decode p v after s h) (loadInput decode p v before s h)
    (fun c hc => loadInput_same p d v alloc decode before after same s h c hs hh hc)
  unfold programEvents
  rw [hd]
  dsimp only
  congr 1
  · apply List.flatMap_congr
    intro h _
    by_cases hh : h < p.groups
    · simp only [if_pos hh]
      apply List.map_congr_left
      intro c hc
      simp only [codes, (hreg h hh).1 c (List.mem_range.mp hc)]
    · simp [hh]
  · apply List.flatMap_congr
    intro h _
    by_cases hh : h < p.groups
    · simp only [if_pos hh]
      apply List.map_congr_left
      intro b _
      simp only [scales, (hreg h hh).2]
    · simp [hh]

end VeriTile.Bench.Repairs.QuantizeKvCopy

#guard_msgs (drop info) in
#auditModuleAxioms
