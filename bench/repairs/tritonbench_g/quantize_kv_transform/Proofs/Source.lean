import bench.repairs.tritonbench_g.quantize_kv_transform.Proofs.Native

namespace VeriTile.Bench.Repairs.QuantizeKvTransform.Native
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 100000

variable {α η : Type} (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w)
    (objects : Role → Nat) (heap : Heap) (binding : TensorBinding m w heap)
include m w h objects heap binding

theorem input_pointer (s a c : Nat) (hs : s<(request m w).tokens)
    (ha : a<(request m w).heads) (hc : c<(request m w).columns) :
    (pointer (directLayout m objects |>.tensor .input).data (inputIndex (request m w) s a c) 4).toNat=
      inputAddress (request m w) (directLayout m objects) s a c := by
  have bound := (accesses m w h objects heap binding s a c hs ha hc).1 0 (by decide)
  rw [index_word,pointer_value _ _ _ (by simpa [inputAddress,Nat.mul_comm] using bound)]
  simp [inputAddress,Nat.mul_comm]

theorem output_pointer (s a c : Nat) (hs : s<(request m w).tokens)
    (ha : a<(request m w).heads) (hc : c<(request m w).columns) :
    (pointer (directLayout m objects |>.tensor .output).data
      (outputIndex (request m w) ((request m w).destination s) a c) 1).toNat=
      outputAddress (request m w) (directLayout m objects) ((request m w).destination s) a c := by
  have bound := (accesses m w h objects heap binding s a c hs ha hc).2.2.1
  change (pointer _ (inputIndex _ _ _ _) 1).toNat=_
  rw [index_word,pointer_value _ _ _ (by simpa [outputAddress] using bound)]
  simp [outputAddress]

theorem scale_pointer (s a : Nat) (b : Fin 4) (hs : s<(request m w).tokens)
    (ha : a<(request m w).heads) :
    (pointer (directLayout m objects |>.tensor .scale).data
      (scaleIndex (request m w) ((request m w).destination s) a) 4).toNat+b.val=
      scaleAddress (request m w) (directLayout m objects) ((request m w).destination s) a b := by
  have hc := h.width.2
  have bound := (accesses m w h objects heap binding s a 0 hs ha hc).2.2.2 ⟨0,by decide⟩
  have word : scaleIndex (request m w) ((request m w).destination s) a=
      signed ((request m w).destination s*(request m w).heads+a) := by
    simp [scaleIndex,signed,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
  rw [word,pointer_value _ _ _ (by simpa [scaleAddress,Nat.mul_comm] using bound)]
  simp [scaleAddress,Nat.mul_comm,Nat.add_assoc]

theorem destination_exact (s : Nat) (hs : s<(request m w).tokens) :
    destination (request m w) (directLayout m objects) heap s=(request m w).destination s := by
  have d := checked_domain m w h
  have alloc := direct_allocations m w h objects
  have bound := (pointer_exact m w objects heap binding .destination _
    (dest_bound _ _ alloc s 0 hs (by decide))).2
  unfold destination
  rw [pointer_value _ _ _ (by simpa [destAddress,Nat.mul_comm] using bound)]
  have hmeta := direct_metadata m w objects heap binding s hs
  rw [Nat.mul_comm (s*(request m w).destStride) 4]
  change loadDest heap ((directLayout m objects |>.tensor .destination).data+4*(s*(request m w).destStride))=_ at hmeta
  rw [hmeta]
  have word : (BitVec.ofNat 32 ((request m w).destination s)).toInt=((request m w).destination s:Int) :=
    BitVec.toInt_ofInt_eq_self (w:=32) (n:=((request m w).destination s:Int))
      (by decide) (by omega) (by have := d.destination_int32 s hs; norm_num at *; omega)
  rw [word]
  change (signed ((request m w).destination s)).toInt.toNat=_
  rw [signed_exact _ (lt_trans (d.destination_int32 s hs) (by norm_num))]
  simp

theorem load_exact (decode : (Fin 4 → Byte) → α) (s a c : Nat)
    (hs : s<(request m w).tokens) (ha : a<(request m w).heads) (hc : c<columnBlock (request m w)) :
    load decode (request m w) (directLayout m objects) heap s a c=
      loadInput decode (request m w) (directLayout m objects) heap s a c := by
  unfold load loadInput
  rw [column_mask_exact m w h c hc]
  by_cases active : c<(request m w).columns
  · simp only [active,decide_true,ite_true]
    rw [input_pointer m w h objects heap binding s a c hs ha active]
  · simp [active]

theorem registers (ops : Operations α η) (plan : Plan (request m w))
    (decode : (Fin 4 → Byte) → α) (s a : Nat) (hs : s<(request m w).tokens) (ha : a<(request m w).heads) :
    (∀ c, c<(request m w).columns →
      (sourceRegisters ops plan.tree (load decode (request m w) (directLayout m objects) heap s a)).1 c=
      codes ops plan decode (directLayout m objects) heap s a c) ∧
    (sourceRegisters ops plan.tree (load decode (request m w) (directLayout m objects) heap s a)).2=
      (sourceRegisters ops plan.tree (loadInput decode (request m w) (directLayout m objects) heap s a)).2 := by
  have maximum : plan.tree.evaluate ops.maximum (fun c => ops.abs (load decode (request m w) (directLayout m objects) heap s a c))=
      plan.tree.evaluate ops.maximum (fun c => ops.abs (loadInput decode (request m w) (directLayout m objects) heap s a c)) := by
    apply RowwiseQuantizationTriton.Tree.evaluate_congr
    intro c hc
    rw [load_exact m w h objects heap binding decode s a c hs ha (List.mem_range.mp (plan.covers.mem_iff.mp hc))]
  constructor
  · intro c hc
    simp only [sourceRegisters,codes,maximum]
    rw [load_exact m w h objects heap binding decode s a c hs ha (lt_of_lt_of_le hc (columnBlock_covers _))]
  · simp only [sourceRegisters,maximum]

omit m w h objects heap binding in
private theorem masked_range {β : Type} (n block : Nat) (bound : n≤block) (f : Nat → β) :
    (List.range block).filterMap (fun c => if c<n then some (f c) else none)=(List.range n).map f := by
  induction block with
  | zero =>
    have hn : n=0 := by omega
    subst n
    rfl
  | succ block ih =>
    rw [List.range_succ,List.filterMap_append]
    by_cases hn : n≤block
    · rw [ih hn]
      simp [show ¬block<n by omega]
    · have hn' : n=block+1 := by omega
      subst n
      have eq : (List.range block).filterMap (fun c => if c<block+1 then some (f c) else none) =
          (List.range block).map f := by
        have eq : (List.range block).filterMap (fun c => if c<block+1 then some (f c) else none) =
            (List.range block).filterMap (fun c => some (f c)) := by
          apply List.filterMap_congr
          intro c hc
          simp [show c<block+1 by have := List.mem_range.mp hc; omega]
        simpa using eq
      rw [eq,List.range_succ,List.map_append]
      simp

theorem program_exact (ops : Operations α η) (plan : Plan (request m w))
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 4 → Byte)
    (s : Nat) (hs : s<(request m w).tokens) :
    programEvents ops (request m w) plan decode encode (directLayout m objects) heap s=
      QuantizeKvTransform.programEvents ops (request m w) plan decode encode (directLayout m objects) heap s := by
  have hmeta := direct_metadata m w objects heap binding s hs
  simp only [programEvents,QuantizeKvTransform.programEvents,destination_exact m w h objects heap binding s hs,hmeta]
  congr 1
  · apply List.flatMap_congr
    intro a member
    rw [mask_exact m w h a (List.mem_range.mp member)]
    by_cases ha : a<(request m w).heads
    · simp only [ha,decide_true,ite_true]
      have eq : (List.range (columnBlock (request m w))).filterMap (fun c =>
          if columnMask (request m w) c then some
            (⟨(pointer (directLayout m objects |>.tensor .output).data
                (outputIndex (request m w) ((request m w).destination s) a c) 1).toNat,
              (sourceRegisters ops plan.tree (load decode (request m w) (directLayout m objects) heap s a)).1 c⟩ : Write)
          else none) =
          (List.range (columnBlock (request m w))).filterMap (fun c =>
            if c<(request m w).columns then some
              (⟨outputAddress (request m w) (directLayout m objects) ((request m w).destination s) a c,
                codes ops plan decode (directLayout m objects) heap s a c⟩ : Write) else none) := by
        apply List.filterMap_congr
        intro c member
        rw [column_mask_exact m w h c (List.mem_range.mp member)]
        by_cases active : c<(request m w).columns
        · simp only [active,decide_true,ite_true]
          rw [output_pointer m w h objects heap binding s a c hs ha active,
            (registers m w h objects heap binding ops plan decode s a hs ha).1 c active]
        · simp [active]
      rw [eq,masked_range _ _ (columnBlock_covers _)]
    · simp [ha]
  · apply List.flatMap_congr
    intro a member
    rw [mask_exact m w h a (List.mem_range.mp member)]
    by_cases ha : a<(request m w).heads
    · simp only [ha,decide_true,ite_true]
      apply List.map_congr_left
      intro b _
      rw [scale_pointer m w h objects heap binding s a b hs ha,
        (registers m w h objects heap binding ops plan decode s a hs ha).2]
      rfl
    · simp [ha]

theorem program_binding (ops : Operations α η) (plan : Plan (request m w))
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 4 → Byte)
    (s : Nat) (hs : s<(request m w).tokens) :
    TensorBinding m w (QuantizeKvTransform.runProgram ops (request m w) plan decode encode (directLayout m objects) heap s) := by
  refine ⟨binding.capacity,?_⟩
  intro t ht
  have reads := runProgram_sameReads ops (request m w) (checked_domain m w h) plan decode encode
    (directLayout m objects) (direct_allocations m w h objects) heap
    (direct_metadata m w objects heap binding) s hs
  have metadata := metadata_same (request m w) (directLayout m objects) (direct_allocations m w h objects)
    heap _ reads (direct_metadata m w objects heap binding)
  have value := metadata t ht
  change loadDest _ (base m .destination + 4*(t*1)) = (w t).toInt.toNat at value
  simpa only [Nat.mul_one] using value

theorem program_step (ops : Operations α η) (plan : Plan (request m w))
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 4 → Byte)
    (s : Nat) (hs : s<(request m w).tokens) :
    runProgram ops (request m w) plan decode encode (directLayout m objects) heap s=
      QuantizeKvTransform.runProgram ops (request m w) plan decode encode (directLayout m objects) heap s := by
  unfold runProgram QuantizeKvTransform.runProgram
  rw [program_exact m w h objects heap binding ops plan decode encode s hs]

omit m w h objects heap binding in
private theorem foldl_equiv (f g : Heap → Nat → Heap) (P : Heap → Prop) (Q : Nat → Prop)
    (step : ∀ b, P b → ∀ s, Q s → f b s=g b s)
    (keep : ∀ b, P b → ∀ s, Q s → P (g b s))
    (order : List Nat) (valid : ∀ s∈order,Q s) (initial : Heap) (hp : P initial) :
    order.foldl f initial=order.foldl g initial := by
  induction order generalizing initial with
  | nil => rfl
  | cons s rest ih =>
    have hs := valid s (by simp)
    simp only [List.foldl_cons]
    rw [step initial hp s hs]
    exact ih (fun t ht => valid t (by simp [ht])) _ (keep initial hp s hs)

theorem grid_exact (ops : Operations α η) (plan : Plan (request m w))
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 4 → Byte)
    (order : List Nat) (valid : ∀ s∈order, s<(request m w).tokens) :
    runGrid ops (request m w) plan decode encode (directLayout m objects) order heap=
      QuantizeKvTransform.runGrid ops (request m w) plan decode encode (directLayout m objects) order heap := by
  exact foldl_equiv _ _ (TensorBinding m w) (fun s => s<(request m w).tokens)
    (fun b hb s hs => program_step m w h objects b hb ops plan decode encode s hs)
    (fun b hb s hs => program_binding m w h objects b hb ops plan decode encode s hs)
    order valid heap binding

#axiomsClean program_exact
#axiomsClean grid_exact
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.QuantizeKvTransform.Native
