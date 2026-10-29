import bench.repairs.tritonbench_g.quantize_kv_transform.Proofs.Contract

namespace VeriTile.Bench.Repairs.QuantizeKvTransform.Native
noncomputable section
set_option maxHeartbeats 3000000

theorem signed_exact (n : Nat) (bound : n<2^63) : (signed n).toInt=(n:Int) :=
  BitVec.toInt_ofInt_eq_self (w:=64) (n:=(n:Int)) (by decide) (by omega) (by norm_num; omega)

theorem index_word (p : Request) (s h c : Nat) :
    inputIndex p s h c=signed ((s*p.heads+h)*p.columns+c) := by
  simp [inputIndex,signed,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,Nat.add_mul,Nat.mul_assoc]

theorem active_index (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w)
    (s a c : Nat) (hs : s<(request m w).tokens) (ha : a<(request m w).heads)
    (hc : c<(request m w).columns) :
    (inputIndex (request m w) s a c).toInt=(((s*(request m w).heads+a)*(request m w).columns+c:Nat):Int) ∧
    (outputIndex (request m w) ((request m w).destination s) a c).toInt=
      (((request m w).destination s*(request m w).heads+a)*(request m w).columns+c:Nat) := by
  have d := checked_domain m w h
  have inputBound := RW.linear_bound (request m w).sourceRows (request m w).heads s a
    (lt_of_lt_of_le hs d.sources) ha
  have inputFlat := RW.linear_bound ((request m w).sourceRows*(request m w).heads) (request m w).columns
    (s*(request m w).heads+a) c inputBound hc
  have outputBound := RW.linear_bound (request m w).capacity (request m w).heads ((request m w).destination s) a
    (d.destination_bound s hs) ha
  have outputFlat := RW.linear_bound ((request m w).capacity*(request m w).heads) (request m w).columns
    ((request m w).destination s*(request m w).heads+a) c outputBound hc
  have inSize := sizes m w h .input
  have outSize := sizes m w h .output
  simp only [elements,elementBytes] at inSize outSize
  have inputSmall := h.extent .input
  have outputSmall := h.extent .output
  change (inputIndex (request m w) s a c).toInt = _ ∧
    (inputIndex (request m w) ((request m w).destination s) a c).toInt=_
  rw [index_word,index_word,signed_exact _ (by omega),signed_exact _ (by omega)]
  exact ⟨rfl,rfl⟩

theorem pointer_exact (m : Inputs) (w : Nat → MetadataWord) (objects : Role → Nat)
    (heap : Heap) (binding : TensorBinding m w heap) (r : Role) (a : Nat)
    (inside : (directLayout m objects |>.tensor r).Contains a) :
    (BitVec.ofNat 64 a).toNat=a ∧ a<2^64 := by
  change base m r≤a ∧ a<base m r+elementBytes r*(m r).numel at inside
  have cap := binding.capacity r
  have bound : a<2^64 := by omega
  exact ⟨by rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt bound],bound⟩

/-- Every actual source byte access is aligned, backed and nonwrapping. -/
theorem accesses (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w)
    (objects : Role → Nat) (heap : Heap) (binding : TensorBinding m w heap)
    (s a c : Nat) (hs : s<(request m w).tokens) (ha : a<(request m w).heads)
    (hc : c<(request m w).columns) :
    (∀ b, b<4 → inputAddress (request m w) (directLayout m objects) s a c+b<2^64) ∧
    (∀ b, b<4 → destAddress (request m w) (directLayout m objects) s+b<2^64) ∧
    outputAddress (request m w) (directLayout m objects) ((request m w).destination s) a c<2^64 ∧
    (∀ b : Fin 4, scaleAddress (request m w) (directLayout m objects) ((request m w).destination s) a b<2^64) := by
  have d := checked_domain m w h
  have alloc := direct_allocations m w h objects
  refine ⟨?_,?_,?_,?_⟩
  · intro b hb
    exact (pointer_exact m w objects heap binding .input _
      (input_bound _ _ alloc s a c b (lt_of_lt_of_le hs d.sources) ha hc hb)).2
  · intro b hb
    exact (pointer_exact m w objects heap binding .destination _ (dest_bound _ _ alloc s b hs hb)).2
  · exact (pointer_exact m w objects heap binding .output _
      (output_bound _ _ alloc _ a c (d.destination_bound s hs) ha hc)).2
  · intro b
    exact (pointer_exact m w objects heap binding .scale _
      (scale_bound _ _ alloc _ a b (d.destination_bound s hs) ha)).2

theorem block_bounds (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w) :
    headBlock (request m w)≤2^20 ∧ columnBlock (request m w)≤2^20 := by
  have bh : 0<headBlock (request m w) := by unfold headBlock; positivity
  have bd : 0<columnBlock (request m w) := by unfold columnBlock; positivity
  have t := h.tile
  constructor <;> nlinarith

theorem mask_exact (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w)
    (a : Nat) (ha : a<headBlock (request m w)) :
    headMask (request m w) a=decide (a<(request m w).heads) := by
  have bounds := block_bounds m w h
  have heads := headBlock_covers (request m w)
  rw [headMask,BitVec.slt_eq_decide,signed_exact _ (by omega),signed_exact _ (by omega)]
  simp

theorem column_mask_exact (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w)
    (c : Nat) (hc : c<columnBlock (request m w)) :
    columnMask (request m w) c=decide (c<(request m w).columns) := by
  have bounds := block_bounds m w h
  have cols := columnBlock_covers (request m w)
  rw [columnMask,BitVec.slt_eq_decide,signed_exact _ (by omega),signed_exact _ (by omega)]
  simp

/-- Program IDs and arange values are first produced in signed int32 and then
widened; grid and tile guards make both casts value preserving. -/
theorem controls (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w)
    (s a c : Nat) (hs : s<(request m w).tokens) (ha : a<headBlock (request m w))
    (hc : c<columnBlock (request m w)) :
    (BitVec.ofNat 32 s).toInt=(s:Int) ∧ (BitVec.ofNat 32 a).toInt=(a:Int) ∧
      (BitVec.ofNat 32 c).toInt=(c:Int) := by
  have bounds := block_bounds m w h
  have grid := h.grid
  have exact32 : ∀ n : Nat, n<2^31 → (BitVec.ofNat 32 n).toInt=(n:Int) := by
    intro n hn
    exact BitVec.toInt_ofInt_eq_self (w:=32) (n:=(n:Int)) (by decide) (by omega) (by norm_num; omega)
  exact ⟨exact32 s (by omega),exact32 a (by omega),exact32 c (by omega)⟩

theorem pointer_value (base index bytes : Nat) (bound : base+index*bytes<2^64) :
    (pointer base (signed index) bytes).toNat=base+index*bytes := by
  simp only [pointer,signed,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
  exact Nat.mod_eq_of_lt bound

#axiomsClean mask_exact
#axiomsClean controls
#axiomsClean pointer_value
#axiomsClean active_index
#axiomsClean pointer_exact
#axiomsClean accesses
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.QuantizeKvTransform.Native
