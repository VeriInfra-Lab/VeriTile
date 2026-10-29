import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.Contract

namespace VeriTile.Bench.Repairs.QuantizeKvCopy.Native
noncomputable section
set_option maxHeartbeats 3000000

theorem signed_exact (n : Nat) (bound : n<2^63) : (signed n).toInt=(n:Int) :=
  BitVec.toInt_ofInt_eq_self (w:=64) (n:=(n:Int)) (by decide) (by omega) (by norm_num; omega)

theorem index_word (p : Request) (s h c : Nat) :
    inputIndex p s h c=signed ((s*p.groups+h)*groupWidth+c) := by
  have split : s/p.headCount*p.headCount+s%p.headCount=s := by
    simpa only [Nat.mul_comm] using Nat.div_add_mod s p.headCount
  simp only [inputIndex,signed,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
  congr 1
  calc
    _ = (s/p.headCount*p.headCount+s%p.headCount)*p.groups*groupWidth+h*groupWidth+c := by ring
    _ = _ := by rw [split]; ring

theorem output_word (p : Request) (s h c : Nat) :
    outputIndex p s h c=signed ((s*p.groups+h)*groupWidth+c) := by
  simp [outputIndex,signed,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,Nat.add_mul,Nat.mul_assoc]

theorem active_index (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w)
    (s a c : Nat) (hs : s<(request m w).programs) (ha : a<(request m w).groups)
    (hc : c<groupWidth) :
    (inputIndex (request m w) s a c).toInt=(((s*(request m w).groups+a)*groupWidth+c:Nat):Int) ∧
    (outputIndex (request m w) ((request m w).destination s) a c).toInt=
      (((request m w).destination s*(request m w).groups+a)*groupWidth+c:Nat) := by
  have d := checked_domain m w h
  have inputBound := RW.linear_bound (request m w).sourcePrograms (request m w).groups s a
    (lt_of_lt_of_le hs d.program_sources) ha
  have inputFlat := RW.linear_bound ((request m w).sourcePrograms*(request m w).groups) groupWidth
    (s*(request m w).groups+a) c inputBound hc
  have outputBound := RW.linear_bound (request m w).outputPrograms (request m w).groups ((request m w).destination s) a
    (d.program_destination_bound s hs) ha
  have outputFlat := RW.linear_bound ((request m w).outputPrograms*(request m w).groups) groupWidth
    ((request m w).destination s*(request m w).groups+a) c outputBound hc
  have inSize := sizes m w h .input
  have outSize := sizes m w h .output
  simp only [elements,elementBytes] at inSize outSize
  have inputSmall := h.extent .input
  have outputSmall := h.extent .output
  rw [index_word,output_word,signed_exact _ (by omega),signed_exact _ (by omega)]
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
    (s a c : Nat) (hs : s<(request m w).programs) (ha : a<(request m w).groups)
    (hc : c<groupWidth) :
    (∀ b, b<4 → inputAddress (request m w) (directLayout m objects) s a c+b<2^64) ∧
    (∀ b, b<4 → destAddress (request m w) (directLayout m objects) s+b<2^64) ∧
    outputAddress (request m w) (directLayout m objects) ((request m w).destination s) a c<2^64 ∧
    (∀ b : Fin 4, scaleAddress (request m w) (directLayout m objects) ((request m w).destination s) a b<2^64) := by
  have d := checked_domain m w h
  have alloc := direct_allocations m w h objects
  refine ⟨?_,?_,?_,?_⟩
  · intro b hb
    exact (pointer_exact m w objects heap binding .input _
      (input_bound _ _ alloc s a c b (lt_of_lt_of_le hs d.program_sources) ha hc hb)).2
  · intro b hb
    exact (pointer_exact m w objects heap binding .destination _ (dest_bound _ _ alloc s b hs hb)).2
  · exact (pointer_exact m w objects heap binding .output _
      (output_bound _ _ alloc _ a c (d.program_destination_bound s hs) ha hc)).2
  · intro b
    exact (pointer_exact m w objects heap binding .scale _
      (scale_bound _ _ alloc _ a b (d.program_destination_bound s hs) ha)).2

theorem block_bounds (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w) :
    groupBlock (request m w)≤2^20 ∧ groupWidth≤2^20 := by
  have bh : 0<groupBlock (request m w) := by unfold groupBlock; positivity
  have bd : 0<groupWidth := by decide
  have t : groupBlock (request m w)*groupWidth≤2^20 := h.tile
  constructor <;> nlinarith

theorem mask_exact (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w)
    (a : Nat) (ha : a<groupBlock (request m w)) :
    groupMask (request m w) a=decide (a<(request m w).groups) := by
  have bounds := block_bounds m w h
  have heads := groupBlock_covers (request m w)
  rw [groupMask,BitVec.slt_eq_decide,signed_exact _ (by omega),signed_exact _ (by omega)]
  simp

/-- The actual two-dimensional IDs and arange coordinates widen exactly. -/
theorem controls (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w)
    (t head a c : Nat) (ht : t<(request m w).tokens) (hh : head<(request m w).headCount)
    (ha : a<groupBlock (request m w)) (hc : c<groupWidth) :
    (BitVec.ofNat 32 t).toInt=(t:Int) ∧ (BitVec.ofNat 32 head).toInt=(head:Int) ∧
    (BitVec.ofNat 32 a).toInt=(a:Int) ∧ (BitVec.ofNat 32 c).toInt=(c:Int) := by
  have bounds := block_bounds m w h
  have grid := h.grid
  have exact32 : ∀ n : Nat, n<2^31 → (BitVec.ofNat 32 n).toInt=(n:Int) := by
    intro n hn
    exact BitVec.toInt_ofInt_eq_self (w:=32) (n:=(n:Int)) (by decide) (by omega) (by norm_num; omega)
  exact ⟨exact32 t (by omega),exact32 head (by omega),exact32 a (by omega),exact32 c (by omega)⟩

theorem pointer_value (base index bytes : Nat) (bound : base+index*bytes<2^64) :
    (pointer base (signed index) bytes).toNat=base+index*bytes := by
  simp only [pointer,signed,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
  exact Nat.mod_eq_of_lt bound

/-- The repaired pointer SSA retains every original view stride. Flattening
only renames the two launch coordinates; this identity is in BitVec64. -/
theorem output_ssa (p : Request) (dest head group lane : Nat) :
    signed dest*signed (p.headCount*p.groups*8)+signed head*signed (p.groups*8)+
      signed group*signed 8+signed lane=outputIndex p (dest*p.headCount+head) group lane := by
  simp only [outputIndex,signed,groupWidth,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
  congr 1
  ring

theorem scale_ssa (p : Request) (dest head group : Nat) :
    signed dest*signed (p.headCount*p.groups)+signed head*signed p.groups+signed group=
      scaleIndex p (dest*p.headCount+head) group := by
  simp only [scaleIndex,signed,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
  congr 1
  ring

#axiomsClean output_ssa
#axiomsClean scale_ssa
#axiomsClean mask_exact
#axiomsClean controls
#axiomsClean pointer_value
#axiomsClean active_index
#axiomsClean pointer_exact
#axiomsClean accesses
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.QuantizeKvCopy.Native
