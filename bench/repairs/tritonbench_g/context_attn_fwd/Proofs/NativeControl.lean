import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.NativeWords

namespace VeriTile.Bench.Repairs.ContextAttnFwd.Native
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000

abbrev Tile (bm : Nat) : Prop := bm=64 ∨ bm=128

def ceilTiles (n bm : Nat) := n/bm + if n%bm=0 then 0 else 1
def endIndex (block bm seq prompt : Nat) := min (block*bm+bm+prompt) seq
def stopIndex (block bm seq prompt : Nat) :=
  if block*bm < seq-prompt then endIndex block bm seq prompt else 0
def blocks (block bm seq prompt : Nat) := ceilTiles (stopIndex block bm seq prompt) bm

def smin (a b : BitVec 64) := if a.slt b then a else b
def rowBits (block bm lane : Nat) := fromI32 block*scalar bm+fromI32 lane
def lengthBits (seq prompt : Nat) := fromI32 seq-fromI32 prompt
/-- Actual source operation order: block start, tile width, prompt, then min. -/
def endBits (block bm seq prompt : Nat) :=
  smin (fromI32 block*scalar bm+scalar bm+fromI32 prompt)
    (lengthBits seq prompt+fromI32 prompt)
def stopBits (block bm seq prompt : Nat) :=
  (if (fromI32 block*scalar bm).slt (lengthBits seq prompt) then word 1 else word 0)*
    endBits block bm seq prompt
/-- PTX carries the actual start_n, increasing it by BLOCK_N on each backedge. -/
def indexBits (step bm : Nat) := word step*scalar bm
def keyBits (step bm lane : Nat) := indexBits step bm+fromI32 lane
def causalBits (block bm lane prompt : Nat) := rowBits block bm lane+fromI32 prompt

theorem sub_word (a b : Nat) (hb : b≤a) : word a-word b=word (a-b) := by
  apply BitVec.sub_eq_iff_eq_add.mpr
  simp only [word,BitVec.ofNat_add_ofNat,Nat.sub_add_cancel hb]

theorem smin_word (a b : Nat) (ha : a<2^63) (hb : b<2^63) :
    smin (word a) (word b)=word (min a b) := by
  unfold smin
  have h := slt_exact a b ha hb
  by_cases ht : a<b
  · rw [if_pos (h.mpr ht),min_eq_left (by omega)]
  · rw [if_neg (by intro q; exact ht (h.mp q)),min_eq_right (by omega)]

theorem length_exact (seq prompt : Nat) (hs : seq<2^31) (hp : prompt≤seq) :
    lengthBits seq prompt=word (seq-prompt) := by
  rw [lengthBits,fromI32_eq seq hs,fromI32_eq prompt (by omega),sub_word seq prompt hp]

theorem row_modular (block bm lane : Nat) (hb : block<2^31) (hm : Tile bm) (hl : lane<bm) :
    rowBits block bm lane=word (block*bm+lane) := by
  have hi : lane<2^31 := by rcases hm with rfl|rfl <;> omega
  rw [rowBits,fromI32_eq block hb,fromI32_eq lane hi,scalar_eq]
  simp only [word,BitVec.ofNat_add_ofNat,BitVec.ofNat_mul_ofNat]

theorem end_le (block bm seq prompt : Nat) : endIndex block bm seq prompt≤seq := min_le_right _ _

theorem end_exact (block bm seq prompt : Nat) (hb : block<2^31) (hm : Tile bm)
    (hs : seq<2^31) (hp : prompt≤seq) : endBits block bm seq prompt=word (endIndex block bm seq prompt) := by
  have hleft : block*bm+bm+prompt<2^63 := by rcases hm with rfl|rfl <;> omega
  rw [endBits,length_exact seq prompt hs hp,fromI32_eq block hb,
    fromI32_eq prompt (by omega),scalar_eq]
  have hl : word block*word bm+word bm+word prompt=word (block*bm+bm+prompt) := by
    simp only [word,BitVec.ofNat_add_ofNat,BitVec.ofNat_mul_ofNat]
  have hr : word (seq-prompt)+word prompt=word seq := by
    simp only [word,BitVec.ofNat_add_ofNat,Nat.sub_add_cancel hp]
  rw [hl,hr,smin_word _ _ hleft (by omega)]
  rfl

theorem stop_le (block bm seq prompt : Nat) : stopIndex block bm seq prompt≤seq := by
  unfold stopIndex
  split
  · exact end_le _ _ _ _
  · omega

theorem stop_exact (block bm seq prompt : Nat) (hb : block<2^31) (hm : Tile bm)
    (hs : seq<2^31) (hp : prompt≤seq) : stopBits block bm seq prompt=word (stopIndex block bm seq prompt) := by
  have hc : block*bm<2^63 := by rcases hm with rfl|rfl <;> omega
  rw [stopBits,length_exact seq prompt hs hp,end_exact block bm seq prompt hb hm hs hp,
    fromI32_eq block hb,scalar_eq]
  have he : word block*word bm=word (block*bm) := by simp only [word,BitVec.ofNat_mul_ofNat]
  rw [he]
  have h := slt_exact (block*bm) (seq-prompt) hc (by omega)
  unfold stopIndex
  by_cases ht : block*bm<seq-prompt
  · rw [if_pos ht,if_pos (h.mpr ht)]; simp
  · rw [if_neg ht,if_neg (by intro q; exact ht (h.mp q))]; simp

theorem ceil_step (n bm step : Nat) (hm : Tile bm) : step<ceilTiles n bm ↔ step*bm<n := by
  rcases hm with rfl|rfl <;> unfold ceilTiles <;> split <;> omega

theorem final_index_bound (stop bm step : Nat) (hm : Tile bm) (hs : stop<2^31)
    (ht : step≤ceilTiles stop bm) : step*bm<2^63 := by
  rcases hm with rfl|rfl <;> unfold ceilTiles at ht <;> split at ht <;> omega

theorem index_exact (step bm : Nat) : indexBits step bm=word (step*bm) := by
  simp only [indexBits,scalar_eq,word,BitVec.ofNat_mul_ofNat]

theorem key_modular (step bm lane : Nat) (hm : Tile bm) (hl : lane<bm) :
    keyBits step bm lane=word (step*bm+lane) := by
  have hi : lane<2^31 := by rcases hm with rfl|rfl <;> omega
  rw [keyBits,index_exact,fromI32_eq lane hi]
  simp only [word,BitVec.ofNat_add_ofNat]

theorem query_mask (block bm seq prompt lane : Nat) (hb : block<2^31) (hm : Tile bm)
    (hs : seq<2^31) (hp : prompt≤seq) (hl : lane<bm) :
    (rowBits block bm lane).slt (lengthBits seq prompt)= Bool.true ↔ block*bm+lane<seq-prompt := by
  rw [row_modular block bm lane hb hm hl,length_exact seq prompt hs hp]
  apply slt_exact
  · rcases hm with rfl|rfl <;> omega
  · omega

theorem padded_key_bound (block bm seq prompt step lane : Nat) (hm : Tile bm)
    (ht : step<blocks block bm seq prompt) (hl : lane<bm) : step*bm+lane<seq+bm := by
  have he := stop_le block bm seq prompt
  have hi := (ceil_step (stopIndex block bm seq prompt) bm step hm).mp ht
  omega

theorem key_mask (block bm seq prompt step lane : Nat) (hb : block<2^31) (hm : Tile bm)
    (hs : seq<2^31) (hp : prompt≤seq) (ht : step<blocks block bm seq prompt) (hl : lane<bm) :
    (keyBits step bm lane).slt (endBits block bm seq prompt)= Bool.true ↔ step*bm+lane<endIndex block bm seq prompt := by
  have hpad := padded_key_bound block bm seq prompt step lane hm ht hl
  have he := end_le block bm seq prompt
  rw [key_modular step bm lane hm hl,end_exact block bm seq prompt hb hm hs hp]
  apply slt_exact
  · rcases hm with rfl|rfl <;> omega
  · omega

theorem causal_mask (block bm seq prompt step i j : Nat) (hb : block<2^31) (hm : Tile bm)
    (hs : seq<2^31) (hp : prompt≤seq) (ht : step<blocks block bm seq prompt) (hi : i<bm) (hj : j<bm) :
    ((causalBits block bm i prompt).slt (keyBits step bm j)= Bool.false ↔ step*bm+j≤block*bm+i+prompt) := by
  have hpad := padded_key_bound block bm seq prompt step j hm ht hj
  rw [key_modular step bm j hm hj,causalBits,row_modular block bm i hb hm hi,fromI32_eq prompt (by omega)]
  have he : word (block*bm+i)+word prompt=word (block*bm+i+prompt) := by simp only [word,BitVec.ofNat_add_ofNat]
  rw [he]
  have hc : block*bm+i+prompt<2^63 := by rcases hm with rfl|rfl <;> omega
  have hk : step*bm+j<2^63 := by rcases hm with rfl|rfl <;> omega
  have h := slt_exact (block*bm+i+prompt) (step*bm+j) hc hk
  cases ht' : (word (block*bm+i+prompt)).slt (word (step*bm+j)) <;> simp_all <;> omega

end
end VeriTile.Bench.Repairs.ContextAttnFwd.Native
