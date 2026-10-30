import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Misc.NativeWords

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.Integer
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000

abbrev Tile (bm : Nat) : Prop := bm=64 ∨ bm=128

def ceilTiles (n bm : Nat) := n/bm + if n%bm=0 then 0 else 1

def endIndex (block bm seq prompt : Nat) := min (block*bm+bm) (seq-prompt)+prompt

def stopIndex (block bm seq prompt : Nat) :=
  if block*bm < seq-prompt then endIndex block bm seq prompt else 0

def blocks (block bm seq prompt : Nat) := ceilTiles (stopIndex block bm seq prompt) bm

def smin (a b : BitVec 64) := if a.slt b then a else b

def rowBits (block bm lane : Nat) := fromI32 block*scalar bm+fromI32 lane

def lengthBits (ws wp : MetadataWidth) (seq prompt : Nat) := metadata ws seq-metadata wp prompt

def endBits (ws wp : MetadataWidth) (block bm seq prompt : Nat) :=
  smin (fromI32 block*scalar bm+scalar bm) (lengthBits ws wp seq prompt)+metadata wp prompt

def stopBits (ws wp : MetadataWidth) (block bm seq prompt : Nat) :=
  (if (fromI32 block*scalar bm).slt (lengthBits ws wp seq prompt) then word 1 else word 0)*
    endBits ws wp block bm seq prompt

def countBits (ws wp : MetadataWidth) (block bm seq prompt : Nat) :=
  let stop := stopBits ws wp block bm seq prompt
  stop.sdiv (scalar bm) + (if stop.srem (scalar bm) = word 0 then word 0 else word 1)

def counterBits : Nat → BitVec 64
  | 0 => word 0 | n+1 => counterBits n+word 1

def indexBits (step bm : Nat) := counterBits step*scalar bm

def keyBits (step bm lane : Nat) := indexBits step bm+fromI32 lane

def causalBits (block bm lane : Nat) (wp : MetadataWidth) (prompt : Nat) :=
  rowBits block bm lane+metadata wp prompt

theorem sub_word (a b : Nat) (hb : b ≤ a) : word a-word b = word (a-b) := by
  apply BitVec.sub_eq_iff_eq_add.mpr
  simp only [word,BitVec.ofNat_add_ofNat,Nat.sub_add_cancel hb]

theorem smin_word (a b : Nat) (ha : a < 2^63) (hb : b < 2^63) :
    smin (word a) (word b) = word (min a b) := by
  unfold smin
  have h := slt_exact a b ha hb
  by_cases hlt : a < b
  · rw [if_pos (h.mpr hlt),min_eq_left (by omega)]
  · rw [if_neg (by intro ht; exact hlt (h.mp ht)),min_eq_right (by omega)]

theorem length_exact (ws wp : MetadataWidth) (seq prompt : Nat)
    (hs : ws.Fits seq) (hp : wp.Fits prompt) (hps : prompt ≤ seq) :
    lengthBits ws wp seq prompt = word (seq-prompt) := by
  rw [lengthBits,metadata_eq ws seq hs,metadata_eq wp prompt hp,sub_word seq prompt hps]

theorem row_modular (block bm lane : Nat) (hb : block < 2^31) (hm : Tile bm) (hl : lane < bm) :
    rowBits block bm lane = word (block*bm+lane) := by
  have hi : lane < 2^31 := by rcases hm with rfl|rfl <;> omega
  rw [rowBits,fromI32_eq block hb,fromI32_eq lane hi,scalar_eq]
  simp only [word,BitVec.ofNat_add_ofNat,BitVec.ofNat_mul_ofNat]

theorem row_signed (block bm lane : Nat) (hb : block < 2^31) (hm : Tile bm) (hl : lane < bm) :
    (rowBits block bm lane).toInt = (block*bm+lane : Nat) := by
  rw [row_modular block bm lane hb hm hl]
  apply signed
  rcases hm with rfl|rfl <;> omega

theorem end_le (block bm seq prompt : Nat) (hps : prompt ≤ seq) : endIndex block bm seq prompt ≤ seq := by
  unfold endIndex
  omega

theorem end_exact (ws wp : MetadataWidth) (block bm seq prompt : Nat)
    (hb : block < 2^31) (hm : Tile bm) (hs : ws.Fits seq) (hp : wp.Fits prompt) (hps : prompt ≤ seq) :
    endBits ws wp block bm seq prompt = word (endIndex block bm seq prompt) := by
  have hs63 := metadata_lt ws seq hs
  have hc : block*bm+bm < 2^63 := by rcases hm with rfl|rfl <;> omega
  rw [endBits,length_exact ws wp seq prompt hs hp hps,fromI32_eq block hb,scalar_eq,
    metadata_eq wp prompt hp]
  have he : word block*word bm+word bm = word (block*bm+bm) := by
    simp only [word,BitVec.ofNat_add_ofNat,BitVec.ofNat_mul_ofNat]
  rw [he,smin_word _ _ hc (by omega)]
  simp only [endIndex,word,BitVec.ofNat_add_ofNat]

theorem stop_le (block bm seq prompt : Nat) (hps : prompt ≤ seq) : stopIndex block bm seq prompt ≤ seq := by
  unfold stopIndex
  split
  · exact end_le block bm seq prompt hps
  · omega

theorem stop_exact (ws wp : MetadataWidth) (block bm seq prompt : Nat)
    (hb : block < 2^31) (hm : Tile bm) (hs : ws.Fits seq) (hp : wp.Fits prompt) (hps : prompt ≤ seq) :
    stopBits ws wp block bm seq prompt = word (stopIndex block bm seq prompt) := by
  have hs63 := metadata_lt ws seq hs
  have hc : block*bm < 2^63 := by rcases hm with rfl|rfl <;> omega
  rw [stopBits,length_exact ws wp seq prompt hs hp hps,end_exact ws wp block bm seq prompt hb hm hs hp hps,
    fromI32_eq block hb,scalar_eq]
  have he : word block*word bm = word (block*bm) := by simp only [word,BitVec.ofNat_mul_ofNat]
  rw [he]
  have h := slt_exact (block*bm) (seq-prompt) hc (by omega)
  unfold stopIndex
  by_cases ht : block*bm < seq-prompt
  · rw [if_pos ht,if_pos (h.mpr ht)]; simp
  · rw [if_neg ht,if_neg (by intro q; exact ht (h.mp q))]; simp

theorem ceil_small (n bm : Nat) (hn : n < 2^63) (hm : Tile bm) : ceilTiles n bm < 2^63 := by
  rcases hm with rfl|rfl <;> unfold ceilTiles <;> split <;> omega

theorem ceil_step (n bm step : Nat) (hm : Tile bm) :
    step < ceilTiles n bm ↔ step*bm < n := by
  rcases hm with rfl|rfl <;> unfold ceilTiles <;> split <;> omega

theorem count_exact (ws wp : MetadataWidth) (block bm seq prompt : Nat)
    (hb : block < 2^31) (hm : Tile bm) (hs : ws.Fits seq) (hp : wp.Fits prompt) (hps : prompt ≤ seq) :
    countBits ws wp block bm seq prompt = word (blocks block bm seq prompt) := by
  have hs63 := metadata_lt ws seq hs
  have he := stop_le block bm seq prompt hps
  have hb63 : bm < 2^63 := by rcases hm with rfl|rfl <;> omega
  rw [countBits,stop_exact ws wp block bm seq prompt hb hm hs hp hps,scalar_eq,
    sdiv_eq _ _ (by omega) hb63,srem_eq _ _ (by omega) hb63]
  have hr : stopIndex block bm seq prompt % bm < 2^64 :=
    lt_of_le_of_lt (Nat.mod_le _ _) (by omega)
  have hz : word (stopIndex block bm seq prompt % bm) = word 0 ↔ stopIndex block bm seq prompt % bm=0 := by
    constructor
    · intro h; have hh := congrArg BitVec.toNat h; simpa [unsigned _ hr] using hh
    · intro h; rw [h]
  simp only [hz,blocks,ceilTiles]
  split <;> simp only [word,BitVec.ofNat_add_ofNat]

theorem counter_exact (step : Nat) : counterBits step = word step := by
  induction step with
  | zero => rfl
  | succ step ih => rw [counterBits,ih]; simp only [word,BitVec.ofNat_add_ofNat]

theorem index_exact (step bm : Nat) : indexBits step bm = word (step*bm) := by
  rw [indexBits,counter_exact,scalar_eq]
  simp only [word,BitVec.ofNat_mul_ofNat]

theorem key_modular (step bm lane : Nat) (hm : Tile bm) (hl : lane < bm) :
    keyBits step bm lane = word (step*bm+lane) := by
  have hi : lane < 2^31 := by rcases hm with rfl|rfl <;> omega
  rw [keyBits,index_exact,fromI32_eq lane hi]
  simp only [word,BitVec.ofNat_add_ofNat]

/-- The signed loop counter and its final backedge/test are bounded; padded
key coordinates use unsigned64 and may cross the signed64 endpoint. -/
theorem loop_control (ws wp : MetadataWidth) (block bm seq prompt step : Nat)
    (hb : block < 2^31) (hm : Tile bm) (hs : ws.Fits seq) (hp : wp.Fits prompt) (hps : prompt ≤ seq)
    (ht : step ≤ blocks block bm seq prompt) :
    (counterBits step).toInt = (step : Int) ∧
    (countBits ws wp block bm seq prompt).toInt = (blocks block bm seq prompt : Int) ∧
    ((counterBits step).slt (countBits ws wp block bm seq prompt) = Bool.true ↔ step < blocks block bm seq prompt) := by
  have hs63 := metadata_lt ws seq hs
  have he := stop_le block bm seq prompt hps
  have hn := ceil_small (stopIndex block bm seq prompt) bm (by omega) hm
  change blocks block bm seq prompt < 2^63 at hn
  rw [counter_exact,count_exact ws wp block bm seq prompt hb hm hs hp hps]
  exact ⟨signed step (by omega),signed _ hn,slt_exact _ _ (by omega) hn⟩

theorem padded_key_bound (block bm seq prompt step lane : Nat)
    (hm : Tile bm) (hs : seq < 2^63) (hps : prompt ≤ seq)
    (ht : step < blocks block bm seq prompt) (hl : lane < bm) : step*bm+lane < 2^64 := by
  have he := stop_le block bm seq prompt hps
  have hi := (ceil_step (stopIndex block bm seq prompt) bm step hm).mp ht
  rcases hm with rfl|rfl <;> omega

theorem key_mask (ws wp : MetadataWidth) (block bm seq prompt step lane : Nat)
    (hb : block < 2^31) (hm : Tile bm) (hs : ws.Fits seq) (hp : wp.Fits prompt) (hps : prompt ≤ seq)
    (ht : step < blocks block bm seq prompt) (hl : lane < bm) :
    (keyBits step bm lane).ult (endBits ws wp block bm seq prompt) = Bool.true ↔
      step*bm+lane < endIndex block bm seq prompt := by
  have hs63 := metadata_lt ws seq hs
  have he := end_le block bm seq prompt hps
  rw [key_modular step bm lane hm hl,end_exact ws wp block bm seq prompt hb hm hs hp hps]
  exact ult_exact _ _ (padded_key_bound block bm seq prompt step lane hm hs63 hps ht hl) (by omega)

theorem causal_mask (wp : MetadataWidth) (block bm seq prompt step i j : Nat)
    (hb : block < 2^31) (hm : Tile bm) (hs : seq < 2^63) (hp : wp.Fits prompt) (hps : prompt ≤ seq)
    (ht : step < blocks block bm seq prompt) (hi : i < bm) (hj : j < bm) :
    (keyBits step bm j).ule (causalBits block bm i wp prompt) = Bool.true ↔
      step*bm+j ≤ block*bm+i+prompt := by
  rw [key_modular step bm j hm hj,causalBits,row_modular block bm i hb hm hi,metadata_eq wp prompt hp]
  have he : word (block*bm+i)+word prompt = word (block*bm+i+prompt) := by
    simp only [word,BitVec.ofNat_add_ofNat]
  rw [he]
  apply ule_exact
  · exact padded_key_bound block bm seq prompt step j hm hs hps ht hj
  · rcases hm with rfl|rfl <;> omega

end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.Integer
