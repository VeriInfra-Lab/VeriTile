import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Misc.Safety

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.NativeSigned
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnLlama Host Program
open Integer
noncomputable section
set_option maxHeartbeats 4000000

/-- The repaired source uses signed64 comparisons and the original
min(block_start+BM+prompt, sequence) association. -/
def ending (c : Call) (idx : GridIndex c.grid) : BitVec 64 :=
  smin (fromI32 (c.launchAt idx).block*scalar c.layout.BM+scalar c.layout.BM+
    metadata .int32 (c.prompt (batchAt c idx))) (metadata .int32 (seqAt c idx))
def keyMask (c : Call) (idx : GridIndex c.grid) (step j : Nat) : Bool :=
  (keyBits step c.layout.BN j).slt (ending c idx)
def causalMask (c : Call) (idx : GridIndex c.grid) (step i j : Nat) : Bool :=
  (keyBits step c.layout.BN j).sle (causalBits (c.launchAt idx).block c.layout.BM i .int32 (c.prompt (batchAt c idx)))
def pageWord (c : Call) (idx : GridIndex c.grid) (step j : Nat) : BitVec 64 :=
  if keyMask c idx step j then metadata .int32 (c.page (batchAt c idx) (step*c.layout.BN+j)) else word 0
def payloadOffset (c : Call) (role : DataRole) (idx : GridIndex c.grid) (step j dim : Nat) : BitVec 64 :=
  valueBits c.layout (c.launchAt idx) (pageWord c idx step j) dim (role.row c) (role.head c) (role.dim c)

theorem ending_exact (r : Request) (heap : ByteMemory) (h : Contract r heap) (idx : GridIndex r.call.grid) :
    ending r.call idx=word (endAt r.call idx) := by
  have finite := Safety.machine r heap h
  have hb := block_abi r.call Safety.widths finite idx
  have hs := finite.sequence _ (Host.batch_bound r.call idx)
  have hp := finite.prompt _ (Host.batch_bound r.call idx)
  change seqAt r.call idx<2^31 at hs
  change r.call.prompt (batchAt r.call idx)<2^31 at hp
  have bound : (r.call.launchAt idx).block*r.call.layout.BM+r.call.layout.BM+
      r.call.prompt (batchAt r.call idx)<2^63 := by
    rcases r.call.blockM with bm|bm <;> rw [bm] <;> omega
  rw [ending,fromI32_eq _ hb,scalar_eq,metadata_eq .int32 _ hp,metadata_eq .int32 _ hs]
  simp only [word,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
  rw [smin_word _ _ bound (by omega)]
  congr 1
  unfold endAt endIndex seqAt
  omega

theorem key_bound (r : Request) (heap : ByteMemory) (h : Contract r heap) (idx : GridIndex r.call.grid)
    (step : Nat) (j : Fin r.call.layout.BN) (ht : step<countAt r.call idx) :
    step*r.call.layout.BN+j.val<2^63 := by
  have finite := Safety.machine r heap h
  have hs := finite.sequence _ (Host.batch_bound r.call idx)
  change seqAt r.call idx<2^31 at hs
  have stop := stop_le (r.call.launchAt idx).block r.call.layout.BM (seqAt r.call idx)
    (r.call.prompt (batchAt r.call idx)) (by dsimp only [seqAt]; omega)
  have reach := (ceil_step _ r.call.layout.BM step r.call.blockM).mp ht
  have lane := j.isLt
  simp only [r.call.blockN] at lane ⊢
  rcases r.call.blockM with bm|bm <;> simp only [bm] at reach lane stop ⊢ <;> omega

theorem key_mask (r : Request) (heap : ByteMemory) (h : Contract r heap) (idx : GridIndex r.call.grid)
    (step : Nat) (j : Fin r.call.layout.BN) (ht : step<countAt r.call idx) :
    keyMask r.call idx step j=Bool.true ↔ step*r.call.layout.BN+j.val<endAt r.call idx := by
  have finite := Safety.machine r heap h
  have hs := finite.sequence _ (Host.batch_bound r.call idx)
  change seqAt r.call idx<2^31 at hs
  have endingBound := end_le (r.call.launchAt idx).block r.call.layout.BM (seqAt r.call idx)
    (r.call.prompt (batchAt r.call idx)) (by dsimp only [seqAt]; omega)
  change endAt r.call idx≤seqAt r.call idx at endingBound
  rw [keyMask,key_modular _ _ _ (by rw [r.call.blockN]; exact r.call.blockM) j.isLt,ending_exact r heap h idx]
  exact slt_exact _ _ (key_bound r heap h idx step j ht) (by omega)

theorem causal_mask (r : Request) (heap : ByteMemory) (h : Contract r heap) (idx : GridIndex r.call.grid)
    (step : Nat) (i : Fin r.call.layout.BM) (j : Fin r.call.layout.BN) (ht : step<countAt r.call idx) :
    causalMask r.call idx step i j=Bool.true ↔
      step*r.call.layout.BN+j.val≤Formula.queryRow r.call.layout (r.call.launchAt idx) i+r.call.prompt (batchAt r.call idx) := by
  have finite := Safety.machine r heap h
  have block := block_abi r.call Safety.widths finite idx
  have prompt := finite.prompt _ (Host.batch_bound r.call idx)
  change r.call.prompt (batchAt r.call idx)<2^31 at prompt
  have causalBound : (r.call.launchAt idx).block*r.call.layout.BM+i.val+r.call.prompt (batchAt r.call idx)<2^63 := by
    have lane := i.isLt
    rcases r.call.blockM with bm|bm <;> simp only [bm] at lane ⊢ <;> omega
  rw [causalMask,key_modular _ _ _ (by rw [r.call.blockN]; exact r.call.blockM) j.isLt,
    causalBits,row_modular _ _ _ block r.call.blockM i.isLt,metadata_eq .int32 _ prompt]
  simp only [word,BitVec.ofNat_add_ofNat]
  rw [BitVec.sle_iff_toInt_le,signed _ (key_bound r heap h idx step j ht),signed _ causalBound]
  exact Int.ofNat_le

theorem page_word (r : Request) (heap : ByteMemory) (h : Contract r heap) (idx : GridIndex r.call.grid)
    (step : Nat) (j : Fin r.call.layout.BN) (ht : step<countAt r.call idx) :
    pageWord r.call idx step j=pageBits r.call Safety.widths idx step j := by
  have native := key_mask r heap h idx step j ht
  have generic := keyMask_exact r.call Safety.widths (Safety.machine r heap h) idx step j ht
  have eq : keyMask r.call idx step j=keyMaskBits r.call Safety.widths idx step j := by
    cases h1 : keyMask r.call idx step j <;> cases h2 : keyMaskBits r.call Safety.widths idx step j <;> simp_all <;> omega
  simp only [pageWord,pageBits,eq,Safety.widths]

theorem payload_offset (r : Request) (heap : ByteMemory) (h : Contract r heap) (idx : GridIndex r.call.grid)
    (step : Nat) (j : Fin r.call.layout.BN) (dim : Fin r.call.layout.D) (ht : step<countAt r.call idx)
    (role : DataRole) (hr : role=.key ∨ role=.value) :
    (payloadOffset r.call role idx step j dim).toInt=(kvOffset r.call role idx step j dim : Int) := by
  rw [payloadOffset,page_word r heap h idx step j ht]
  exact ((Safety.all_formed r heap h idx step ⟨0,Host.bm_positive r.call⟩ j dim ht).2 role hr).1

#axiomsClean ending_exact
#axiomsClean key_mask
#axiomsClean causal_mask
#axiomsClean payload_offset
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.NativeSigned
