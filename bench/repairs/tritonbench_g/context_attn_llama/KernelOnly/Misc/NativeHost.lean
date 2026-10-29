import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Misc.NativeAddresses
import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.FPGrid

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.Integer
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ContextAttnLlama
noncomputable section
set_option maxHeartbeats 4000000

abbrev batchAt (c : Host.Call) (idx : GridIndex c.grid) := Formula.batch c.layout (c.launchAt idx)
abbrev seqAt (c : Host.Call) (idx : GridIndex c.grid) := c.length (batchAt c idx)+c.prompt (batchAt c idx)
abbrev endAt (c : Host.Call) (idx : GridIndex c.grid) := endIndex (c.launchAt idx).block c.layout.BM (seqAt c idx) (c.prompt (batchAt c idx))
abbrev countAt (c : Host.Call) (idx : GridIndex c.grid) := blocks (c.launchAt idx).block c.layout.BM (seqAt c idx) (c.prompt (batchAt c idx))

def queryMaskBits (c : Host.Call) (w : Widths) (idx : GridIndex c.grid) (i : Nat) :=
  (rowBits (c.launchAt idx).block c.layout.BM i).slt
    (lengthBits w.sequence w.prompt (seqAt c idx) (c.prompt (batchAt c idx)))
def keyMaskBits (c : Host.Call) (w : Widths) (idx : GridIndex c.grid) (step j : Nat) :=
  (keyBits step c.layout.BN j).ult
    (endBits w.sequence w.prompt (c.launchAt idx).block c.layout.BM (seqAt c idx) (c.prompt (batchAt c idx)))
def pageBits (c : Host.Call) (w : Widths) (idx : GridIndex c.grid) (step j : Nat) :=
  if keyMaskBits c w idx step j then metadata w.pages (c.page (batchAt c idx) (step*c.layout.BN+j)) else word 0

def mathPage (c : Host.Call) (idx : GridIndex c.grid) (step j : Nat) :=
  if step*c.layout.BN+j < endAt c idx then c.page (batchAt c idx) (step*c.layout.BN+j) else 0

theorem source_limit (c : Host.Call) (A : FlatAlloc) (s : BlockState) (v : Host.Valid c A s)
    (idx : GridIndex c.grid) :
    Formula.limit c.layout (c.launchAt idx) (Program.inputs s c.regions) = (endAt c idx : Int) := by
  have hb := Host.batch_bound c idx
  unfold Formula.limit
  rw [v.prompt_value _ hb,v.length_value _ hb]
  simp only [endAt,endIndex,seqAt,batchAt,Nat.add_sub_cancel_right,Nat.cast_add,Nat.cast_mul,Nat.cast_min]
  rw [add_mul,one_mul]
  omega

theorem source_count (c : Host.Call) (A : FlatAlloc) (s : BlockState) (v : Host.Valid c A s)
    (idx : GridIndex c.grid) :
    Formula.count c.layout (c.launchAt idx) (Program.inputs s c.regions) = countAt c idx := by
  have hs := Host.sequence_eq c A s v idx
  change Formula.sequence c.layout (c.launchAt idx) (Program.inputs s c.regions) = _ at hs
  unfold Formula.count
  rw [hs,source_limit c A s v idx,c.blockN,Int.toNat_natCast]
  have hc : ((c.launchAt idx).block*c.layout.BM : Int) < (c.length (batchAt c idx) : Nat) ↔
      (c.launchAt idx).block*c.layout.BM < c.length (batchAt c idx) := by omega
  simp only [hc,countAt,blocks,stopIndex,seqAt,Nat.add_sub_cancel_right]
  split <;> simp_all [ceilTiles]

theorem queryMask_exact (c : Host.Call) (w : Widths)
    (abi : Machine c w) (idx : GridIndex c.grid) (i : Fin c.layout.BM) :
    queryMaskBits c w idx i = Bool.true ↔ Formula.queryRow c.layout (c.launchAt idx) i < c.length (batchAt c idx) := by
  have h := query_mask w.sequence w.prompt (c.launchAt idx).block c.layout.BM (seqAt c idx)
    (c.prompt (batchAt c idx)) i (block_abi c w abi idx) c.blockM
    (abi.sequence _ (Host.batch_bound c idx)) (abi.prompt _ (Host.batch_bound c idx)) (by dsimp only [seqAt]; omega) i.isLt
  simpa only [queryMaskBits,seqAt,Nat.add_sub_cancel_right,Formula.queryRow] using h

theorem keyMask_exact (c : Host.Call) (w : Widths)
    (abi : Machine c w) (idx : GridIndex c.grid) (step : Nat) (j : Fin c.layout.BN)
    (ht : step < countAt c idx) :
    keyMaskBits c w idx step j = Bool.true ↔ step*c.layout.BN+j < endAt c idx := by
  have hj : j.val < c.layout.BM := by simpa only [c.blockN] using j.isLt
  have h := key_mask w.sequence w.prompt (c.launchAt idx).block c.layout.BM (seqAt c idx)
    (c.prompt (batchAt c idx)) step j (block_abi c w abi idx) c.blockM
    (abi.sequence _ (Host.batch_bound c idx)) (abi.prompt _ (Host.batch_bound c idx)) (by dsimp only [seqAt]; omega) ht hj
  simpa only [keyMaskBits,c.blockN] using h

theorem pageBits_exact (c : Host.Call) (w : Widths)
    (abi : Machine c w) (idx : GridIndex c.grid) (step : Nat) (j : Fin c.layout.BN)
    (ht : step < countAt c idx) : pageBits c w idx step j = word (mathPage c idx step j) := by
  have hm := keyMask_exact c w abi idx step j ht
  unfold pageBits mathPage
  by_cases hj : step*c.layout.BN+j.val < endAt c idx
  · rw [if_pos hj,if_pos (hm.mpr hj)]
    apply metadata_eq
    apply abi.pages _ (Host.batch_bound c idx)
    have he := end_le (c.launchAt idx).block c.layout.BM (seqAt c idx) (c.prompt (batchAt c idx)) (by dsimp only [seqAt]; omega)
    exact lt_of_lt_of_le hj he
  · rw [if_neg hj,if_neg (by intro h; exact hj (hm.mp h))]

theorem source_page (c : Host.Call) (A : FlatAlloc) (s : BlockState) (v : Host.Valid c A s)
    (idx : GridIndex c.grid) (step j : Nat) :
    Formula.page c.layout (c.launchAt idx) (Program.inputs s c.regions) (step*c.layout.BN+j) =
      (mathPage c idx step j : Int) := by
  by_cases hj : step*c.layout.BN+j < endAt c idx
  · have hl : ((step*c.layout.BN+j : Nat) : Int) <
        Formula.limit c.layout (c.launchAt idx) (Program.inputs s c.regions) := by
      rw [source_limit c A s v idx]; exact_mod_cast hj
    have h := Host.page_eq c A s v idx (step*c.layout.BN+j) hl
    simpa only [Host.launch_grid,Host.inputs_grid,mathPage,if_pos hj] using h
  · rw [Formula.page,source_limit c A s v idx]
    have hn : ¬ ((step*c.layout.BN+j : Nat) : Int) < (endAt c idx : Int) := by exact_mod_cast hj
    simp only [if_neg hn,mathPage,if_neg hj,Nat.cast_zero]

/-- Both the source and native loop reach exactly these steps. The final test
uses the tile counter, so its increment remains signed64-safe at the endpoint. -/
theorem source_loop (c : Host.Call) (A : FlatAlloc) (s : BlockState) (v : Host.Valid c A s)
    (w : Widths) (abi : Machine c w) (idx : GridIndex c.grid)
    (step : Nat) (ht : step ≤ Formula.count c.layout (c.launchAt idx) (Program.inputs s c.regions)) :
    (counterBits step).toInt = (step : Int) ∧
    (countBits w.sequence w.prompt (c.launchAt idx).block c.layout.BM (seqAt c idx)
      (c.prompt (batchAt c idx))).toInt = (Formula.count c.layout (c.launchAt idx) (Program.inputs s c.regions) : Int) ∧
    ((counterBits step).slt (countBits w.sequence w.prompt (c.launchAt idx).block c.layout.BM (seqAt c idx)
      (c.prompt (batchAt c idx))) = Bool.true ↔ step < Formula.count c.layout (c.launchAt idx) (Program.inputs s c.regions)) := by
  rw [source_count c A s v idx] at ht ⊢
  exact loop_control _ _ _ _ _ _ _ (block_abi c w abi idx) c.blockM
    (abi.sequence _ (Host.batch_bound c idx)) (abi.prompt _ (Host.batch_bound c idx)) (by dsimp only [seqAt]; omega) ht

/-- Metadata in the typed FP state must also bind request/page gathers. The
older success theorem needed only start/length/prompt to prove its store frame. -/
def FullMetadata {α : Type} [Inhabited α] (c : Host.Call) (s : FP.Structural.State α) : Prop :=
  FPSuccess.Metadata c s ∧
  (∀ b, b < c.batches → (s.mem c.regions.request b).read .int = (c.request b : Int)) ∧
  (∀ b, b < c.batches → ∀ j, j < c.length b+c.prompt b →
    (s.mem c.regions.pages (c.layout.reqRow*c.request b+c.layout.reqToken*j)).read .int = (c.page b j : Int))

end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.Integer
