import bench.tritonbench_g.layer_norm_welfold.LayerNormWelfold
import bench.tritonbench_g.fused_recurrent_delta.Proofs.FPStages
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.LayerNormWelfold.FPStages
open VeriTile.Triton FP.Structural
open FusedRecurrentHgrn.FPStages (scatter scatter_memory_congr)
open FusedRecurrentDelta.FPStages (Post post_of_memory)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def normalizeChunk
    (in_out_ptr0 in_out_ptr1 in_ptr0 in_ptr1 in_ptr2 out_ptr0 : RegionName)
    (rnumel RBLOCK : Nat) : ComputeKernel := triton {
  tl.region in_out_ptr0 = tl.float32, in_out_ptr1 = tl.float32, in_ptr0 = tl.float32, in_ptr1 = tl.float32, in_ptr2 = tl.float32, out_ptr0 = tl.float32
  x0 = tl.program_id(0)
  i_t = tl.program_id(1)
  rindex = i_t * $(RBLOCK) + tl.arange(0, $(RBLOCK))
  rmask = rindex < $(rnumel)
  r1 = rindex
  tmp6 = tl.load(in_out_ptr0 + (x0))
  tmp18 = tl.load(in_out_ptr1 + (x0))
  tmp19 = tl.load(in_ptr0 + (r1 + ($(rnumel) * x0)), mask=rmask, other=0.0).to(tl.float32)
  tmp23 = tl.load(in_ptr1 + (r1), mask=rmask, other=0.0).to(tl.float32)
  tmp26 = tl.load(in_ptr2 + (r1), mask=rmask, other=0.0).to(tl.float32)
  tmp20 = (tmp19).to(tl.float32)
  tmp21 = tmp20 - tmp6
  tmp22 = tmp21 * tmp18
  tmp24 = (tmp23).to(tl.float32)
  tmp25 = tmp22 * tmp24
  tmp27 = (tmp26).to(tl.float32)
  tmp28 = tmp25 + tmp27
  tmp29 = (tmp28).to(tl.float32)
  tl.store(out_ptr0 + (r1 + ($(rnumel) * x0)), tmp29, mask=rmask)
}

def column {α : Type} (s : State α) (RBLOCK : Nat) (i : Fin RBLOCK) : Nat :=
  s.pids 1 * RBLOCK + i.val

def rowAddress {α : Type} (s : State α) (rnumel RBLOCK : Nat) (i : TileIndex [RBLOCK]) : Nat :=
  column s RBLOCK i.1 + rnumel * s.pids 0

def convertedLoad {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (R : RegionName) (offset : Nat) (active : Prop) [Decidable active] : α :=
  M.cast (some .fp32) .real .real
    (if active then (s.mem R offset).read .real else M.literal (some .fp32) .real 0)

/-- Exact source order, including both input conversions and the final conversion. -/
def normalizeValues {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (mean rstd X W Bias : RegionName) (rnumel RBLOCK : Nat) : TileIndex [RBLOCK] → α := fun i =>
  let active := column s RBLOCK i.1 < rnumel
  let x := M.cast (some .fp32) .real .real
    (convertedLoad M s X (rowAddress s rnumel RBLOCK i) active)
  let weight := M.cast (some .fp32) .real .real
    (convertedLoad M s W (column s RBLOCK i.1) active)
  let bias := M.cast (some .fp32) .real .real
    (convertedLoad M s Bias (column s RBLOCK i.1) active)
  let average := M.fp32Load ((s.mem mean (s.pids 0)).read .real)
  let inverse := M.fp32Load ((s.mem rstd (s.pids 0)).read .real)
  M.cast (some .fp32) .real .real (M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .mul
      (M.binary (some .fp32) .real .mul
        (M.binary (some .fp32) .real .sub x average) inverse) weight) bias)

theorem normalize_memory {α : Type} [Inhabited α] (M : Algebra α)
    (mean rstd X W Bias Y : RegionName) (rnumel RBLOCK : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (normalizeChunk mean rstd X W Bias Y rnumel RBLOCK) s = some final ∧
      final.mem = (scatter s Y RBLOCK (rowAddress s rnumel RBLOCK)
        (fun i => column s RBLOCK i.1 < rnumel)
        (normalizeValues M s mean rstd X W Bias rnumel RBLOCK)).mem := by
  simp [normalizeChunk, FP.Structural.exec, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, store, Option.bind, ComputeDType.eraseDType,
    numeric, bop, natLt, numericLt, Region.cast, normalizeValues, convertedLoad,
    column, rowAddress, scatter, show (0.0 : ℝ) = 0 by norm_num]
  apply scatter_memory_congr
  rfl

specification normalize_chunk_fp32 {α : Type} [Inhabited α] (M : Algebra α)
    (mean rstd X W Bias Y : RegionName) (rnumel RBLOCK : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (normalizeChunk mean rstd X W Bias Y rnumel RBLOCK) s = some final ∧
      Post s final Y RBLOCK (rowAddress s rnumel RBLOCK)
        (fun i => column s RBLOCK i.1 < rnumel)
        (normalizeValues M s mean rstd X W Bias rnumel RBLOCK) := by
  obtain ⟨final, he, hm⟩ := normalize_memory M mean rstd X W Bias Y rnumel RBLOCK s
  refine ⟨final, he, post_of_memory s final Y RBLOCK _ _ _ ?_ hm⟩
  intro i j hij
  simp only [rowAddress, column] at hij
  have h : i.1.val = j.1.val := by omega
  exact Prod.ext (Fin.ext h) (Subsingleton.elim _ _)

#axiomsClean normalize_chunk_fp32
end
end VeriTile.Bench.TritonBenchG.LayerNormWelfold.FPStages
