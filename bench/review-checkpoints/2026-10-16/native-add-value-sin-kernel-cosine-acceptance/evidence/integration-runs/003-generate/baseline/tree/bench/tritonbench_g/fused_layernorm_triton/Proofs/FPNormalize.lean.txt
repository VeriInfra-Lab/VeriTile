import bench.tritonbench_g.fused_layernorm_triton.FusedLayernormTriton
import bench.tritonbench_g.layer_norm_welfold.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.FusedLayernormTriton.FPStages
open VeriTile.Triton FP.Structural
open FusedRecurrentHgrn.FPStages (scatter scatter_memory_congr)
open FusedRecurrentDelta.FPStages (Post post_of_memory word)
open LayerNormWelfold.FPStages (column rowAddress normalizeValues roundedNormalize)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

/-- Actual normalization operations at the accepted materialized mean/rstd
boundary. Source masked loads have no `other`; their undefined tails remain. -/
def normalizeSourceChunk
    (in_out_ptr0 in_ptr0 in_ptr1 in_ptr2 out_ptr0 out_ptr1 : RegionName)
    (rnumel RBLOCK : Nat) : ComputeKernel := triton {
  tl.region in_out_ptr0 = tl.float32, in_ptr0 = tl.float32, in_ptr1 = tl.float32, in_ptr2 = tl.float32, out_ptr0 = tl.float32, out_ptr1 = tl.float32
  x0 = tl.program_id(0)
  i_t = tl.program_id(1)
  rindex = i_t * $(RBLOCK) + tl.arange(0, $(RBLOCK))
  rmask = rindex < $(rnumel)
  r1 = rindex
  tmp3 = tl.load(out_ptr0 + (x0))
  tmp10 = tl.load(in_out_ptr0 + (x0))
  tmp11 = tl.load(in_ptr0 + (r1 + ($(rnumel) * x0)), mask=rmask).to(tl.float32)
  tmp15 = tl.load(in_ptr1 + (r1), mask=rmask).to(tl.float32)
  tmp18 = tl.load(in_ptr2 + (r1), mask=rmask).to(tl.float32)
  tmp12 = (tmp11).to(tl.float32)
  tmp13 = tmp12 - tmp3
  tmp14 = tmp13 * tmp10
  tmp16 = (tmp15).to(tl.float32)
  tmp17 = tmp14 * tmp16
  tmp19 = (tmp18).to(tl.float32)
  tmp20 = tmp17 + tmp19
  tmp21 = (tmp20).to(tl.float32)
  tl.store(out_ptr1 + (r1 + ($(rnumel) * x0)), tmp21, mask=rmask)
}

def sourceInput {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (R : RegionName) (offset : Nat) (active : Prop) [Decidable active]
    {RBLOCK : Nat} (i : TileIndex [RBLOCK]) : α :=
  M.cast (some .fp32) .real .real
    (if active then (s.mem R offset).read .real else s.undef .real R offset)

def sourceValues {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (mean rstd X W Bias : RegionName) (rnumel RBLOCK : Nat) : TileIndex [RBLOCK] → α := fun i =>
  let active := column s RBLOCK i.1 < rnumel
  let x := M.cast (some .fp32) .real .real
    (sourceInput M s X (rowAddress s rnumel RBLOCK i) active i)
  let weight := M.cast (some .fp32) .real .real
    (sourceInput M s W (column s RBLOCK i.1) active i)
  let bias := M.cast (some .fp32) .real .real
    (sourceInput M s Bias (column s RBLOCK i.1) active i)
  let average := M.fp32Load ((s.mem mean (s.pids 0)).read .real)
  let inverse := M.fp32Load ((s.mem rstd (s.pids 0)).read .real)
  M.cast (some .fp32) .real .real (M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .mul
      (M.binary (some .fp32) .real .mul
        (M.binary (some .fp32) .real .sub x average) inverse) weight) bias)

theorem normalize_memory {α : Type} [Inhabited α] (M : Algebra α)
    (mean rstd X W Bias Y : RegionName) (rnumel RBLOCK : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (normalizeSourceChunk rstd X W Bias mean Y rnumel RBLOCK) s = some final ∧
      final.mem = (scatter s Y RBLOCK (rowAddress s rnumel RBLOCK)
        (fun i => column s RBLOCK i.1 < rnumel)
        (sourceValues M s mean rstd X W Bias rnumel RBLOCK)).mem := by
  simp [normalizeSourceChunk, FP.Structural.exec, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, store, Option.bind, ComputeDType.eraseDType,
    numeric, bop, natLt, numericLt, Region.cast, sourceValues, sourceInput,
    column, rowAddress, scatter]
  apply scatter_memory_congr
  rfl

/-- Undefined masked input lanes cannot affect an active output of this
pointwise stage. This is a structural mask fact, not a numerical law. -/
specification normalize_source_fp32 {α : Type} [Inhabited α] (M : Algebra α)
    (mean rstd X W Bias Y : RegionName) (rnumel RBLOCK : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (normalizeSourceChunk rstd X W Bias mean Y rnumel RBLOCK) s = some final ∧
      Post s final Y RBLOCK (rowAddress s rnumel RBLOCK)
        (fun i => column s RBLOCK i.1 < rnumel)
        (normalizeValues M s mean rstd X W Bias rnumel RBLOCK) := by
  obtain ⟨final, he, hm⟩ := normalize_memory M mean rstd X W Bias Y rnumel RBLOCK s
  have hinj : Function.Injective (rowAddress s rnumel RBLOCK) := by
    intro i j hij
    simp only [rowAddress, column] at hij
    have h : i.1.val = j.1.val := by omega
    exact Prod.ext (Fin.ext h) (Subsingleton.elim _ _)
  have hp := post_of_memory s final Y RBLOCK _ _ _ hinj hm
  refine ⟨final, he, ?_, hp.2⟩
  intro i hi
  simpa [sourceValues, sourceInput, normalizeValues, LayerNormWelfold.FPStages.convertedLoad, hi] using hp.1 i hi

specification normalize_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (mean rstd X W Bias Y : RegionName) (rnumel RBLOCK : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg)
      (normalizeSourceChunk rstd X W Bias mean Y rnumel RBLOCK) s = some final ∧
      Post s final Y RBLOCK (rowAddress s rnumel RBLOCK)
        (fun i => column s RBLOCK i.1 < rnumel)
        (roundedNormalize cfg s mean rstd X W Bias rnumel RBLOCK) := by
  obtain ⟨final, he, hp⟩ := normalize_source_fp32 (FP.Software.scalarFP32 rest cfg)
    mean rstd X W Bias Y rnumel RBLOCK s
  refine ⟨final, he, ?_, hp.2⟩
  intro i hi
  simpa [normalizeValues, LayerNormWelfold.FPStages.convertedLoad, roundedNormalize, word,
    FP.Software.scalarFP32, FP.Software.fp32Context] using hp.1 i hi

open LayerNormWelfold.FPStages (roundingInitial witnessOne witness_rounding)
specification normalize_nonempty_rounding [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest .ieeeValue)
      (normalizeSourceChunk "rstd" "x" "weight" "bias" "mean" "out" 1 2) roundingInitial = some final ∧
      final.mem "out" 0 = .mk .real witnessOne ∧
      final.mem "out" 1 = roundingInitial.mem "out" 1 ∧
      (1 + 1 / 33554432 : Rat) ≠ 1 := by
  obtain ⟨final, he, hp⟩ := normalize_software_fp32 rest .ieeeValue
    "mean" "rstd" "x" "weight" "bias" "out" 1 2 roundingInitial
  refine ⟨final, he, ?_, ?_, witness_rounding.2.2.2⟩
  · have hv := hp.1 (⟨0, by decide⟩, PUnit.unit) (by simp [column, roundingInitial])
    exact hv.trans (congrArg (Cell.mk .real) witness_rounding.1)
  · apply hp.2 "out" 1
    right
    intro i hi
    simp [column, roundingInitial] at hi
    simp [rowAddress, column, roundingInitial]
    omega

#axiomsClean normalize_source_fp32
#axiomsClean normalize_software_fp32
#axiomsClean normalize_nonempty_rounding
end
end VeriTile.Bench.TritonBenchG.FusedLayernormTriton.FPStages
