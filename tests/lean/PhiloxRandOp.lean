import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.EvalOpR
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Random.Values

/-! Operational regressions for the source-profiled random primitive.
The half-way integer at 2^24+1 makes omitting the first fp32 rounding
observable in the dropout comparison, not just in a generator helper.
-/
namespace VeriTile.Bench.Tests.PhiloxRandOp
open VeriTile.Triton

set_option maxRecDepth 10000
set_option maxHeartbeats 2000000

local instance : Inhabited (FP.Value .fp32) := ⟨FP.Value.ofBits .fp32 0⟩

def zeroRoundProfile : Random.SourceProfile := ⟨.i32, .i32, 0⟩
def boundary : Int := 0x01000001
def boundaryOp : Op .real [] :=
  .rand zeroRoundProfile (.constInt 0) (.constInt boundary)

theorem boundary_bits :
    (Random.rand 0 boundary .bits32 0).bits = (0x3bffffff : BitVec 32) := by
  decide +kernel

theorem boundary_rat :
    Random.randRat 0 boundary .bits32 0 = (16777215 : Rat) / 2^31 := by
  apply Option.some.inj
  rw [← Random.rand_toRat_randRat]
  decide +kernel

noncomputable def threshold : ℝ := 16777215 / 2^31

theorem boundary_real : Random.randReal 0 boundary .bits32 0 = threshold := by
  unfold Random.randReal
  rw [boundary_rat]
  norm_num [threshold]

theorem boundary_execution (s : BlockState) :
    evalOp boundaryOp s = some (Tile.scalar (some threshold)) := by
  have valid : zeroRoundProfile.inputsValid 0 (fun _ : TileIndex [] => boundary) = true :=
    by decide
  simp only [boundaryOp, evalOp, bind, Option.bind, Tile.scalar]
  rw [if_pos valid]
  change some (Tile.scalar (dtype := .real)
    (some (Random.randReal 0 boundary .bits32 0))) = _
  rw [boundary_real]
  rfl

theorem boundary_rounding_model (R : RoundingModel) (s : BlockState) :
    evalOpR R boundaryOp s = some (Tile.scalar (some threshold)) := by
  have valid : zeroRoundProfile.inputsValid 0 (fun _ : TileIndex [] => boundary) = true :=
    by decide
  simp only [boundaryOp, evalOpR, bind, Option.bind, Tile.scalar]
  rw [if_pos valid]
  change some (Tile.scalar (dtype := .real)
    (some (Random.randReal 0 boundary .bits32 0))) = _
  rw [boundary_real]
  rfl

theorem boundary_structural {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType)
    (s : FP.Structural.State α) :
    FP.Structural.evalOp M p boundaryOp s =
      some (fun _ => M.fp32Bits ⟨0x3bffffff⟩) := by
  change some (fun _ => M.fp32Bits ⟨(Random.rand 0 boundary .bits32 0).bits⟩) = _
  rw [boundary_bits]

noncomputable def dropoutMask : Op .bool [] :=
  .gt .real .nil boundaryOp (.const threshold)

theorem dropout_mask_false (s : BlockState) :
    evalOp dropoutMask s = some (Tile.scalar false) := by
  classical
  simp only [dropoutMask, evalOp, boundary_execution, bind, Option.bind]
  apply congrArg some
  apply Tile.ext
  intro i
  simp only [Tile.cop, Tile.scalar, ComparableDType.gt, Broadcast.leftIndex,
    Broadcast.rightIndex]
  simp

/-- Replacing integer-to-fp32 rounding with exact arithmetic changes the
comparison at this reachable zero-round offset. The scale is the exact fp32
literal; only the integer conversion is omitted in this rejected expression. -/
theorem omitted_first_round_changes_mask :
    ((0x01000001 : ℝ) * (16777215 / 2^55)) > threshold := by
  norm_num [threshold]

/-- The same threshold comparison under the concrete fp32 software comparator.
Only the two floating operations actually used by this expression are fixed;
the rest of the structural algebra is irrelevant to this regression. -/
theorem dropout_mask_fp32
    (M : FP.Structural.Algebra (FP.Value .fp32)) (p : Option ComputeDType)
    (s : FP.Structural.State (FP.Value .fp32))
    (hbits : M.fp32Bits = fun b => (⟨b.bits⟩ : FP.Value .fp32))
    (hcmp : M.compareLt p .real = some (FP.lt FP.Config.ieeeValue))
    (hthreshold : s.regs .real [] "threshold" =
      some (fun _ => FP.Value.ofBits .fp32 0x3bffffff)) :
    FP.Structural.evalOp M p
      (.gt .real .nil boundaryOp (.ref .real [] "threshold")) s =
      some (fun _ => false) := by
  have hsame : FP.lt FP.Config.ieeeValue
      (FP.Value.ofBits .fp32 0x3bffffff)
      (FP.Value.ofBits .fp32 0x3bffffff) = false := by decide +kernel
  simp only [FP.Structural.evalOp, FP.Structural.numericLt, boundary_structural,
    hbits, hcmp, hthreshold, bind, Option.bind]
  apply congrArg some
  funext i
  simpa only [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    FP.Value.ofBits] using hsame

theorem invalid_seed_rejected (s : BlockState) :
    evalOp (.rand zeroRoundProfile (.constInt (2^31)) (.constInt 0)) s = none := by
  simp only [evalOp, bind, Option.bind, Tile.scalar]
  rfl

theorem unsigned_negative_seed_rejected (s : BlockState) :
    evalOp (.rand ⟨.u32, .i32, 0⟩ (.constInt (-1)) (.constInt 0)) s = none := by
  simp only [evalOp, bind, Option.bind, Tile.scalar]
  rfl

theorem invalid_offset_rejected (s : BlockState) :
    evalOp (.rand zeroRoundProfile (.constInt 0) (.constInt (2^31))) s = none := by
  simp only [evalOp, bind, Option.bind, Tile.scalar]
  rfl

theorem negative_i32_seed_admitted :
    zeroRoundProfile.inputsValid (-1) (fun _ : TileIndex [] => -1) = true := by
  decide

theorem unsigned_high_word_admitted :
    (Random.SourceProfile.mk .u64 .u64 0).inputsValid (2^64-1)
      (fun _ : TileIndex [] => 2^64-1) = true := by
  decide

theorem empty_tile_still_checks_seed (s : BlockState) :
    evalOp (.rand zeroRoundProfile (.constInt (2^31))
      (.full [0] (.constInt 0))) s = none := by
  simp only [evalOp, bind, Option.bind, Tile.scalar]
  rfl

theorem missing_seed_is_not_skipped_on_empty_tile (s : BlockState)
    (h : s.regs .int [] "seed" = none) :
    evalOp (.rand zeroRoundProfile (.ref .int [] "seed")
      (.full [0] (.constInt 0))) s = none := by
  simp [evalOp, h]

theorem missing_offsets_are_not_skipped (s : BlockState)
    (h : s.regs .int [0] "offsets" = none) :
    evalOp (.rand zeroRoundProfile (.constInt 0) (.ref .int [0] "offsets")) s = none := by
  simp [evalOp, h]

theorem configuration_survives_erasure (cfg : Random.SourceProfile)
    (seed : Op .int []) (offsets : Op .int shape) :
    (Op.rand cfg seed offsets).eraseDType =
      .rand cfg seed.eraseDType offsets.eraseDType := by
  rw [Op.eraseDType]

#print axioms boundary_bits
#print axioms boundary_execution
#print axioms boundary_rounding_model
#print axioms boundary_structural
#print axioms dropout_mask_false
#print axioms omitted_first_round_changes_mask
#print axioms dropout_mask_fp32
#print axioms invalid_seed_rejected
#print axioms invalid_offset_rejected
#print axioms empty_tile_still_checks_seed
#print axioms configuration_survives_erasure
end VeriTile.Bench.Tests.PhiloxRandOp
