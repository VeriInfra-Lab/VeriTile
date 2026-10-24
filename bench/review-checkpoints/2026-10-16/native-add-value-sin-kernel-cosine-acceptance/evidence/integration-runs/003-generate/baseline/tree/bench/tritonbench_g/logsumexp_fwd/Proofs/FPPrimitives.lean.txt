import bench.tritonbench_g.logsumexp_fwd.Proofs.FPStages
import bench.tritonbench_g.fused_recurrent_delta.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.LogsumexpFwd.FPStages
open VeriTile.Triton FP.Structural
open FusedRecurrentDelta.FPStages (sequentialFP32 sequentialSum)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 15000

/-- Unresolved native primitives are explicit functions on binary32 words.
In particular `maxReduction` records the reduction layout and NaN/tie policy. -/
structure Primitives where
  exp32 : FP.Value .fp32 → FP.Value .fp32
  log32 : FP.Value .fp32 → FP.Value .fp32
  maxReduction : {shape : TileShape} → (ax : Fin shape.length) → (keepDims : Bool) →
    Values (FP.Value .fp32) .real shape →
    Values (FP.Value .fp32) .real (TileShape.reduceShape shape ax keepDims)

def fp32WithPrimitives (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (primitives : Primitives) : Algebra (FP.Value .fp32) :=
  { sequentialFP32 rest cfg with
    unary := fun p op x => match p, op with
      | some .fp32, .exp => primitives.exp32 x
      | some .fp32, .log => primitives.log32 x
      | _, _ => rest.unary p op x
    reduceMax := fun p _ ax keepDims xs =>
      if p = some .fp32 then primitives.maxReduction ax keepDims xs
      else rest.reduceMax p ax keepDims xs }

/-- No callback from `rest` is used in this formula. Max/exp/log stay explicit. -/
def roundedRow [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (primitives : Primitives)
    (s : State (FP.Value .fp32)) (x : RegionName) (D B : Nat) (HAS_SCALE : Bool) : FP.Value .fp32 :=
  let loaded : TileIndex [B] → FP.Value .fp32 := fun i =>
    if s.pids 1 * B + i.1.val < D then (s.mem x (inputAddress s D B i.1)).read .real
    else FP.Value.infinity .fp32 Bool.true
  let scaled := if HAS_SCALE then fun i => FP.mul cfg (loaded i) (FP.Value.ofBits .fp32 0x3f800000)
    else loaded
  let maximum := primitives.maxReduction (shape := [B]) ⟨0, by simp⟩ Bool.false scaled PUnit.unit
  let exponentials := fun i => primitives.exp32 (FP.sub cfg (scaled i) maximum)
  let total := sequentialSum cfg (shape := [B]) ⟨0, by simp⟩ Bool.false exponentials PUnit.unit
  FP.add cfg (primitives.log32 total) maximum

specification row_block_fp32_with_primitives [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (primitives : Primitives)
    (x z : RegionName) (D B : Nat) (HAS_SCALE : Bool) (s : State (FP.Value .fp32)) (hB : 0 < B) :
    ∃ final, FP.Structural.exec (fp32WithPrimitives rest cfg primitives)
      (rowBlock x z D B HAS_SCALE 1) s = some final ∧
      final.mem = (s.write z (outputAddress s D B)
        (.mk .real (roundedRow cfg primitives s x D B HAS_SCALE))).mem := by
  obtain ⟨final, he, hm⟩ := row_block_fp32 (fp32WithPrimitives rest cfg primitives) x z D B HAS_SCALE 1 s hB
  refine ⟨final, he, ?_⟩
  cases HAS_SCALE <;> simpa [rowValue, roundedRow, fp32WithPrimitives, sequentialFP32,
    FP.Software.scalarFP32, FP.Software.fp32Context] using hm

/-- Nonempty program with one valid lane and one -inf padded lane. Execution,
exact output and the full frame hold for every supplied primitive profile. -/
specification row_block_nonempty [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (primitives : Primitives)
    (s : State (FP.Value .fp32)) (hrow : s.pids 0 = 0) (hblock : s.pids 1 = 0) :
    ∃ final, FP.Structural.exec (fp32WithPrimitives rest cfg primitives)
      (rowBlock "x" "z" 1 2 Bool.false 1) s = some final ∧
      final.mem "z" 0 = .mk .real (roundedRow cfg primitives s "x" 1 2 Bool.false) ∧
      (∀ (R : RegionName) (o : Nat), R ≠ "z" ∨ o ≠ 0 → final.mem R o = s.mem R o) := by
  obtain ⟨final, he, hm⟩ := row_block_fp32_with_primitives rest cfg primitives "x" "z" 1 2 Bool.false s (by decide)
  refine ⟨final, he, ?_, ?_⟩
  · rw [hm]
    simp [outputAddress, hrow, hblock]
  · intro R o h
    rw [hm]
    simp only [outputAddress, hrow, hblock, Nat.zero_mul, Nat.zero_add]
    exact State.write_other s "z" R 0 o _ h

#axiomsClean row_block_fp32_with_primitives
#axiomsClean row_block_nonempty
end
end VeriTile.Bench.TritonBenchG.LogsumexpFwd.FPStages
