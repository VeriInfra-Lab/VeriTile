import bench.tritonbench_g.fused_rotary_embedding.Proofs.LogicalCoordinates

namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.Correct
open VeriTile.Triton FP.Structural FusedRotaryEmbedding
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Complete FP32 source grid. Q is updated in place; K/V outputs use the
selected cache layout. Int32 length/table channels control indirect writes. -/
def rotaryIO (rest : Algebra ℝ) (c : Config) := PublicIO.logicalIO rest c

/-- Pure split-half scalar rotation. Low and high halves retain their distinct
signs; `input`, `cos` and `sin` are logical rows of the entry-state tensors. -/
def rotate (half : Nat) (input cos sin : Nat → ℝ) (d : Nat) : ℝ :=
  if d<half then input d*cos d-input (d+half)*sin d
  else input (d-half)*sin (d-half)+input d*cos (d-half)

/-- Q/K rotate independently; V is copied into the selected cache layout. -/
def rotaryFormula (c : Config) (xs : PublicIO.RotaryInputs ℝ c.shape) : PublicIO.RotaryOutputs ℝ c.shape where
  q := fun t h d => rotate c.shape.half (PublicIO.tensorInput c.shape xs .q t.val h.val)
    (PublicIO.tensorCos c.shape xs t.val) (PublicIO.tensorSin c.shape xs t.val) d.val
  k := fun t h d => rotate c.shape.half (PublicIO.tensorInput c.shape xs .k t.val h.val)
    (PublicIO.tensorCos c.shape xs t.val) (PublicIO.tensorSin c.shape xs t.val) d.val
  v := fun t h d => PublicIO.tensorInput c.shape xs .v t.val h.val d.val

/-- Every source program satisfies the typed logical IO, actual access bounds
and exterior frame. Config validity supplies geometry and write separation;
the metadata domain binds the actual input arrays to that configuration. -/
specification fused_rotary_real (rest : Algebra ℝ) (c : Config) (valid : c.Valid) :
    Spec.Real (rotaryIO rest c ⊨{PublicIO.logicalMetadataDomain c} rotaryFormula c) :=
  PublicIO.logical_real rest c valid

#axiomsClean fused_rotary_real
#specNonCircular rotaryFormula avoiding [Source.kernel,Source.grid,actualSource]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.Correct
