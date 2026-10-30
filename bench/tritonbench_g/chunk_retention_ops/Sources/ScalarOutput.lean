import bench.tritonbench_g.chunk_retention_ops.Sources.ScalarAlgebra
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
noncomputable section
open scoped BigOperators
open MathTarget

namespace ScalarSource
/-- The source's active prefix of one chunk, after zero padding and the causal mask.
The saved carry is read from the current history buffer by the execution layer. -/
def localState (c : Shape) (xs : Inputs ℝ) (head chunk lane key value : Nat)
    (history : Nat → Nat → ℝ) : ℝ :=
  (headRate c head)^(lane+1)*history key value+
  ∑ i ∈ Finset.range (lane+1), (headRate c head)^(lane-i)*
    (xs.k head (chunk*64+i) key*xs.v head (chunk*64+i) value)

def localCoefficient (c : Shape) (xs : Inputs ℝ) (scale : ℝ)
    (head chunk lane key value : Nat) (dhistory : Nat → Nat → ℝ) : ℝ :=
  (headRate c head)^(c.length chunk-1-lane)*dhistory key value+
  ∑ i ∈ Finset.range (c.length chunk-lane), (headRate c head)^i*
    forcing xs scale head key value (chunk*64+lane+i)

/-- Two source dot contractions in fwd_o, before their final scale. -/
def outputTile (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (head chunk lane value : Nat)
    (history : Nat → Nat → ℝ) : ℝ :=
  ((∑ key ∈ Finset.range c.K,
      (xs.q head (chunk*64+lane) key*(headRate c head)^(lane+1))*history key value)+
    ∑ i ∈ Finset.range (lane+1),
      (∑ key ∈ Finset.range c.K, xs.q head (chunk*64+lane) key*xs.k head (chunk*64+i) key)*
      ((headRate c head)^(lane-i)*xs.v head (chunk*64+i) value))*scale

/-- The source dQ contractions after the value tile accumulator and causal mask. -/
def queryTile (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (head chunk lane key : Nat)
    (history : Nat → Nat → ℝ) : ℝ :=
  ((∑ value ∈ Finset.range c.V,
      (xs.incoming head (chunk*64+lane) value*(headRate c head)^(lane+1))*history key value)+
    ∑ i ∈ Finset.range (lane+1),
      (∑ value ∈ Finset.range c.V,
        xs.incoming head (chunk*64+lane) value*xs.v head (chunk*64+i) value)*
      ((headRate c head)^(lane-i)*xs.k head (chunk*64+i) key))*scale

/-- The transposed dS dot is the chunk suffix; its weights include scale once. -/
def keyTile (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (head chunk lane key : Nat)
    (dhistory : Nat → Nat → ℝ) : ℝ :=
  (∑ value ∈ Finset.range c.V, xs.v head (chunk*64+lane) value*dhistory key value)*
    (headRate c head)^(c.length chunk-1-lane)+
  ∑ i ∈ Finset.range (c.length chunk-lane),
    (∑ value ∈ Finset.range c.V,
      xs.v head (chunk*64+lane) value*xs.incoming head (chunk*64+lane+i) value)*
    ((headRate c head)^i*scale*xs.q head (chunk*64+lane+i) key)

/-- One dV allocation slice owns exactly one key tile. -/
def valuePartial (c : Shape) (xs : Inputs ℝ) (scale : ℝ)
    (block head chunk lane value : Nat) (dhistory : Nat → Nat → ℝ) : ℝ :=
  ∑ i ∈ Finset.range c.BK, if block*c.BK+i<c.K then
    xs.k head (chunk*64+lane) (block*c.BK+i)*
      localCoefficient c xs scale head chunk lane (block*c.BK+i) value dhistory else 0

def valueSum (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (head chunk lane value : Nat)
    (dhistory : Nat → Nat → ℝ) : ℝ :=
  ∑ block ∈ Finset.range c.keyBlocks, valuePartial c xs scale block head chunk lane value dhistory

theorem local_state_correct (c : Shape) (xs : Inputs ℝ) (hasSeed : Bool)
    (head chunk lane key value : Nat) (hc : chunk<c.chunks) :
    localState c xs head chunk lane key value (forwardCarry c xs hasSeed head chunk)=
      state c xs hasSeed head (chunk*64+lane+1) key value := by
  rw [show chunk*64+lane+1=chunk*64+(lane+1) by omega]
  unfold state
  rw [weighted_add]
  unfold localState
  rw [history_correct c xs hasSeed head chunk key value hc]
  simp only [weighted,Nat.add_sub_cancel]
  rfl

theorem local_coefficient_correct (c : Shape) (xs : Inputs ℝ) (scale : ℝ)
    (head chunk lane key value : Nat) (hc : chunk<c.chunks) (hl : lane<c.length chunk) :
    localCoefficient c xs scale head chunk lane key value
      (reverseCarry c xs scale head (c.chunks-1-chunk))=
      tokenCoefficient c xs scale head (chunk*64+lane) key value := by
  unfold localCoefficient
  rw [history_gradient_correct c xs scale head chunk key value hc,
    token_chunk c xs scale head chunk lane key value hc hl]

theorem output_contraction (c : Shape) (xs : Inputs ℝ) (scale : ℝ)
    (head chunk lane value : Nat) (history : Nat → Nat → ℝ) :
    outputTile c xs scale head chunk lane value history=
      scale*∑ key ∈ Finset.range c.K,
        xs.q head (chunk*64+lane) key*localState c xs head chunk lane key value history := by
  rw [outputTile,ScalarAlgebra.contraction_scaled]
  unfold localState
  congr 1
  apply Finset.sum_congr rfl
  intro key hk
  congr 1
  congr 1
  apply Finset.sum_congr rfl
  intro i hi
  ring

theorem query_contraction (c : Shape) (xs : Inputs ℝ) (scale : ℝ)
    (head chunk lane key : Nat) (history : Nat → Nat → ℝ) :
    queryTile c xs scale head chunk lane key history=
      scale*∑ value ∈ Finset.range c.V,
        xs.incoming head (chunk*64+lane) value*localState c xs head chunk lane key value history := by
  rw [queryTile,ScalarAlgebra.contraction_scaled]
  unfold localState
  congr 1
  apply Finset.sum_congr rfl
  intro value hv
  congr 1
  congr 1
  apply Finset.sum_congr rfl
  intro i hi
  ring

theorem key_contraction (c : Shape) (xs : Inputs ℝ) (scale : ℝ)
    (head chunk lane key : Nat) (dhistory : Nat → Nat → ℝ) :
    keyTile c xs scale head chunk lane key dhistory=
      ∑ value ∈ Finset.range c.V,
        xs.v head (chunk*64+lane) value*localCoefficient c xs scale head chunk lane key value dhistory := by
  rw [keyTile,ScalarAlgebra.contraction]
  unfold localCoefficient forcing
  apply Finset.sum_congr rfl
  intro value hv
  congr 1
  congr 1
  apply Finset.sum_congr rfl
  intro i hi
  ring

theorem value_reduction (c : Shape) (xs : Inputs ℝ) (scale : ℝ)
    (head chunk lane value : Nat) (dhistory : Nat → Nat → ℝ) :
    valueSum c xs scale head chunk lane value dhistory=
      ∑ key ∈ Finset.range c.K,
        xs.k head (chunk*64+lane) key*localCoefficient c xs scale head chunk lane key value dhistory := by
  unfold valueSum valuePartial Shape.keyBlocks
  exact ScalarAlgebra.padded_sum (fun key => xs.k head (chunk*64+lane) key*
    localCoefficient c xs scale head chunk lane key value dhistory) c.K c.BK c.key_tile_pos

theorem output_correct (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (hasSeed : Bool)
    (head chunk lane value : Nat) (hc : chunk<c.chunks) :
    outputTile c xs scale head chunk lane value (forwardCarry c xs hasSeed head chunk)=
      output c xs scale hasSeed head (chunk*64+lane) value := by
  rw [output_contraction]
  simp_rw [local_state_correct c xs hasSeed head chunk lane _ _ hc]
  rfl

theorem query_correct (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (hasSeed : Bool)
    (head chunk lane key : Nat) (hc : chunk<c.chunks) :
    queryTile c xs scale head chunk lane key (forwardCarry c xs hasSeed head chunk)=
      queryGradient c xs scale hasSeed head (chunk*64+lane) key := by
  rw [query_contraction]
  simp_rw [local_state_correct c xs hasSeed head chunk lane _ _ hc]
  rfl

theorem key_correct (c : Shape) (xs : Inputs ℝ) (scale : ℝ)
    (head chunk lane key : Nat) (hc : chunk<c.chunks) (hl : lane<c.length chunk) :
    keyTile c xs scale head chunk lane key (reverseCarry c xs scale head (c.chunks-1-chunk))=
      keyGradient c xs scale head (chunk*64+lane) key := by
  rw [key_contraction]
  simp_rw [local_coefficient_correct c xs scale head chunk lane _ _ hc hl]
  rfl

theorem value_correct (c : Shape) (xs : Inputs ℝ) (scale : ℝ)
    (head chunk lane value : Nat) (hc : chunk<c.chunks) (hl : lane<c.length chunk) :
    valueSum c xs scale head chunk lane value (reverseCarry c xs scale head (c.chunks-1-chunk))=
      valueGradient c xs scale head (chunk*64+lane) value := by
  rw [value_reduction]
  simp_rw [local_coefficient_correct c xs scale head chunk lane _ _ hc hl]
  rfl

#axiomsClean output_correct
#axiomsClean query_correct
#axiomsClean key_correct
#axiomsClean value_correct
#specNonCircular MathTarget.output avoiding [outputTile,output_correct]
#specNonCircular MathTarget.valueGradient avoiding [valuePartial,value_correct]
end ScalarSource
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001

