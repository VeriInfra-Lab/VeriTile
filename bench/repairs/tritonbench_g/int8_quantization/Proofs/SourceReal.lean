import bench.repairs.tritonbench_g.int8_quantization.Proofs.SourceExecution
import bench.repairs.tritonbench_g.int8_quantization.Proofs.Real
import bench.repairs.tritonbench_g.int8_quantization.Proofs.Geometry

namespace VeriTile.Bench.Repairs.Int8Quantization.Revision002.Math
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def listMaximum : List ℝ → ℝ | []=>0 | x::xs=> max x (listMaximum xs)

def realMaximum {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (x : TileIndex shape→ℝ) : TileIndex (TileShape.reduceShape shape ax keep)→ℝ :=
  match keep with
  | Bool.false=> fun out=> listMaximum ((List.finRange (TileShape.axisDim shape ax)).map
      (fun j=> x (TileShape.insertAxisIndex shape ax out j)))
  | Bool.true=> fun out=> listMaximum ((List.finRange (TileShape.axisDim shape ax)).map
      (fun j=> x (TileShape.replaceAxisIndex shape ax out j)))

/-- Mathematical interpretation: casts and loads are exact; the same typed
source, separate multiplication/addition, masks, comparisons and stores run.
This does not identify Real execution with production FP32 execution. -/
def algebra (rest : Algebra ℝ) : Algebra ℝ :=
  {rest with
    literal:=fun _ _ x=> x
    fp32Load:=id
    binary:=fun ctx dtype op x y=> match op with
      | .add=> x+y | .sub=> x-y | .mul=> x*y | .div=> x/y | .max=> max x y
      | .pow=> rest.binary ctx dtype op x y
    unary:=fun ctx op x=> match op with
      | .abs _=>|x| | .neg _=>-x | _=> rest.unary ctx op x
    cast:=fun _ _ _ x=> x
    reduceMax:=fun _=> realMaximum}

def primitives : Model.Primitives ℝ :=
  ⟨fun x y=> decide (x< y),fun x y=> decide (x≤ y),tritonTruncTowardZero,min⟩

def raw (p : Parameters) (entry : State ℝ) (i : Lane p) : ℝ :=
  (entry.mem xRegion (Model.offset p entry i)).read .real

/-- Logical coordinates of this block in row-major order. Geometry.coordinate
proves every caller batch/row/channel has exactly one such active lane, for
arbitrary positive extents and padded width. Inactive padding is excluded. -/
def logicalIndices (p : Parameters) (entry : State ℝ) : List (Fin p.width) :=
  (List.finRange p.width).filter (fun j=> decide (Model.active p entry (j,PUnit.unit)))

def logicalBlock (p : Parameters) (entry : State ℝ) : List ℝ :=
  (logicalIndices p entry).map (fun j=> raw p entry (j,PUnit.unit))

def ideal (p : Parameters) (entry : State ℝ) : Lane p→ℝ :=
  fun i=> if Model.active p entry i then RealSpec.factor p.query p.channels*raw p entry i else 0

def blockMaximum (p : Parameters) (entry : State ℝ) : ℝ :=
  RealSpec.maxAbs ((logicalBlock p entry).map (fun x=> RealSpec.factor p.query p.channels*x))

def FiniteDomain (p : Parameters) (entry : State ℝ) : Prop :=
  ∀ i,Model.active p entry i → |RealSpec.factor p.query p.channels*raw p entry i|≤
    340282346638528859811704183484516925440

theorem transformed_ideal (rest : Algebra ℝ) (p : Parameters) (entry : State ℝ)
    (factor : p.factor=RealSpec.factor p.query p.channels) :
    transformed (algebra rest) p entry=ideal p entry := by
  funext i
  cases query : p.query <;> by_cases active : Model.active p entry i <;>
    norm_num [transformed,loaded,algebra,scalar32,ideal,raw,query,active,factor,RealSpec.factor,mul_comm]

theorem ideal_finite (p : Parameters) (entry : State ℝ) (finite : FiniteDomain p entry)
    (i : Lane p) : |ideal p entry i|≤340282346638528859811704183484516925440 := by
  by_cases active : Model.active p entry i
  · simpa [ideal,active] using finite i active
  · norm_num [ideal,active]

theorem flags_true (rest : Algebra ℝ) (p : Parameters) (entry : State ℝ)
    (factor : p.factor=RealSpec.factor p.query p.channels) (finite : FiniteDomain p entry)
    (i : Lane p) : flags (algebra rest) primitives p entry i=Bool.true := by
  have h:=ideal_finite p entry finite i
  unfold flags
  rw [transformed_ideal rest p entry factor]
  norm_num [algebra,primitives]
  exact h

theorem bad_zero (rest : Algebra ℝ) (p : Parameters) (positive : 0< p.width) (entry : State ℝ)
    (factor : p.factor=RealSpec.factor p.query p.channels) (finite : FiniteDomain p entry) :
    badValue (algebra rest) primitives p positive entry=0 := by
  simp [badValue,flags_true rest p entry factor finite]

theorem sanitized_ideal (rest : Algebra ℝ) (p : Parameters) (entry : State ℝ)
    (factor : p.factor=RealSpec.factor p.query p.channels) (finite : FiniteDomain p entry) :
    sanitized (algebra rest) primitives p entry=ideal p entry := by
  funext i
  simp [sanitized,flags_true rest p entry factor finite,transformed_ideal rest p entry factor]

theorem list_max_abs (xs : List ℝ) : listMaximum (xs.map abs)=RealSpec.maxAbs xs := by
  induction xs with
  | nil=> rfl
  | cons x xs ih=> simp [listMaximum,RealSpec.maxAbs,ih]

theorem padding_max {ι : Type} (xs : List ι) (active : ι→Bool) (value : ι→ℝ) :
    RealSpec.maxAbs (xs.map (fun i=> if active i then value i else 0))=
      RealSpec.maxAbs ((xs.filter active).map value) := by
  induction xs with
  | nil=> rfl
  | cons x xs ih=>
    cases flag : active x
    · simpa [flag,RealSpec.maxAbs,ih,max_eq_right (RealSpec.maxAbs_nonneg _)]
    · simp [flag,RealSpec.maxAbs,ih]

theorem maximum_ideal (rest : Algebra ℝ) (p : Parameters) (entry : State ℝ)
    (factor : p.factor=RealSpec.factor p.query p.channels) (finite : FiniteDomain p entry) :
    maximum (algebra rest) primitives p entry=blockMaximum p entry := by
  rw [maximum,sanitized_ideal rest p entry factor finite]
  change listMaximum ((List.finRange p.width).map (fun j=>|ideal p entry (j,PUnit.unit)|))=_
  rw [show (List.finRange p.width).map (fun j=>|ideal p entry (j,PUnit.unit)|)=
    ((List.finRange p.width).map (fun j=> ideal p entry (j,PUnit.unit))).map abs by simp]
  rw [list_max_abs]
  simpa [ideal,blockMaximum,logicalBlock,logicalIndices,List.map_map] using
    padding_max (List.finRange p.width) (fun j=> decide (Model.active p entry (j,PUnit.unit)))
      (fun j=> RealSpec.factor p.query p.channels*raw p entry (j,PUnit.unit))

theorem scale_ideal (rest : Algebra ℝ) (p : Parameters) (entry : State ℝ)
    (factor : p.factor=RealSpec.factor p.query p.channels) (finite : FiniteDomain p entry) :
    scale (algebra rest) primitives p entry=blockMaximum p entry/127 := by
  unfold scale
  rw [maximum_ideal rest p entry factor finite]
  norm_num [algebra,scalar32]

theorem maximum_bound (p : Parameters) (entry : State ℝ) (i : Lane p)
    (active : Model.active p entry i) :
    |RealSpec.factor p.query p.channels*raw p entry i|≤ blockMaximum p entry := by
  apply RealSpec.abs_le_maxAbs
  apply List.mem_map.mpr
  refine ⟨raw p entry i,?_,rfl⟩
  apply List.mem_map.mpr
  refine ⟨i.1,?_,?_⟩
  · simp [logicalIndices,active]
  · congr 1

theorem quantized_scalar (rest : Algebra ℝ) (p : Parameters) (positive : 0< p.width) (entry : State ℝ)
    (factor : p.factor=RealSpec.factor p.query p.channels) (finite : FiniteDomain p entry)
    (i : Lane p) :
    quantized (algebra rest) primitives p positive entry i=
      RealSpec.sourceScalar (ideal p entry i) (blockMaximum p entry/127) := by
  simp only [quantized,biased,clipped,normalized,quotient,denominator,usable,
    bad_zero rest p positive entry factor finite,sanitized_ideal rest p entry factor finite,
    scale_ideal rest p entry factor finite]
  norm_num only [algebra,primitives,scalar32] 
  simp only [decide_true, Bool.true_and, decide_eq_true_eq]
  simp only [←min_def_lt,RealSpec.sourceScalar,RealSpec.clip]
  norm_num

theorem real_readback (rest : Algebra ℝ) (p : Parameters) (positive : 0< p.width) (entry : State ℝ)
    (factor : p.factor=RealSpec.factor p.query p.channels) (finite : FiniteDomain p entry)
    (i : Lane p) (active : Model.active p entry i) :
    (finalState (algebra rest) primitives p positive entry).mem outRegion (Model.offset p entry i)=
      .mk .int (RealSpec.nearestAway
        (127*(RealSpec.factor p.query p.channels*raw p entry i)/blockMaximum p entry)) := by
  rw [source_readback _ _ _ _ _ i active,quantized_scalar rest p positive entry factor finite]
  rw [RealSpec.scalar_correct _ (blockMaximum p entry) (RealSpec.maxAbs_nonneg _) (by
    simpa [ideal,active] using maximum_bound p entry i active)]
  simp [ideal,active]

theorem real_scale (rest : Algebra ℝ) (p : Parameters) (positive : 0< p.width) (entry : State ℝ)
    (factor : p.factor=RealSpec.factor p.query p.channels) (finite : FiniteDomain p entry) :
    (finalState (algebra rest) primitives p positive entry).mem scaleRegion (Model.group p entry)=
      .mk .real ((RealSpec.target p.query p.channels (logicalBlock p entry)).2) := by
  rw [source_scale]
  simp [savedScale,bad_zero rest p positive entry factor finite,
    scale_ideal rest p entry factor finite,RealSpec.target,blockMaximum]

#axiomsClean real_readback
#axiomsClean real_scale
end
end VeriTile.Bench.Repairs.Int8Quantization.Revision002.Math
