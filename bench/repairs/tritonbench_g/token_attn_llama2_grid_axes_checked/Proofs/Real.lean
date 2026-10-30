import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Proofs.Geometry
import Mathlib.Analysis.SpecialFunctions.Sqrt

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.Whole
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

/-- Independent score-stage mathematics; softmax and values are absent. -/
def dotScore (D : Nat) (scale : ℝ) (q k : Fin D → ℝ) : ℝ :=
  (∑ d, q d * k d) * scale

def idealScale (D : Nat) : ℝ := 1 / Real.sqrt (D : ℝ)

def realFormula (c : Call) (scale : ℝ) (m : Metadata c) (x : Buffers ℝ c)
    (b h t : Nat) : ℝ := dotScore c.dim scale (query c x b h) (key c m x b h t)

def realSum {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (xs : Values ℝ .real shape) : Values ℝ .real (TileShape.reduceShape shape ax keep) := by
  cases keep
  · exact fun out => ∑ j : Fin (TileShape.axisDim shape ax),
      xs (TileShape.insertAxisIndex shape ax out j)
  · exact fun out => ∑ j : Fin (TileShape.axisDim shape ax),
      xs (TileShape.replaceAxisIndex shape ax out j)

/-- Exact Real projection of all reached floating primitives. -/
def realAlgebra : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op x y => match op with
    | .add => x+y | .sub => x-y | .mul => x*y | .div => x/y | _ => 0
  unary := fun _ _ _ => 0
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ => fun _ => 0
  reduceSum := fun _ {_} ax keep xs => realSum ax keep xs

theorem score_real (p : Parameters) (s : State ℝ) (i : Index p) (hi : active p s i) :
    score realAlgebra p s i = dotScore p.dimensions p.scale
      (fun d => (s.mem p.q (queryOffset p s d)).read .fp32)
      (fun d => (s.mem p.k (keyOffset p s i d)).read .fp32) := by
  rcases i with ⟨i,⟨⟩⟩
  simp only [score,realAlgebra,realSum,TileShape.axisDim,TileShape.eraseAxis,
    TileShape.insertAxisIndex,products,queryValue,keyValue,hi,ite_true,dotScore]
  apply congrArg (fun z : ℝ => z*p.scale)
  apply Finset.sum_congr
  · ext d; simp
  · intro d _; rfl

theorem score_initial_real (c : Call) (scale : ℝ) (m : Metadata c) (x : Buffers ℝ c)
    (s : State ℝ) (v : Valid c m) (b h t : Nat)
    (hb : b<c.batches) (ht : t<m.len b) :
    score realAlgebra (c.parameters scale)
      (start c.grid (b,h,t/32) (initial c m x s)) (lane c scale t) =
      realFormula c scale m x b h t := by
  have hsplit : t/32*32+t%32=t := by omega
  have hl := length_bound c m v b hb
  have ha : active (c.parameters scale) (start c.grid (b,h,t/32) (initial c m x s))
      (lane c scale t) := by
    rw [initial_active]
    change c.maxLength-m.len b+(t/32*32+t%32)<c.maxLength
    omega
  rw [score_real _ _ _ ha]
  unfold realFormula
  apply congrArg₂ (dotScore c.dim scale)
  · funext d
    simp [query,queryOffset,Call.parameters,start,ids,initial]
  · funext d
    have hwindow : c.maxLength-m.len b+t<c.maxLength := by omega
    simp [key,keyOffset,keyLocation,ha,Call.parameters,start,ids,initial,
      Region.cast,Metadata.page,windowIndex,length,token,lane,hsplit,active,hwindow]

#axiomsClean score_real
#axiomsClean score_initial_real
#specNonCircular dotScore avoiding [VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue.source,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.Whole
