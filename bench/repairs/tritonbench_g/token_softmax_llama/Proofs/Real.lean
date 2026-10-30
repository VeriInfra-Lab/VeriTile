import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.RowContract
import bench.tritonbench_g.softmax_triton1.Proofs.Real
namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided.Real
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SoftmaxTriton1 (softmaxSpec softmaxInputTile softmaxSpec_eq_formula)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false
abbrev Scalar := WithBot ℝ

/-- Exact mathematical interpretation retains -infinity padding explicitly. -/
def algebra : Algebra Scalar where
  literal := fun _ _ x => some x
  negInf := none
  binary := fun _ _ op => match op with
    | .add => NumericDType.add .real
    | .sub => NumericDType.sub .real
    | .mul => NumericDType.mul .real
    | .div => NumericDType.div .real
    | .max => max
    | .pow => fun _ _ => none
  unary := fun _ op => match op with | .exp => WithBot.realExp | _ => id
  cast := fun _ _ _ x => x
  fromNat := fun _ n => some n
  fromInt := fun _ n => some n
  fp32Bits := fun _ => none
  fp32Load := id
  reduceMax := fun _ {_} ax keep xs =>
    match Tile.reduceMax ax keep ⟨xs⟩ with | some tile => tile.data | none => fun _ => none
  reduceSum := fun _ {_} ax keep xs => (Tile.reduceSum ax keep ⟨xs⟩).data

def input (p : Parameters) (s : State Scalar) (n : Nat) : ℝ :=
  WithBot.unbotD 0 ((s.mem p.input
    (s.pids 1*p.inputHeadStride+(startIndex p s+n)*p.inputTokenStride)).read .fp32)

def NumericInput (p : Parameters) (s : State Scalar) :=
  ∀ n < count p s, ∃ x : ℝ,
    (s.mem p.input (s.pids 1*p.inputHeadStride+(startIndex p s+n)*p.inputTokenStride)).read .fp32=some x

/-- Independent unpadded mathematical softmax: no kernel, graph, reduction
tree, block width, output or execution is referenced in the target. -/
def target (p : Parameters) (s : State Scalar) (i : Nat) : ℝ :=
  Real.exp (input p s i)/(∑ j ∈ Finset.range (count p s),Real.exp (input p s j))

theorem input_row (p : Parameters) (s : State Scalar) (numeric : NumericInput p s) :
    inputRow algebra .fp32 p s=fun i => if i.1.val < count p s then some (input p s i.1.val) else none := by
  funext i
  by_cases hi : i.1.val < count p s
  · obtain ⟨x,hx⟩ := numeric i.1.val hi
    simp [inputRow,algebra,input,hi,toFloat,hx]
  · simp [inputRow,algebra,hi]

theorem output_spec (p : Parameters) (s : State Scalar) (positive : 0 < p.width)
    (numeric : NumericInput p s) (i : Fin p.width) :
    WithBot.unbotD 0 (outputValues algebra .fp32 p s (i,PUnit.unit))=
      softmaxSpec (count p s) p.width (fun j => input p s j.val) i := by
  simp only [outputValues,numerators,input_row p s numeric]
  simp [algebra,softmaxSpec,softmaxInputTile,Tile.reduceMax,Tile.reduceMaxDrop,
    Tile.reduceSum,Tile.reduceSumDrop,TileShape.axisDim,TileShape.eraseAxis,
    TileShape.insertAxisIndex,Tile.bop,Tile.uop,positive]
  rfl

theorem padded_sum (n B : Nat) (cover : n ≤ B) (f : Nat→ℝ) :
    (∑ j : Fin B, if j.val < n then f j.val else 0)=∑ j ∈ Finset.range n,f j := by
  rw [Fin.sum_univ_eq_sum_range (fun j => if j < n then f j else 0)]
  calc
    (∑ j ∈ Finset.range B, if j < n then f j else 0)=∑ j ∈ Finset.range n,if j < n then f j else 0 := by
      symm
      exact Finset.sum_subset (Finset.range_mono cover) (by
        intro j _ outside
        simp only [Finset.mem_range] at outside
        simp [outside])
    _=∑ j ∈ Finset.range n,f j := Finset.sum_congr rfl (by intro j hj; exact if_pos (Finset.mem_range.mp hj))

theorem output_target (p : Parameters) (s : State Scalar) (positive : 0 < p.width)
    (numeric : NumericInput p s) (cover : count p s ≤ p.width)
    (i : Fin p.width) (active : i.val < count p s) :
    WithBot.unbotD 0 (outputValues algebra .fp32 p s (i,PUnit.unit))=target p s i.val := by
  rw [output_spec p s positive numeric,softmaxSpec_eq_formula _ _ positive _ i active,
    padded_sum _ _ cover (fun j => Real.exp (input p s j))]
  rfl

#axiomsClean output_target
#specNonCircular target avoiding [FP.Structural.exec,reference,outputValues]
end
end VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided.Real
