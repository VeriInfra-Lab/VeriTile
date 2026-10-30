import bench.tritonbench_g.token_attn_reduceV.Proofs.Contract
namespace VeriTile.Bench.Repairs.TokenAttnReduceV.Real
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

def algebra : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op x y => match op with
    | .add => x+y | .sub => x-y | .mul => x*y | .div => x/y
    | .max => max x y | .pow => 0
  unary := fun _ _ _ => 0
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => 0
  reduceSum := fun _ {_} _ _ _ _ => 0

def sum : SumPrimitive ℝ where
  apply := fun _ {s} ax keep xs => match keep with
    | Bool.false => fun out => ∑ k : Fin (TileShape.axisDim s ax), xs (TileShape.insertAxisIndex s ax out k)
    | Bool.true => fun out => ∑ k : Fin (TileShape.axisDim s ax), xs (TileShape.replaceAxisIndex s ax out k)

/-- Independent page-table weighted sum; no source execution, tiles or padding. -/
def weighted (p : Parameters) (s : State ℝ) (d : Nat) : ℝ :=
  ∑ n ∈ Finset.range (count p s),
    (s.mem p.prob (probOffset p s n)).read .fp32 *
      (s.mem p.value (valueOffset p s d +
        (s.mem (Region.cast p.tokens) (gatherOffset p s n)).read .nat*p.valueBatchStride)).read .fp32

def term (p : Parameters) (s : State ℝ) (n d : Nat) : ℝ :=
  if n<count p s then
    (s.mem p.prob (probOffset p s n)).read .fp32 *
      (s.mem p.value (valueOffset p s d +
        (s.mem (Region.cast p.tokens) (gatherOffset p s n)).read .nat*p.valueBatchStride)).read .fp32
  else 0

def partialSum (p : Parameters) (s : State ℝ) (n : Nat) : TileIndex [p.width] → ℝ :=
  fun i => ∑ k ∈ Finset.range n, term p s k i.1.val

theorem product_term (p : Parameters) (s : State ℝ) (n : Nat) (i : TileIndex [p.block,p.width]) :
    products algebra p s n i = term p s (n+i.1.val) i.2.1.val := by
  by_cases h : n+i.1.val<count p s
  · simp [products,probability,value,gather,padding,algebra,term,h,probOffset,gatherOffset,
      Nat.add_mul,Nat.mul_add,Nat.add_comm,Nat.add_left_comm,Nat.add_assoc]
  · simp [products,probability,value,gather,padding,algebra,term,h]

theorem update_prefix (p : Parameters) (s : State ℝ) (n : Nat) :
    update algebra sum p s n (partialSum p s n) = partialSum p s (n+p.block) := by
  funext i
  rcases i with ⟨i,⟨⟩⟩
  simp only [update,sum,algebra]
  change partialSum p s n (i,PUnit.unit) +
    (∑ k : Fin p.block, products algebra p s n (k,i,PUnit.unit)) = _
  simp only [product_term,partialSum]
  rw [Finset.sum_range_add]
  congr 1
  exact Fin.sum_univ_eq_sum_range (fun k => term p s (n+k) i.val) p.block

theorem prefix_done (p : Parameters) (s : State ℝ) (n : Nat) (h : count p s≤n) :
    partialSum p s n = fun i => weighted p s i.1.val := by
  funext i
  unfold partialSum weighted
  rw [← Finset.sum_range_add_sum_Ico _ h]
  have tail : ∑ k ∈ Finset.Ico (count p s) n,term p s k i.1.val=0 := by
    apply Finset.sum_eq_zero
    intro k hk
    simp only [Finset.mem_Ico] at hk
    simp [term,Nat.not_lt.mpr hk.1]
  rw [tail,add_zero]
  apply Finset.sum_congr rfl
  intro k hk
  simp [term,Finset.mem_range.mp hk]

theorem fold_prefix (p : Parameters) (s : State ℝ) (n : Nat) (positive : 0<p.block) :
    foldRange (update algebra sum p s) n (count p s) p.block (partialSum p s n) =
      fun i => weighted p s i.1.val := by
  rw [foldRange]
  have nonzero : p.block≠0 := Nat.ne_of_gt positive
  simp only [nonzero,↓reduceDIte]
  by_cases live : n<count p s
  · simp only [live,↓reduceDIte,update_prefix]
    exact fold_prefix p s (n+p.block) positive
  · simp only [live,↓reduceDIte]
    exact prefix_done p s n (Nat.le_of_not_gt live)
termination_by count p s-n
decreasing_by omega

theorem output_weighted (p : Parameters) (s : State ℝ) (positive : 0<p.block)
    (i : TileIndex [p.width]) : outputValues algebra sum p s i=weighted p s i.1.val := by
  have initial_eq : initial algebra p=partialSum p s 0 := by funext j; simp [initial,algebra,partialSum]
  change foldRange (update algebra sum p s) 0 (count p s) p.block (initial algebra p) i = _
  rw [initial_eq,fold_prefix p s 0 positive]

theorem complete (p : Parameters) (s : State ℝ) (positive : 0<p.block)
    (injective : Function.Injective (outputOffset p s)) :
    ∃ t, FP.Structural.exec (withSum algebra sum) (kernel p) s=some t ∧
      (∀ i : TileIndex [p.width], (t.mem p.output (outputOffset p s i)).read .fp32=weighted p s i.1.val) ∧
      (∀ r o, Outside p s r o → t.mem r o=s.mem r o) := by
  obtain ⟨t,run,mem⟩ := complete_run algebra sum p s
  refine ⟨t,run,?_,?_⟩
  · intro i
    rw [mem,reference_readback algebra sum p s injective i,Cell.read_same,output_weighted p s positive]
  · intro r o outside
    rw [mem]
    exact reference_frame algebra sum p s r o outside

#axiomsClean product_term
#axiomsClean output_weighted
#axiomsClean complete
#specNonCircular weighted avoiding [FP.Structural.exec,reference,outputValues]
end
end VeriTile.Bench.Repairs.TokenAttnReduceV.Real
