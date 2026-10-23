import bench.repairs.tritonbench_g.token_attn_mistral_grid_y_checked.Proofs.Model
import bench.repairs.tritonbench_g.token_attn_mistral.Real
import bench.repairs.tritonbench_g.token_attn_mistral.Native

namespace VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked
open VeriTile.Triton FP.Structural TokenAttnMistral
open VeriTile.Bench.Repairs.TokenAttnMistral.Range (foldRange)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def probabilityIndex (m : Inputs) (metadata : Reader) (b h n : Nat) :=
  h*tokens m+(index m metadata .start b+n)
def valueIndex (m : Inputs) (metadata : Reader) (b h n d : Nat) :=
  h/group m*width m+d+page m metadata b n*(kvheads m*width m)

/-- Independent gathered weighted sum on the original caller pointers.
No source program, loop, tile, reduction tree or final state is referenced. -/
def realTarget (m : Inputs) (metadata : Reader) (read : Nat → ℝ) (b h d : Nat) : ℝ :=
  ∑ n ∈ Finset.range (index m metadata .length b),
    if first m metadata b+n < index m metadata .sequence b then
    read (base m .probability+4*probabilityIndex m metadata b h n) *
    read (base m .value+4*valueIndex m metadata b h n d)
    else 0

theorem real_value (m : Inputs) (metadata : Reader) (read : Nat → ℝ)
    (before : Cells ℝ) (b h : Nat) (d : Fin (width m)) :
    outputValues Real.algebra Real.sum (parameters m)
      (Grid.state (view m read metadata before) (b,h)) (d,PUnit.unit)=
        realTarget m metadata read b h d.val := by
  rw [Real.output_weighted _ _ (by change 0<128; decide)]
  simp [Real.weighted,realTarget,probabilityIndex,valueIndex,page,
    parameters,Grid.state,view,count,startIndex,requestIndex,probOffset,gatherOffset,valueOffset,
    sequenceLength,windowStart,first,Nat.mul_one,Nat.add_assoc,Nat.add_comm]

namespace NativeTarget

def pad : Native.Term := .cast none .real .fp32 (.literal none .real 0)

/-- Every padded FP32 product is retained. Logical pointer indices occur in
leaf identities; compiler reduction and contraction choices see only shapes. -/
def products (m : Inputs) (metadata : Reader) (read : Reader) (b h start : Nat)
    (i : TileIndex [128,width m]) : Native.Term :=
  let n := start+i.1.val
  let p := if n < index m metadata .length b then
    .leaf "probability" (probabilityIndex m metadata b h n)
      (read (base m .probability+4*probabilityIndex m metadata b h n)) else pad
  let v := if first m metadata b+n < index m metadata .sequence b then
    .leaf "value" (valueIndex m metadata b h n i.2.1.val)
      (read (base m .value+4*valueIndex m metadata b h n i.2.1.val)) else pad
  .binary (some .fp32) .fp32 .mul p v

def step (R : Native.Reductions) (m : Inputs) (metadata read : Reader)
    (b h start : Nat) (acc : TileIndex [width m] → Native.Term)
    (i : TileIndex [width m]) : Native.Term :=
  let tile := products m metadata read b h start
  let reduced := (Native.reduction R).apply (some .fp32) ⟨0,by simp⟩ Bool.false tile i
  .binary (some .fp32) .real .add (acc i) (.cast (some .fp32) .fp32 .real reduced)

/-- Independent ordered native graph: exact 128-lane blocks, loop order,
rounded multiply/add or selected adjacent FMA, and final FP32 cast. -/
def term (R : Native.Reductions) (m : Inputs) (metadata read : Reader)
    (b h : Nat) (d : Fin (width m)) : Native.Term :=
  .cast none .real .fp32
    (foldRange (step R m metadata read b h) 0 (index m metadata .length b) 128
      (fun _ => .literal (some .fp32) .real 0) (d,PUnit.unit))

def reference (I : Native.Instructions) (R : Native.Reductions) (S : Native.Schedule)
    (m : Inputs) (metadata read : Reader) (b h : Nat) (d : Fin (width m)) : Word :=
  Native.eval I S (term R m metadata read b h d)

theorem products_eq (m : Inputs) (metadata read : Reader) (before : Cells Word)
    (b h start : Nat) :
    TokenAttnMistral.products Native.algebra (parameters m)
      (Grid.state (Native.entry (view m read metadata before)) (b,h)) start=
        products m metadata read b h start := by
  funext i
  by_cases active : start+i.1.val < index m metadata .length b
  all_goals by_cases valueActive : first m metadata b+(start+i.1.val) < index m metadata .sequence b
  all_goals simp only [first,Nat.add_assoc,Nat.add_left_comm,Nat.add_comm] at valueActive
  all_goals simp [TokenAttnMistral.products,probability,value,gather,padding,Native.algebra,
    parameters,Grid.state,Native.entry,Native.inject,view,products,pad,
    count,startIndex,requestIndex,probOffset,gatherOffset,valueOffset,
    probabilityIndex,valueIndex,page,sequenceLength,windowStart,first,
    active,valueActive,Nat.add_assoc,Nat.add_left_comm,Nat.add_comm]

theorem step_eq (R : Native.Reductions) (m : Inputs) (metadata read : Reader)
    (before : Cells Word) (b h : Nat) :
    update Native.algebra (Native.reduction R) (parameters m)
      (Grid.state (Native.entry (view m read metadata before)) (b,h))=
        step R m metadata read b h := by
  funext start acc i
  simp only [update,products_eq]
  rfl

theorem term_eq (R : Native.Reductions) (m : Inputs) (metadata read : Reader)
    (before : Cells Word) (b h : Nat) (d : Fin (width m)) :
    outputValues Native.algebra (Native.reduction R) (parameters m)
      (Grid.state (Native.entry (view m read metadata before)) (b,h)) (d,PUnit.unit)=
        term R m metadata read b h d := by
  unfold outputValues accumulated TokenAttnMistral.initial
  rw [step_eq]
  simp [term,TokenAttnMistral.initial,Native.algebra,count,Grid.state,Native.entry,Native.inject,view,parameters]

#axiomsClean products_eq
#axiomsClean term_eq
#specNonCircular reference avoiding [kernel,FP.Structural.exec,Grid.run,outputValues]
end NativeTarget

#axiomsClean real_value
#specNonCircular realTarget avoiding [kernel,FP.Structural.exec,Grid.run,outputValues]
end
end VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked
