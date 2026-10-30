import bench.tritonbench_g.fast_rms_layernorm.Proofs.FPTensor

/-! Runtime epsilon has a separate scalar register throughout source execution
and the independent reference. The historical literal-specialized path remains
available only to the Real/optional host interpretation. -/
namespace VeriTile.Bench.Repairs.FastRmsLayernorm.RuntimeForward
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

namespace Reference
open FPReference
def inv (M : Algebra α) (h : Call) (epsilon : α) (x : Fin h.block → α) : α :=
  M.unary (some .fp32) .rsqrt
    (add M (div M (reduce M h.block (fun j => mul M (input M h x j) (input M h x j)))
      (literal M (h.cols:ℝ))) epsilon)

def normalized (M : Algebra α) (h : Call) (epsilon : α) (x : Fin h.block → α) (j : Fin h.block) :=
  mul M (input M h x j) (inv M h epsilon x)

def output (M : Algebra α) (h : Call) (epsilon : α) (x w : Fin h.block → α) (j : Fin h.block) : Cell α :=
  .mk .real (cast M (if h.gemma then
    mul M (normalized M h epsilon x j) (add M (input M h w j) (literal M 1))
  else mul M (cast M (normalized M h epsilon x j)) (M.fp32Load (padded M h w j))))

def savedInv (M : Algebra α) (h : Call) (epsilon : α) (x : Fin h.block → α) : Cell α := .mk .real (cast M (inv M h epsilon x))
/-- The Real/optional historical interpretation is exactly the specialization
that supplies the mathematical epsilon literal at this one scalar position. -/
theorem literal_inv (M : Algebra α) (h : Call) (x : Fin h.block → α) :
    inv M h (FPReference.literal M h.eps) x=FPReference.inv M h x := rfl
theorem literal_output (M : Algebra α) (h : Call) (x w : Fin h.block → α) (j : Fin h.block) :
    output M h (FPReference.literal M h.eps) x w j=FPReference.output M h x w j := rfl
theorem literal_saved (M : Algebra α) (h : Call) (x : Fin h.block → α) :
    savedInv M h (FPReference.literal M h.eps) x=FPReference.savedInv M h x := rfl
end Reference
namespace Source
/-- Only the inverse-statistic assignment (slot 9) changes when epsilon becomes
an external register. All loads, masks, addresses, casts and stores are identical. -/
theorem runtime_body (h : Call) :
    (Kernels.forwardRuntime h).surfaceBody.eraseIdx 9=(Kernels.forward h).surfaceBody.eraseIdx 9 := by
  unfold Kernels.forwardRuntime Kernels.forward
  split <;> rfl

theorem runtime_length (h : Call) :
    (Kernels.forwardRuntime h).surfaceBody.length=(Kernels.forward h).surfaceBody.length := by
  unfold Kernels.forwardRuntime Kernels.forward
  split <;> rfl

#axiomsClean runtime_body
#axiomsClean runtime_length
open FPSource
def bind (u : State α) (epsilon : α) : State α := u.setReg "eps" .real [] (fun _ => epsilon)
def forwardHeap (M : Algebra α) (h : Call) (epsilon : α) (u : State α) : RegionName → Nat → Cell α :=
  let x:=fun j : Fin h.block => (u.mem "x" (u.pids 0*h.cols+j.val)).read .real
  let w:=fun j : Fin h.block => (u.mem "w" j.val).read .real
  commit "y" (fun i : TileIndex [h.block] => u.pids 0*h.cols+i.1.val)
    (fun i => Reference.output M h epsilon x w i.1) (fun i => i.1.val<h.cols) (TileShape.allIndices [h.block])
    (update "r" (u.pids 0) (Reference.savedInv M h epsilon x) u.mem)

theorem forward_memory (M : Algebra α) (h : Call) (epsilon : α) (u : State α) :
    ∃ v, FP.Structural.exec (engine M) (Kernels.forwardRuntime h) (bind u epsilon)=some v ∧ v.mem=forwardHeap M h epsilon u := by
  unfold Kernels.forwardRuntime
  split <;> rename_i branch
  all_goals
    simp [branch,ofFloat,toFloat,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,FP.Structural.exec,Kernels.regularRuntimeSource,Kernels.gemmaRuntimeSource,bind,
      ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,natLt,numericLt,numericLe,numericEq,bop,Region.cast,
      State.setReg,FP.Structural.store,TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
      forwardHeap,Reference.output,Reference.savedInv,Reference.normalized,Reference.inv,
      FPReference.reduce,FPReference.input,FPReference.padded,FPReference.literal,FPReference.cast,
      FPReference.add,FPReference.mul,FPReference.div,fold_mem]
    <;> dsimp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,State.write,update]
    <;> norm_num
    <;> rfl


#axiomsClean forward_memory
end Source
namespace Memory
open FPSource FPMemory
theorem forward_complete (M : Algebra α) (h : Call) (epsilon : α) (u : State α) :
    ∃ v, FP.Structural.exec (FP.ProfiledRewrite.engine M) (Kernels.forwardRuntime h) (Source.bind u epsilon)=some v ∧
      (∀ j : Fin h.cols, v.mem "y" (u.pids 0*h.cols+j.val)=
        Reference.output M h epsilon (raw h u "x" h.cols) (raw h u "w" 0) ⟨j.val,j.isLt.trans_le h.covered⟩) ∧
      v.mem "r" (u.pids 0)=Reference.savedInv M h epsilon (raw h u "x" h.cols) ∧
      (∀ q o,¬forwardFootprint h u q o → v.mem q o=u.mem q o) := by
  obtain ⟨v,run,mem⟩:=Source.forward_memory M h epsilon u
  refine ⟨v,run,?_,?_,?_⟩
  · intro j
    rw [mem]
    let initial:=u.write "r" (u.pids 0) (Reference.savedInv M h epsilon (raw h u "x" h.cols))
    change commit "y" _ _ _ _ initial.mem _ _=_
    have readback := commit_readback "y" (fun i : TileIndex [h.block] => u.pids 0*h.cols+i.1.val)
      (fun i => Reference.output M h epsilon (raw h u "x" h.cols) (raw h u "w" 0) i.1)
      (fun i => i.1.val<h.cols) initial (⟨j.val,j.isLt.trans_le h.covered⟩,PUnit.unit) j.isLt (row_injective _ _)
    have weightZero : raw h u "w" 0=(fun k : Fin h.block => (u.mem "w" k.val).read .real) := by
      funext k
      simp [raw]
    rw [weightZero] at readback ⊢
    exact readback
  · rw [mem]
    let initial:=u.write "r" (u.pids 0) (Reference.savedInv M h epsilon (raw h u "x" h.cols))
    change commit "y" _ _ _ _ initial.mem _ _=_
    rw [commit_frame _ _ _ _ _ initial _ _ (Or.inl (by decide))]
    exact State.write_same _ _ _ _
  · intro q o outside
    rw [mem]
    let initial:=u.write "r" (u.pids 0) (Reference.savedInv M h epsilon (raw h u "x" h.cols))
    change commit "y" _ _ _ _ initial.mem q o=_
    rw [commit_frame]
    · apply State.write_other
      by_cases same : q="r"
      · right
        intro eq
        exact outside (Or.inr ⟨same,eq⟩)
      · exact Or.inl same
    · by_cases same : q="y"
      · right
        intro i _ active eq
        exact outside (Or.inl ⟨same,⟨⟨i.1.val,active⟩,eq⟩⟩)
      · exact Or.inl same


end Memory
namespace Grid
open FPGridForward (Address Heap entry)
abbrev Output (h : Call) := Fin h.cols ⊕ Unit
def address (h : Call) (row : Fin h.rows) : Output h → Address
  | .inl j => ("y",row.val*h.cols+j.val)
  | .inr _ => ("r",row.val)
def writes (h : Call) (row : Fin h.rows) (a : Address) : Prop :=
  (∃ j : Fin h.cols,a=(("y":RegionName),row.val*h.cols+j.val)) ∨ a=(("r":RegionName),row.val)
def reads (_h : Call) (_row : Fin _h.rows) (a : Address) : Prop := a.1="x" ∨ a.1="w"
def expected (M : Algebra α) (h : Call) (epsilon : α) (heap : Heap α) (row : Fin h.rows) : Output h → Cell α
  | .inl j => Reference.output M h epsilon
      (FPMemory.raw h (entry heap row.val h.rows) "x" h.cols)
      (FPMemory.raw h (entry heap row.val h.rows) "w" 0) ⟨j.val,j.isLt.trans_le h.covered⟩
  | .inr _ => Reference.savedInv M h epsilon (FPMemory.raw h (entry heap row.val h.rows) "x" h.cols)
def execute (M : Algebra α) (h : Call) (epsilon : α) (row : Fin h.rows) (heap : Heap α) : Option (Heap α) :=
  (FP.Structural.exec (engine M) (Kernels.forwardRuntime h) (Source.bind (entry heap row.val h.rows) epsilon)).map
    (fun v a => v.mem a.1 a.2)

theorem inside (h : Call) (row : Fin h.rows) (i : Output h) : writes h row (address h row i) := by
  cases i with
  | inl j => exact Or.inl ⟨j,rfl⟩
  | inr _ => exact Or.inr rfl

theorem separate (h : Call) (a b : Fin h.rows) (ne : a≠b) (addr : Address)
    (ha : writes h a addr) : ¬writes h b addr ∧ ¬reads h b addr := by
  constructor
  · intro hb
    rcases ha with ⟨i,ha⟩|ha <;> rcases hb with ⟨j,hb⟩|hb
    · have eq := congrArg Prod.snd (ha.symm.trans hb)
      exact ne (Fin.ext (Geometry.row_injective h a.val b.val i j eq).1)
    · have eq := congrArg Prod.fst (ha.symm.trans hb)
      exact (by decide : ("y":RegionName)≠"r") eq
    · have eq := congrArg Prod.fst (ha.symm.trans hb)
      exact (by decide : ("r":RegionName)≠"y") eq
    · exact ne (Fin.ext (congrArg Prod.snd (ha.symm.trans hb)))
  · intro hr
    rcases ha with ⟨j,rfl⟩|rfl <;> simp [reads] at hr

theorem step_complete (M : Algebra α) (h : Call) (epsilon : α) (row : Fin h.rows) (heap : Heap α) :
    ∃ final,execute M h epsilon row heap=some final ∧
      (∀ i,final (address h row i)=expected M h epsilon heap row i) ∧
      (∀ addr,¬writes h row addr → final addr=heap addr) := by
  obtain ⟨v,he,hy,hr,hf⟩ := Memory.forward_complete M h epsilon (entry heap row.val h.rows)
  refine ⟨fun a => v.mem a.1 a.2,by simp [execute,he],?_,?_⟩
  · intro i
    cases i with
    | inl j => exact hy j
    | inr _ => exact hr
  · rintro ⟨q,o⟩ outside
    apply hf
    rintro (⟨same,j,eq⟩|⟨same,eq⟩)
    · exact outside (Or.inl ⟨j,Prod.ext same eq⟩)
    · exact outside (Or.inr (Prod.ext same eq))

theorem expected_stable (M : Algebra α) (h : Call) (epsilon : α) (row : Fin h.rows) (a b : Heap α)
    (stable : ∀ addr,reads h row addr → a addr=b addr) (i : Output h) :
    expected M h epsilon a row i=expected M h epsilon b row i := by
  have x : FPMemory.raw h (entry a row.val h.rows) "x" h.cols=
      FPMemory.raw h (entry b row.val h.rows) "x" h.cols := by
    funext j
    exact congrArg (Cell.read .real) (stable ("x",row.val*h.cols+j.val) (Or.inl rfl))
  have w : FPMemory.raw h (entry a row.val h.rows) "w" 0=
      FPMemory.raw h (entry b row.val h.rows) "w" 0 := by
    funext j
    exact congrArg (Cell.read .real) (stable ("w",row.val*0+j.val) (Or.inr rfl))
  cases i <;> simp only [expected,x,w]

def run (M : Algebra α) (h : Call) (epsilon : α) := GridRun.run (execute M h epsilon) (List.finRange h.rows)

theorem complete (M : Algebra α) (h : Call) (epsilon : α) (heap : Heap α) :
    ∃ final,run M h epsilon heap=some final ∧
      (∀ row : Fin h.rows,∀ i,final (address h row i)=expected M h epsilon heap row i) ∧
      (∀ addr,(∀ row : Fin h.rows,¬writes h row addr) → final addr=heap addr) := by
  obtain ⟨final,he,ho,hf⟩ := GridRun.complete (execute M h epsilon) (fun _ => Output h) (address h) id
    (expected M h epsilon) (reads h) (writes h) (inside h) (List.finRange h.rows) (List.nodup_finRange _)
    (fun a _ b _ => separate h a b) (fun w _ => step_complete M h epsilon w)
    (fun w _ => expected_stable M h epsilon w) heap
  exact ⟨final,he,fun row => ho row (List.mem_finRange row),fun addr out => hf addr (fun row _ => out row)⟩


#axiomsClean complete
end Grid
namespace Tensor
open KernelTensor KernelStorage FPTensor
open FPGridForward (Heap entry)
def forwardReference (M : Algebra α) (h : Call) (epsilon : α) (logical : RegionName → Nat → α) (i : Fin h.rows) (j : Fin h.cols) :=
  Reference.output M h epsilon (row h logical "x" i) (weight h logical) ⟨j.val,j.isLt.trans_le h.covered⟩
def savedReference (M : Algebra α) (h : Call) (epsilon : α) (logical : RegionName → Nat → α) (i : Fin h.rows) :=
  Reference.savedInv M h epsilon (row h logical "x" i)
theorem forward {β : Type} (M : Algebra α) (h : Call) (epsilon : α) (storage : Placement h .forward)
    (tensors : RegionName → VeriTile.Torch.TensorMetadata) (binding : Binding h .forward storage tensors)
    (heap : Nat → β) (decode : (Fin 4 → β) → Cell α) (other : Cell α)
    (logical : RegionName → Nat → α) (reads : ReadLaw .forward tensors heap decode (Cell.read .real) logical) :
    let initial:=lift storage heap decode other
    ∃ final,Grid.run M h epsilon initial=some final ∧
      (∀ i : Fin h.rows,∀ j : Fin h.cols,final ("y",i.val*h.cols+j.val)=forwardReference M h epsilon logical i j) ∧
      (∀ i : Fin h.rows,final ("r",i.val)=savedReference M h epsilon logical i) ∧
      Frame h .forward initial final := by
  dsimp only
  let initial:=lift storage heap decode other
  have lr:=logical_read h .forward storage tensors binding heap decode (Cell.read .real) logical reads other
  obtain ⟨final,run,values,frame⟩:=Grid.complete M h epsilon initial
  have xr (i : Fin h.rows) : FPReference.padded M h (FPMemory.raw h (entry initial i.val h.rows) "x" h.cols)=
      FPReference.padded M h (row h logical "x" i) := by
    apply padded_agrees
    intro j hj
    apply lr "x" (by simp [inputs])
    simpa [cells] using Native.offset_bound h i.val j.val i.isLt hj
  have wr (i : Fin h.rows) : FPReference.padded M h (FPMemory.raw h (entry initial i.val h.rows) "w" 0)=
      FPReference.padded M h (weight h logical) := by
    apply padded_agrees
    intro j hj
    simpa [FPMemory.raw,entry,weight] using lr "w" (by simp [inputs]) j.val (by simpa [cells] using hj)
  refine ⟨final,run,?_,?_,forward_frame h initial final frame⟩
  · intro i j
    have value:=values i (.inl j)
    simp only [Grid.expected,forwardReference,Reference.output,Reference.normalized,
      Reference.inv,FPReference.input,xr,wr] at value ⊢
    exact value
  · intro i
    have value:=values i (.inr ())
    simp only [Grid.expected,savedReference,Reference.savedInv,Reference.inv,FPReference.input,xr] at value ⊢
    exact value


#axiomsClean forward
end Tensor

end
end VeriTile.Bench.Repairs.FastRmsLayernorm.RuntimeForward
