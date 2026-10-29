import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.TensorBinding
namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.RealTensor
open VeriTile.Triton FP.Structural Checked TensorBinding
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
abbrev RealHeap := Nat → Nat → ℝ

/-- Ideal-real scalar interpretation of the same half Tensor allocations.
Packed int32 values are decoded from the caller's actual raw bytes. -/
def entry (p : Metadata) (raw : Heap) (heap : RealHeap) : State ℝ where
  mem := fun r a => if r="Q" ∨ r="Z" then .mk .int
    (Physical.decode32 (fun i => memory p raw r (4*a+i.val))).toInt
    else .mk .fp16 (heap (observation p r).storageObject ((observation p r).storageOffset+a))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d

def physical (p : Metadata) (heap : RealHeap) (state : State ℝ) : RealHeap := fun r a =>
  if r=p.output.storageObject ∧ p.output.storageOffset≤a then
    (state.mem "C" (a-p.output.storageOffset)).read .fp16 else heap r a

structure Result (p : Metadata) where
  memory : RealHeap
  output : Tensor p.output p.call.C
  returned : Option (Tensor p.output p.call.C)
  allocated : Bool

def invoke (p : Metadata) (t : Tensors p) (i : Fin 17) (raw : Heap) (heap : RealHeap) :
    Except Error (Option (Result p)) := do
  checkInputs p
  pure ((RealHost.grid p.call i (entry p raw heap)).map fun state =>
    ⟨physical p heap state,t.output,if p.call.supplied then none else some t.output,!p.call.supplied⟩)

def Guarantee (p : Metadata) (t : Tensors p) (i : Fin 17) (raw : Heap) (heap : RealHeap) (result : Result p) :=
  result.output=t.output ∧ result.returned=(if p.call.supplied then none else some t.output) ∧
  result.allocated=(!p.call.supplied) ∧ InputContract p ∧
  (∀ x : Grid.Index (params p.call i),
    result.memory p.output.storageObject
      (p.output.storageOffset+Geometry.address (Grid.geometry (params p.call i)) x)=
        Real.mathematical p.call (RealHost.input p.call (entry p raw heap)) x.1.val x.2.val) ∧
  (∀ r a,(r≠p.output.storageObject ∨ a<p.output.storageOffset ∨
      ∀ x : Grid.Index (params p.call i),a-p.output.storageOffset≠Geometry.address (Grid.geometry (params p.call i)) x) →
    result.memory r a=heap r a)

theorem whole_tensor (p : Metadata) (t : Tensors p) (h : InputContract p) (i : Fin 17)
    (raw : Heap) (heap : RealHeap) :
    ∃ result,invoke p t i raw heap=.ok (some result) ∧ Guarantee p t i raw heap result := by
  obtain ⟨state,he,values,frame⟩ := RealHost.whole_grid p.call (admitted p h).valid i (entry p raw heap)
  refine ⟨⟨physical p heap state,t.output,if p.call.supplied then none else some t.output,!p.call.supplied⟩,
    ?_,rfl,rfl,rfl,h,?_,?_⟩
  · simp [invoke,(checkInputs_iff p).mpr h,he]
  · intro x
    have hv := congrArg (fun c : Cell ℝ => c.read .fp16) (values x trivial)
    simpa [physical,Cell.read_same] using hv
  · intro r a outside
    by_cases hw : r=p.output.storageObject ∧ p.output.storageOffset≤a
    · have ho : ∀ x : Grid.Index (params p.call i),a-p.output.storageOffset≠Geometry.address (Grid.geometry (params p.call i)) x := by
        rcases outside with hr|ha|ho
        · exact False.elim (hr hw.1)
        · omega
        · exact ho
      have hf := congrArg (fun c : Cell ℝ => c.read .fp16) (frame "C" (a-p.output.storageOffset) (Or.inr (fun x _ => ho x)))
      simp only [physical,hw,true_and,ite_true]
      change (state.mem "C" (a-p.output.storageOffset)).read .fp16=_
      dsimp only at hf
      rw [hf]
      simp only [entry,observation,show ("C":RegionName)≠"A" by decide,
        show ("C":RegionName)≠"Q" by decide,show ("C":RegionName)≠"S" by decide,
        show ("C":RegionName)≠"Z" by decide,false_or,ite_false,Cell.read_same]
      congr 1
      omega
    · simp only [physical,hw,ite_false]

#specNonCircular Guarantee avoiding [invoke,RealHost.grid,RealHost.stored,Real.value]
#axiomsClean whole_tensor
end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.RealTensor
