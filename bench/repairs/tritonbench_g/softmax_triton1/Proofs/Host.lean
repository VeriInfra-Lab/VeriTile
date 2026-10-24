import bench.repairs.tritonbench_g.softmax_triton1.Proofs.InputBinding
import bench.repairs.tritonbench_g.softmax_triton1.Proofs.Wrapper
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.SoftmaxTriton1.Contiguous
open VeriTile.Triton
noncomputable section

theorem descriptor_read {S C V} (a : Access S C V) (neg : V → V)
    (s : World S) (x : Object) (rank : x.metadata.shape.length=2)
    (contiguous : x.metadata.isContiguous=Bool.true)
    (r c : Nat) (hr : r<rows x.metadata) (hc : c<cols x.metadata) :
    logical a neg s (descriptor x) r c=objectValue a neg s.program x r c := by
  have hs : x.metadata.shape=[rows x.metadata,cols x.metadata] := by
    obtain ⟨m,n,eq⟩ := List.length_eq_two.mp rank
    simp [rows,cols,eq]
  have hi : r*cols x.metadata+c<x.metadata.numel := by
    simpa [VeriTile.Torch.TensorMetadata.numel,hs] using
      VeriTile.Bench.Repairs.SquareMatrix.flat_lt _ _ _ _ hr hc
  simp [logical,descriptor,Tensor.index,objectValue,x.laws.contiguous contiguous _ hi,
    Nat.add_assoc]

/-- Successful contiguous empty_like allocation. These fields state only
Tensor metadata, freshness and the allocation frame, never kernel values. -/
structure Allocation {S C V} (a : Access S C V) (layout : Layout)
    (x : Object) (before : World S) where
  output : Object
  allocated : World S
  shape : output.metadata.shape=x.metadata.shape
  device : output.metadata.device=x.metadata.device
  contiguous : output.metadata.isContiguous=Bool.true
  noNeg : output.metadata.isNeg=Bool.false
  alignment : 0<output.metadata.numel → output.metadata.dataPtr.toNat%4=0
  valid : Valid layout allocated (descriptor output)
  fresh : ¬before.live output.region
  freshObject : output.tensor.object≠x.tensor.object
  frame : Preserves a before allocated

theorem Allocation.sameShape {S C V} {a : Access S C V} {layout : Layout}
    {x : Object} {before : World S} (allocation : Allocation a layout x before) :
    SameShape (descriptor x) (descriptor allocation.output) := by
  constructor
  · change rows allocation.output.metadata=rows x.metadata
    simp [rows,allocation.shape]
  · change cols allocation.output.metadata=cols x.metadata
    simp [cols,allocation.shape]
  · exact congrArg VeriTile.Torch.Device.index allocation.device

def invoke {S} (evaluate : Tensor → Tensor → Nat → Nat → S → Option S)
    (x y : Tensor) (B p : Nat) (s : World S) : Result (World S) Unit :=
  match evaluate x y B p s.program with
  | none => .error "source execution failed" s
  | some t => .ok () ⟨t,s.live⟩

theorem grid_terminates {S} (evaluate : Tensor → Tensor → Nat → Nat → S → Option S)
    (terminates : ∀ x y B p,0<B → ∀ s,∃ t,evaluate x y B p s=some t)
    (x y : Tensor) (B : Nat) (positive : 0<B) (order : List Nat) (s : World S) :
    ∃ t,runRows (invoke evaluate x y B) order s=.ok () t := by
  induction order generalizing s with
  | nil => exact ⟨s,rfl⟩
  | cons p ps ih =>
    obtain ⟨u,hu⟩ := terminates x y B p positive s.program
    obtain ⟨t,ht⟩ := ih ⟨u,s.live⟩
    exact ⟨t,by simp [runRows,invoke,hu,Result.bind,ht]⟩

/-- Direct guarded host interpretation. The original input descriptor is
passed to every row. Allocation is supplied by its ordinary successful API
outcome; an empty matrix returns that exact object without source execution. -/
def runWrapper {S C V} (a : Access S C V) (layout : Layout)
    (evaluate : Tensor → Tensor → Nat → Nat → S → Option S)
    (x : Object) (before : World S) (allocation : Allocation a layout x before) :
    Result (World S) Object :=
  match checkInput (.tensor x.metadata) with
  | .error _ => .error "input contract rejected" before
  | .ok () =>
    if x.metadata.numel=0 then .ok allocation.output allocation.allocated else
      (runRows (invoke evaluate (descriptor x) (descriptor allocation.output)
        (blockSize (cols x.metadata))) (List.range (rows x.metadata)) allocation.allocated).bind
        (fun _ t => .ok allocation.output t)

theorem returned_tensor {S C V} (a : Access S C V) (neg : V → V) (layout : Layout)
    (evaluate : Tensor → Tensor → Nat → Nat → S → Option S)
    (formula : Nat → Nat → (Nat → V) → Nat → V)
    (source : SourceContract a evaluate formula)
    (congruent : ∀ n B xs ys c,c<n → (∀ j,j<n → xs j=ys j) →
      formula n B xs c=formula n B ys c)
    (terminates : ∀ x y B p,0<B → ∀ s,∃ t,evaluate x y B p s=some t)
    (x : Object) (h : InputContract x.metadata) (before : World S)
    (valid : Valid layout before (descriptor x)) (allocation : Allocation a layout x before) :
    ∃ final,runWrapper a layout evaluate x before allocation=.ok allocation.output final ∧
      Returned a neg layout formula (descriptor x) (descriptor allocation.output) before final ∧
      (∀ r c,r<rows x.metadata → c<cols x.metadata →
        objectValue a neg final.program allocation.output r c=
          formula (cols x.metadata) (blockSize (cols x.metadata))
            (objectValue a neg before.program x r) c) := by
  have shape := allocation.sameShape
  have accepted : checkInput (.tensor x.metadata)=.ok () := (checkTensorInput_iff _).mpr h
  have outRank : allocation.output.metadata.shape.length=2 := by rw [allocation.shape];exact h.rank
  have transport : ∀ (final : World S),
      Returned a neg layout formula (descriptor x) (descriptor allocation.output) before final →
      ∀ r c,r<rows x.metadata → c<cols x.metadata →
        objectValue a neg final.program allocation.output r c=
          formula (cols x.metadata) (blockSize (cols x.metadata))
            (objectValue a neg before.program x r) c := by
    intro final result r c hr hc
    rw [←descriptor_read a neg final allocation.output outRank allocation.contiguous r c
      (by simpa [rows,allocation.shape] using hr) (by simpa [cols,allocation.shape] using hc)]
    rw [result.values r c hr hc]
    exact congruent _ _ _ _ c hc (fun j hj => descriptor_read a neg before x h.rank h.contiguous r j hr hj)
  by_cases empty : x.metadata.numel=0
  · have result : Returned a neg layout formula (descriptor x) (descriptor allocation.output)
        before allocation.allocated := by
      refine ⟨shape,allocation.valid,allocation.noNeg,descriptor_injective _,allocation.fresh,
        allocation.frame,?_⟩
      intro r c hr hc
      have hi := VeriTile.Bench.Repairs.SquareMatrix.flat_lt _ _ _ _ hr hc
      change r*cols x.metadata+c<rows x.metadata*cols x.metadata at hi
      rw [←numel_eq x.metadata h,empty] at hi
      omega
    exact ⟨allocation.allocated,by simp [runWrapper,accepted,empty],result,transport _ result⟩
  · obtain ⟨final,run⟩ := grid_terminates evaluate terminates (descriptor x)
      (descriptor allocation.output) (blockSize (cols x.metadata)) (blockSize_bounds _).1 (List.range (rows x.metadata)) allocation.allocated
    let values := logical a neg before (descriptor x)
    have input : Inputs (worldAccess a) (descriptor x) values allocation.allocated := by
      intro r c hr hc
      change a.decode (a.cells allocation.allocated.program (descriptor x).region ((descriptor x).index r c))=_
      rw [allocation.frame.cells _ _ valid.1]
      simp [values,logical,descriptor,h.noNeg,Access.read]
    have separate : (descriptor x).region≠(descriptor allocation.output).region := by
      intro eq
      apply allocation.fresh
      change before.live (descriptor allocation.output).region
      rw [←eq]
      exact valid.1
    have law : RowLaw (worldAccess a) (descriptor x) (descriptor allocation.output) values
        (fun r c => formula (cols x.metadata) (blockSize (cols x.metadata)) (values r) c)
        (invoke evaluate (descriptor x) (descriptor allocation.output) (blockSize (cols x.metadata)))
        (fun w => w.live=allocation.allocated.live) := by
      constructor
      intro p w z hp inputs inv call
      cases he : evaluate (descriptor x) (descriptor allocation.output) (blockSize (cols x.metadata)) p w.program with
      | none => simp [invoke,he] at call
      | some t =>
        have eq : (⟨t,w.live⟩ : World S)=z := by simpa [invoke,he] using call
        subst z
        have src := source.row _ _ _ _ w.program t values shape (descriptor_injective _)
          (blockSize_bounds _).1 (blockSize_bounds _).2 hp inputs he
        exact ⟨src.1,src.2,inv⟩
    have complete := rows_return (worldAccess a) (descriptor x) (descriptor allocation.output)
      values (fun r c => formula (cols x.metadata) (blockSize (cols x.metadata)) (values r) c)
      _ _ law ⟨shape.rows,shape.cols⟩ separate (descriptor_injective _) _
      (fun p hp => List.mem_range.mp hp) allocation.allocated final input rfl run
    have frame : Preserves a before final := by
      constructor
      · intro r hr
        rw [complete.2.2]
        exact allocation.frame.live r hr
      · intro r i hr
        have off : r≠allocation.output.region := fun eq => allocation.fresh (eq ▸ hr)
        exact (complete.2.1 r i (by intro p hp c hc eq;exact off eq.1.symm)).trans
          (allocation.frame.cells r i hr)
    have result : Returned a neg layout formula (descriptor x) (descriptor allocation.output) before final := by
      refine ⟨shape,⟨by rw [complete.2.2];exact allocation.valid.1,allocation.valid.2⟩,
        allocation.noNeg,descriptor_injective _,allocation.fresh,frame,?_⟩
      intro r c hr hc
      have value := complete.1 r (List.mem_range.mpr hr) c hc
      simpa [logical,descriptor,allocation.noNeg,worldAccess,Access.read,values] using value
    exact ⟨final,by simp [runWrapper,accepted,empty,run,Result.bind],result,transport _ result⟩
/-- The returned allocation has the same checked domain. Shape controls the
launch bounds; the allocator separately supplies alignment and normal layout. -/
theorem Allocation.outputContract {S C V} {a : Access S C V} {layout : Layout}
    {x : Object} {before : World S} (allocation : Allocation a layout x before)
    (h : InputContract x.metadata) : InputContract allocation.output.metadata := by
  have hn : allocation.output.metadata.numel=x.metadata.numel := congrArg List.prod allocation.shape
  refine ⟨allocation.output.laws.layout,allocation.output.laws.dtype,?_,?_,allocation.contiguous,
    allocation.noNeg,?_,allocation.alignment,?_⟩
  · rw [allocation.device];exact h.cuda
  · rw [allocation.shape];exact h.rank
  · simpa [rows,allocation.shape,hn] using h.grid
  · simpa [cols,allocation.shape,hn] using h.frontend

#axiomsClean descriptor_read
#axiomsClean returned_tensor
#axiomsClean Allocation.outputContract
#guard_msgs (drop info) in
#auditModuleAxioms

end
end VeriTile.Bench.Repairs.SoftmaxTriton1.Contiguous
