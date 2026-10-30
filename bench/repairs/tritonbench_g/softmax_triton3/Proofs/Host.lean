import bench.repairs.tritonbench_g.softmax_triton3.Proofs.Binding
import bench.repairs.tritonbench_g.softmax_triton1.Proofs.Storage
namespace VeriTile.Bench.Repairs.SoftmaxTriton3.Contiguous
open VeriTile.Triton
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding blockSize numWarps
noncomputable section

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

theorem Allocation.outputContract {S C V} {a : Access S C V} {layout : Layout}
    {x : Object} {before : World S} (allocation : Allocation a layout x before)
    (h : TensorContract x.metadata) : TensorContract allocation.output.metadata := by
  have hn : allocation.output.metadata.numel=x.metadata.numel := congrArg List.prod allocation.shape
  refine ⟨allocation.output.laws.layout,allocation.output.laws.dtype,?_,?_,allocation.contiguous,
    allocation.noNeg,?_,allocation.alignment,?_⟩
  · rw [allocation.device];exact h.cuda
  · rw [allocation.shape];exact h.rank
  · simpa [rows,allocation.shape,hn] using h.grid
  · simpa [cols,allocation.shape,hn] using h.frontend

def invoke {S C V} {a : Access S C V} (memory : Memory a) (evaluate : Evaluator S)
    (x y : Object) (mask : Option Object) (B p : Nat) (s : World S) : Result (World S) Unit :=
  match physicalEvaluate memory evaluate x y mask B p s.program with
  | none => .error "source execution failed" s
  | some t => .ok () ⟨t,s.live⟩

def runWrapper {S C V} (a : Access S C V) (memory : Memory a) (layout : Layout)
    (evaluate : Evaluator S) (x : Object) (mask : Option Object) (dim : Int)
    (before : World S) (allocation : Allocation a layout x before) : Result (World S) Object :=
  match checkTyped x.metadata (mask.map (·.metadata)) dim with
  | .error _ => .error "input contract rejected" before
  | .ok () =>
    if x.metadata.numel=0 then .ok allocation.output allocation.allocated else
      (runRows (invoke memory evaluate x allocation.output mask (blockSize (cols x.metadata)))
        (List.range (rows x.metadata)) allocation.allocated).bind (fun _ t => .ok allocation.output t)

def maskValues {S C V} (a : Access S C V) (neg : V → V) (s : S) (mask : Option Object) (r : Nat) :
    Option (Nat → V) := mask.map (fun m => objectValue a neg s m r)

theorem grid_terminates {S C V} {a : Access S C V} (memory : Memory a) (evaluate : Evaluator S)
    (terminates : ∀ out inp mask rows n B p,0<B → ∀ s,∃ t,evaluate out inp mask rows n B p s=some t)
    (x y : Object) (mask : Option Object) (B : Nat) (positive : 0<B)
    (order : List Nat) (s : World S) :
    ∃ t,runRows (invoke memory evaluate x y mask B) order s=.ok () t := by
  induction order generalizing s with
  | nil => exact ⟨s,rfl⟩
  | cons p ps ih =>
    obtain ⟨u,hu⟩ := terminates "out" "inp" (mask.map (fun _ => "mask"))
      (rows x.metadata) (cols x.metadata) B p positive (rebase memory x y mask s.program)
    obtain ⟨t,ht⟩ := ih ⟨project memory y p s.program u,s.live⟩
    exact ⟨t,by simp [runRows,invoke,physicalEvaluate,hu,Result.bind,ht]⟩

theorem returned_tensor {S C V} (a : Access S C V) (memory : Memory a) (neg : V → V)
    (layout : Layout) (evaluate : Evaluator S) (formula : Formula V) (source : Source a evaluate formula)
    (terminates : ∀ out inp mask rows n B p,0<B → ∀ s,∃ t,evaluate out inp mask rows n B p s=some t)
    (x : Object) (mask : Option Object) (dim : Int) (h : InputContract x.metadata (mask.map (·.metadata)) dim)
    (before : World S) (valid : Valid layout before (descriptor x))
    (maskValid : ∀ m,mask=some m → Valid layout before (descriptor m))
    (allocation : Allocation a layout x before) :
    ∃ final,runWrapper a memory layout evaluate x mask dim before allocation=.ok allocation.output final ∧
      Preserves a before final ∧ Valid layout final (descriptor allocation.output) ∧
      (∀ r c,r<rows x.metadata → c<cols x.metadata →
        objectValue a neg final.program allocation.output r c=
          formula (cols x.metadata) (blockSize (cols x.metadata)) (objectValue a neg before.program x r)
            (maskValues a neg before.program mask r) c) := by
  have accepted := (checkTyped_iff _ _ _).mpr h
  have outRows : rows allocation.output.metadata=rows x.metadata := by simp [rows,allocation.shape]
  have outCols : cols allocation.output.metadata=cols x.metadata := by simp [cols,allocation.shape]
  have outContract := allocation.outputContract h.input
  by_cases empty : x.metadata.numel=0
  · refine ⟨allocation.allocated,by simp [runWrapper,accepted,empty],allocation.frame,allocation.valid,?_⟩
    intro r c hr hc
    have hi := VeriTile.Bench.Repairs.SquareMatrix.flat_lt _ _ _ _ hr hc
    rw [←numel_eq _ h.input.rank,empty] at hi
    omega
  · let xs := objectValue a neg before.program x
    let target := fun r c => formula (cols x.metadata) (blockSize (cols x.metadata)) (xs r)
      (maskValues a neg before.program mask r) c
    let masks := fun w : World S => ∀ m,mask=some m → ∀ r c,r<rows x.metadata → c<cols x.metadata →
      a.read w.program m.region (m.tensor.storageOffset+r*cols x.metadata+c)=objectValue a neg before.program m r c
    have hm : ∀ m,mask=some m → TensorContract m.metadata ∧ cols m.metadata=cols x.metadata ∧
        rows m.metadata=rows x.metadata := by
      intro m eq
      have law := h.mask m.metadata (by simp [eq])
      exact ⟨law.1,law.2.2.1,law.2.2.2⟩
    have input : Inputs (worldAccess a) (descriptor x) xs allocation.allocated := by
      intro r c hr hc
      change a.decode (a.cells allocation.allocated.program (descriptor x).region ((descriptor x).index r c))=_
      rw [allocation.frame.cells _ _ valid.1]
      exact descriptor_read a neg before.program x h.input r c hr hc
    have masksInitial : masks allocation.allocated := by
      intro m eq r c hr hc
      have law := hm m eq
      unfold Access.read
      rw [allocation.frame.cells m.region _ (maskValid m eq).1]
      have read := descriptor_read a neg before.program m law.1 r c
        (by rwa [law.2.2]) (by rwa [law.2.1])
      simpa [Access.read,descriptor,Tensor.index,law.2.1] using read
    have separate : (descriptor x).region≠(descriptor allocation.output).region := by
      intro eq
      apply allocation.fresh
      change before.live (descriptor allocation.output).region
      rw [←eq]
      exact valid.1
    have maskSeparate : ∀ m,mask=some m → m.region≠allocation.output.region := by
      intro m eq same
      apply allocation.fresh
      rw [←same]
      exact (maskValid m eq).1
    have law : RowLaw (worldAccess a) (descriptor x) (descriptor allocation.output) xs target
        (invoke memory evaluate x allocation.output mask (blockSize (cols x.metadata)))
        (fun w => w.live=allocation.allocated.live ∧ masks w) := by
      constructor
      intro p w z hp inputs inv call
      cases he : physicalEvaluate memory evaluate x allocation.output mask (blockSize (cols x.metadata)) p w.program with
      | none => simp [invoke,he] at call
      | some t =>
        have eq : (⟨t,w.live⟩ : World S)=z := by simpa [invoke,he] using call
        subst z
        have src := physical_row a memory evaluate formula source x allocation.output mask allocation.shape
          _ p (block_bounds _).1 (block_bounds _).2 hp w.program t (xs p)
          (maskValues a neg before.program mask p)
          (fun c hc => inputs p c hp hc)
          (by intro m eq c hc;simpa [maskValues,eq] using inv.2 m eq p c hp hc)
          (by cases mask <;> rfl) he
        refine ⟨src.1,src.2,inv.1,?_⟩
        intro m eq r c hr hc
        have frame := src.2 m.region (m.tensor.storageOffset+r*cols x.metadata+c) (by
          intro q hq j hj same
          exact maskSeparate m eq same.1.symm)
        exact (congrArg a.decode frame).trans (inv.2 m eq r c hr hc)
    obtain ⟨final,run⟩ := grid_terminates memory evaluate terminates x allocation.output mask
      (blockSize (cols x.metadata)) (block_bounds _).1 (List.range (rows x.metadata)) allocation.allocated
    have complete := rows_return (worldAccess a) (descriptor x) (descriptor allocation.output) xs target
      _ _ law ⟨outRows,outCols⟩ separate (descriptor_injective _) _
      (fun p hp => List.mem_range.mp hp) allocation.allocated final input ⟨rfl,masksInitial⟩ run
    have frame : Preserves a before final := by
      constructor
      · intro r hr
        rw [complete.2.2.1]
        exact allocation.frame.live r hr
      · intro r i hr
        have off : r≠allocation.output.region := fun eq => allocation.fresh (eq ▸ hr)
        exact (complete.2.1 r i (by intro p hp c hc eq;exact off eq.1.symm)).trans
          (allocation.frame.cells r i hr)
    refine ⟨final,by simp [runWrapper,accepted,empty,run,Result.bind],frame,
      ⟨by rw [complete.2.2.1];exact allocation.valid.1,allocation.valid.2⟩,?_⟩
    intro r c hr hc
    rw [←descriptor_read a neg final.program allocation.output outContract r c
      (by rwa [outRows]) (by rwa [outCols])]
    exact complete.1 r (List.mem_range.mpr hr) c hc

#axiomsClean returned_tensor
#axiomsClean Allocation.outputContract
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxTriton3.Contiguous
