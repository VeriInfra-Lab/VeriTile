import bench.repairs.tritonbench_g.square_matrix.Proofs.Memory

namespace VeriTile.Bench.Repairs.SquareMatrix.Host
open VeriTile.Triton
open VeriTile.Torch.FP32
open VeriTile.Bench.TritonBenchG.AddExample.Finite
open Native
noncomputable section
set_option maxHeartbeats 6000000

def Separate (a b : Storage) : Prop :=
  a.base+4*a.elements ≤ b.base ∨ b.base+4*b.elements ≤ a.base

theorem separate_data (x y : Tensor) (lx : TensorLaws x) (ly : TensorLaws y)
    (cx : x.metadata.isContiguous=Bool.true) (cy : y.metadata.isContiguous=Bool.true)
    (sep : Separate x.storage y.storage) :
    (view (dataView x lx cx)).Disjoint (view (dataView y ly cy)) := by
  by_cases hx : x.metadata.numel=0
  · simp [PhysicalView.Disjoint,view,dataView,DataView.base,hx]
  by_cases hy : y.metadata.numel=0
  · simp [PhysicalView.Disjoint,view,dataView,DataView.base,hy]
  have bx := contiguous_capacity x lx cx (Nat.pos_of_ne_zero hx)
  have by' := contiguous_capacity y ly cy (Nat.pos_of_ne_zero hy)
  simp only [PhysicalView.Disjoint,view,dataView,DataView.base,Storage.address,
    if_neg hx,if_neg hy,Separate] at *
  omega

/-- Ordinary successful empty_like(..., contiguous_format) outcome. There is
no output-value or kernel-success field. Allocation preserves exterior bytes. -/
structure Allocation {M : Type} (ops : MemoryPrimitives M Word)
    (x : Tensor) (before : M) where
  output : Tensor
  laws : TensorLaws output
  allocated : M
  shape : output.metadata.shape=x.metadata.shape
  device : output.metadata.device=x.metadata.device
  contiguous : output.metadata.isContiguous=Bool.true
  noNeg : output.metadata.isNeg=Bool.false
  alignment : 0<output.metadata.numel → output.dataPtr.toNat%4=0
  fresh : Separate x.storage output.storage
  freshObject : output.object≠x.object
  frame : Preserved ops (view (dataView output laws contiguous)) allocated before

theorem Allocation.numel {M : Type} {ops : MemoryPrimitives M Word} {x : Tensor} {before : M}
    (a : Allocation ops x before) : a.output.metadata.numel=x.metadata.numel :=
  congrArg List.prod a.shape

def layout {M : Type} (ops : MemoryPrimitives M Word) (x : Tensor) (lx : TensorLaws x)
    (h : InputContract x.metadata) (before : M) (a : Allocation ops x before) : Layout x.metadata where
  input := dataView x lx h.contiguous
  output := dataView a.output a.laws a.contiguous
  inputSize := rfl
  outputSize := a.numel
  inputAlignment := dataView_aligned x lx h.contiguous h.alignment
  outputAlignment := dataView_aligned a.output a.laws a.contiguous a.alignment
  fresh := separate_data x a.output lx a.laws h.contiguous a.contiguous a.fresh

def runWrapper {M : Type} (ops : MemoryPrimitives M Word) (mul : Word → Word → Word)
    (x : Tensor) (lx : TensorLaws x) (h : InputContract x.metadata)
    (before : M) (a : Allocation ops x before) (schedule : List Nat) :
    Except InputError (Tensor × M) := do
  checkInput (.tensor x.metadata)
  if x.metadata.numel=0 then return (a.output,a.allocated)
  return (a.output,Native.runRows ops mul (layout ops x lx h before a) schedule a.allocated)

def logical {M : Type} (ops : MemoryPrimitives M Word) (s : M)
    (t : Tensor) (laws : TensorLaws t) (i : Nat) : Word :=
  logicalValue (ops.read s .fp32) t laws i

theorem returned_tensor {M : Type} (ops : MemoryPrimitives M Word) (mul : Word → Word → Word)
    (x : Tensor) (lx : TensorLaws x) (h : InputContract x.metadata)
    (before : M) (a : Allocation ops x before)
    (memory : LocalMemory ops (layout ops x lx h before a))
    (schedule : List Nat) (complete : schedule.Perm (List.range (rows x.metadata))) :
    ∃ final, runWrapper ops mul x lx h before a schedule=.ok (a.output,final) ∧
      (∀ i,i<x.metadata.numel → logical ops final a.output a.laws i=
        mul (logical ops before x lx i) (logical ops before x lx i)) ∧
      Preserved ops (view (dataView a.output a.laws a.contiguous)) final before := by
  let L := layout ops x lx h before a
  have accepted : checkInput (.tensor x.metadata)=.ok () := (checkTensorInput_iff x.metadata).mpr h
  by_cases empty : x.metadata.numel=0
  · exact ⟨a.allocated,by simp [runWrapper,accepted,empty],by simp [empty],a.frame⟩
  · obtain ⟨values,frame⟩ := launch_correct ops mul x.metadata h L memory schedule complete (by omega) a.allocated
    refine ⟨Native.runRows ops mul L schedule a.allocated,by simp [runWrapper,accepted,empty,L],?_,
      preserved_trans ops _ _ _ _ frame a.frame⟩
    intro i hi
    unfold logical
    rw [←direct_read_eq_logical (ops.read _ .fp32) a.output a.laws a.contiguous a.noNeg i
      (by rw [a.numel]; exact hi)]
    change ops.read _ .fp32 (address L.output i)=_
    rw [values i hi]
    have same := input_preserved ops (view L.input) (view L.output) a.allocated before a.frame L.fresh _
      (inside x.metadata L.input L.inputSize i hi)
    simp only [view] at same
    unfold result
    rw [same]
    have input := direct_read_eq_logical (ops.read before .fp32) x lx h.contiguous h.noNeg i hi
    exact congrArg (fun v => mul v v) input

#axiomsClean returned_tensor
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SquareMatrix.Host
