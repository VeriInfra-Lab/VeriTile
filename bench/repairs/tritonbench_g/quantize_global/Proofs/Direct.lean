import bench.repairs.tritonbench_g.quantize_global.Proofs.InputBinding
import bench.repairs.tritonbench_g.quantize_global.Proofs.Host
import bench.repairs.tritonbench_g.quantize_global.Proofs.Native
import bench.repairs.tritonbench_g.quantize_global.Proofs.Contract
/-! Direct input specialization of the retained complete typed-grid proof.
Only fresh maximum/output windows are initialized. No snapshot is allocated or
written, and each source input comes from the entry state. -/
namespace QuantizeGlobalRepair.Direct
open VeriTile.Triton FP.Structural
noncomputable section

/-- Tensor object returned by an ordinary successful allocation. Its region
is the exact memory window used by the typed source; object identity is kept. -/
structure Tensor where
  object : Nat
  metadata : TensorMetadata
  region : RegionName

structure Allocation (x : TensorMetadata) (X : RegionName) where
  output : Tensor
  maximum : Tensor
  outputShape : output.metadata.shape=x.shape
  maximumShape : maximum.metadata.shape=[1]
  outputType : output.metadata.dtype=.other "int8"
  maximumType : maximum.metadata.dtype=.floating .fp32
  outputDevice : output.metadata.device=x.device
  maximumDevice : maximum.metadata.device=x.device
  separated : X≠output.region ∧ X≠maximum.region ∧ maximum.region≠output.region
  objects : output.object≠maximum.object
  outputSpan : output.metadata.dataPtr.toNat+x.numel≤2^64
  maximumSpan : maximum.metadata.dataPtr.toNat+4≤2^64
  maximumAlignment : maximum.metadata.dataPtr.toNat%4=0
  initialOutput : Nat → BitVec 8

def parameters (x : TensorMetadata) (X : RegionName) (alloc : Allocation x X) : Parameters :=
  ⟨X,alloc.maximum.region,alloc.output.region,x.numel⟩

def prepare (x : TensorMetadata) (X : RegionName) (alloc : Allocation x X)
    (maximum : α) (s : State α) : State α :=
  {s with mem := fun r i =>
    if r=alloc.maximum.region ∧ i=0 then .mk .real maximum
    else if r=alloc.output.region ∧ i<x.numel then .mk .int (alloc.initialOutput i).toInt
    else s.mem r i}

structure Returned (α : Type) where
  state : State α
  output : Tensor
  maximum : Tensor

def hostTree (x : TensorMetadata) (tree : MaxTree) : MaxTree :=
  if x.numel=0 then .empty else tree

def run [Inhabited α] (rest : Algebra α) (a : Scalar α)
    (absolute : α → α) (maxOp : α → α → α) (tree : MaxTree)
    (x : TensorMetadata) (X : RegionName) (finite : Bool) (alloc : Allocation x X)
    (trials : Nat → List Config) (s : State α) : Except InputError (Option (Returned α)) := do
  checkTensorInputs x finite
  let xs := inputValues X s
  let m := (hostTree x tree).eval (a.literal 0) absolute maxOp xs
  return (execute rest a (parameters x X alloc) (schedule x.numel trials)
    (prepare x X alloc m s)).map (fun t => ⟨t,alloc.output,alloc.maximum⟩)

theorem hostTree_covers (x : TensorMetadata) (tree : MaxTree) (covers : tree.Covers x.numel) :
    (hostTree x tree).Covers x.numel := by
  by_cases empty : x.numel=0
  · simp [hostTree,empty,MaxTree.Covers,MaxTree.leaves]
  · simpa [hostTree,empty] using covers

theorem prepared_inputs [Inhabited α] (x : TensorMetadata) (X : RegionName) (alloc : Allocation x X)
    (maximum : α) (s : State α) :
    Inputs (parameters x X alloc) (inputValues X s) maximum (prepare x X alloc maximum s) := by
  constructor
  · intro i hi
    simp [parameters,prepare,inputValues,alloc.separated.1,alloc.separated.2.1]
  · simp [parameters,prepare]

theorem host_execution [Inhabited α] (rest : Algebra α) (a : Scalar α)
    (absolute : α → α) (maxOp : α → α → α) (tree : MaxTree)
    (x : TensorMetadata) (X : RegionName) (finite : Bool) (h : InputContract x finite)
    (alloc : Allocation x X) (trials : Nat → List Config) (s : State α)
    (nonempty : ∀ k,k<segmentCount x.numel → trials k≠[]) :
    ∃ ret,run rest a absolute maxOp tree x X finite alloc trials s=.ok (some ret) ∧
      ret.output=alloc.output ∧ ret.maximum=alloc.maximum ∧
      let xs := inputValues X s
      let m := (hostTree x tree).eval (a.literal 0) absolute maxOp xs
      ret.state.mem=complete (parameters x X alloc) (fun i => scalar a (xs i) m)
        (prepare x X alloc m s).mem := by
  have accepted := (checkTensorInputs_iff x finite).mpr h
  obtain ⟨t,ht,hm⟩ := whole_grid rest a (parameters x X alloc) (inputValues X s)
    ((hostTree x tree).eval (a.literal 0) absolute maxOp (inputValues X s)) trials nonempty
    ⟨alloc.separated.1,alloc.separated.2.2⟩ _ (prepared_inputs x X alloc _ s)
  refine ⟨⟨t,alloc.output,alloc.maximum⟩,?_,rfl,rfl,hm⟩
  change execute rest a (parameters x X alloc) (schedule x.numel trials)
    (prepare x X alloc _ s)=some t at ht
  simp [run,accepted,ht]

theorem full_frame (x : TensorMetadata) (X : RegionName) (alloc : Allocation x X)
    (maximum : α) (values : Nat → BitVec 8) (s : State α) (r : RegionName) (i : Nat)
    (hm : ¬(r=alloc.maximum.region ∧ i=0)) (ho : ¬(r=alloc.output.region ∧ i<x.numel)) :
    complete (parameters x X alloc) values (prepare x X alloc maximum s).mem r i=s.mem r i := by
  simp [complete,parameters,prepare,hm,ho]

theorem input_frame (x : TensorMetadata) (X : RegionName) (alloc : Allocation x X)
    (maximum : α) (values : Nat → BitVec 8) (s : State α) (i : Nat) :
    complete (parameters x X alloc) values (prepare x X alloc maximum s).mem X i=s.mem X i := by
  apply full_frame <;> simp [alloc.separated.1,alloc.separated.2.1]

#axiomsClean host_execution
#axiomsClean input_frame
#axiomsClean full_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end QuantizeGlobalRepair.Direct
