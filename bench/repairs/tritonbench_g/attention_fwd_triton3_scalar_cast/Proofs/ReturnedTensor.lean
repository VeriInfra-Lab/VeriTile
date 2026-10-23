import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.TensorGeometry

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.TensorReturn
open VeriTile Triton
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open Host TensorBinding
noncomputable section

structure Tuple where
  output : Torch.FP32.Tensor
  center : Torch.FP32.Tensor
  denominator : Torch.FP32.Tensor

/-- Storage-level API law for a bit-preserving slice/contiguous/FP32-to-FP32
copy. The source and destination are actual shared Tensor objects. This law
contains no attention target, execution state, floating primitive, or result. -/
structure Copy (before after : Nat → Torch.FP32.Word)
    (source destination : Torch.FP32.Tensor) (indexMap : Nat → Nat) : Prop where
  sourceLaws : Torch.FP32.TensorLaws source
  destinationLaws : Torch.FP32.TensorLaws destination
  sourceResolved : source.metadata.isNeg=Bool.false
  destinationResolved : destination.metadata.isNeg=Bool.false
  sourceBound : ∀ i,i<destination.metadata.numel → indexMap i<source.metadata.numel
  bytes : ∀ i,i<destination.metadata.numel →
    after (destination.storage.address (destination.physicalIndex i))=
      before (source.storage.address (source.physicalIndex (indexMap i)))

theorem Copy.logical (before after : Nat → Torch.FP32.Word)
    (source destination : Torch.FP32.Tensor) (indexMap : Nat → Nat)
    (copy : Copy before after source destination indexMap) (i : Nat)
    (bound : i<destination.metadata.numel) :
    Torch.FP32.logicalValue after destination copy.destinationLaws i=
      Torch.FP32.logicalValue before source copy.sourceLaws (indexMap i) := by
  simp only [Torch.FP32.logicalValue,copy.sourceResolved,copy.destinationResolved,
    Bool.false_eq_true,if_false]
  exact copy.bytes i bound

/-- Linear index of the public END slice in the supplied rounded workspace. -/
def outputMap (c : Call) (i : Nat) : Nat :=
  if c.finish=0 then i else
    i/(c.NQ*c.D)*(c.rounded*c.D)+i%(c.NQ*c.D)

theorem slice_index (c : Call) (z h row dim : Nat)
    (finish : c.finish≠0) (hr : row<c.NQ) (hd : dim<c.D) :
    outputMap c (index4 c.H c.NQ c.D z h row dim)=index4 c.H c.rounded c.D z h row dim := by
  have tail := join_lt _ _ _ _ hr hd
  have positive : 0<c.NQ*c.D := by omega
  have regroup : index4 c.H c.NQ c.D z h row dim=
      (z*c.H+h)*(c.NQ*c.D)+(row*c.D+dim) := by unfold index4; ring
  rw [outputMap,if_neg finish,regroup]
  simp only [Nat.mul_comm (z*c.H+h) (c.NQ*c.D),Nat.mul_add_div positive,
    Nat.div_eq_of_lt tail,Nat.add_zero,Nat.mul_add_mod,Nat.mod_eq_of_lt tail,index4]
  ring

/-- The ordinary wrapper API: M/L retain their Tensor identity; unfinalized
O retains its identity; finalized O has the sliced shape and is contiguous.
Copy obligations concern storage words only, independently of their origin. -/
structure API (c : Call) (r : Request) (before after : Nat → Torch.FP32.Word)
    (returned : Tuple) : Prop where
  centerIdentity : returned.center=(r .center).tensor
  denominatorIdentity : returned.denominator=(r .denominator).tensor
  outputIdentity : c.finish=0 → returned.output=(r .output).tensor
  outputShape : returned.output.metadata.shape=
    [c.Z,c.H,if c.finish=0 then c.rounded else c.NQ,c.D]
  outputContiguous : c.finish≠0 → returned.output.metadata.isContiguous=Bool.true
  outputCopy : Copy before after (r .output).tensor returned.output (outputMap c)
  centerCopy : Copy before after (r .center).tensor returned.center id
  denominatorCopy : Copy before after (r .denominator).tensor returned.denominator id

theorem returned_values (c : Call) (r : Request) (before after : Nat → Torch.FP32.Word)
    (returned : Tuple) (api : API c r before after returned) :
    (∀ i,i<returned.output.metadata.numel →
      Torch.FP32.logicalValue after returned.output api.outputCopy.destinationLaws i=
        Torch.FP32.logicalValue before (r .output).tensor api.outputCopy.sourceLaws (outputMap c i)) ∧
    (∀ i,i<returned.center.metadata.numel →
      Torch.FP32.logicalValue after returned.center api.centerCopy.destinationLaws i=
        Torch.FP32.logicalValue before (r .center).tensor api.centerCopy.sourceLaws i) ∧
    (∀ i,i<returned.denominator.metadata.numel →
      Torch.FP32.logicalValue after returned.denominator api.denominatorCopy.destinationLaws i=
        Torch.FP32.logicalValue before (r .denominator).tensor api.denominatorCopy.sourceLaws i) := by
  exact ⟨api.outputCopy.logical _ _ _ _ _,api.centerCopy.logical _ _ _ _ _,
    api.denominatorCopy.logical _ _ _ _ _⟩

theorem output_coordinate (c : Call) (r : Request) (before after : Nat → Torch.FP32.Word)
    (returned : Tuple) (api : API c r before after returned) (z h row dim : Nat)
    (hz : z<c.Z) (hh : h<c.H) (hr : row<c.NQ) (hd : dim<c.D) :
    Torch.FP32.logicalValue after returned.output api.outputCopy.destinationLaws
      (index4 c.H (if c.finish=0 then c.rounded else c.NQ) c.D z h row dim)=
    Torch.FP32.logicalValue before (r .output).tensor api.outputCopy.sourceLaws
      (index4 c.H c.rounded c.D z h row dim) := by
  have bound : index4 c.H (if c.finish=0 then c.rounded else c.NQ) c.D z h row dim<
      returned.output.metadata.numel := by
    rw [Torch.TensorMetadata.numel,api.outputShape]
    simp only [List.prod_cons,List.prod_nil,Nat.mul_one]
    apply index4_bound _ _ _ _ _ _ _ _ hz hh _ hd
    split_ifs
    · exact query_within_workspace c row hr
    · exact hr
  rw [(returned_values c r before after returned api).1 _ bound]
  by_cases finish : c.finish=0
  · simp [outputMap,finish]
  · simp only [finish,if_false]
    rw [slice_index c z h row dim finish hr hd]

#axiomsClean returned_values
#axiomsClean output_coordinate
#specNonCircular Copy avoiding [FP.Structural.exec,FP.Structural.run,Raw.output,Whole.result]
#specNonCircular API avoiding [FP.Structural.exec,FP.Structural.run,Raw.output,Whole.result]
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.TensorReturn
