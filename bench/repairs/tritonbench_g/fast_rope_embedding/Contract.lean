import bench.repairs.tritonbench_g.fast_rope_embedding.Kernels
import bench.repairs.tritonbench_g.rope_embedding.Proofs.KernelBinding
import bench.repairs.tritonbench_g.rope_embedding.Proofs.KernelAccess
import bench.repairs.tritonbench_g.rope_embedding.Proofs.RealBinding

namespace VeriTile.Bench.Repairs.FastRopeEmbedding.KernelContract
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding
noncomputable section

abbrev Geometry := Request
abbrev Inputs := KernelBinding.Inputs
abbrev Word := Numerics.Word

/-- Actual prepared Tensor arguments and stride(0) observations at one JIT call.
The Q buffer is updated in place. No wrapper return or allocation execution is
part of this contract. -/
structure Call where
  geometry : Geometry
  inputs : Inputs
  qStride : Nat
  cosineStride : Nat
  sineStride : Nat
  backward : Bool

def Call.parameters (c : Call) : Source.Parameters :=
  ⟨"Q","C","S",c.qStride,c.cosineStride,c.sineStride,c.geometry.sequence,
    c.geometry.dimension,c.geometry.heads,c.geometry.block⟩

def Call.source (c : Call) : ComputeKernel :=
  Kernels.rope "Q" "C" "S" c.qStride c.cosineStride c.sineStride
    c.geometry.sequence c.geometry.dimension c.geometry.heads c.geometry.block c.backward

/-- Ordinary row-index API law for the actual packed reshape. A singleton row
may retain any stride; neither a canonical-stride guard nor a copy is added. -/
def RowStride (c : Call) : Prop :=
  ∀ row col,row<c.geometry.rows → col<c.geometry.heads*c.geometry.dimension →
    c.inputs.q.physicalIndex (row*(c.geometry.heads*c.geometry.dimension)+col)=
      c.inputs.q.storageOffset+row*c.qStride+col

/-- Existing checked metadata, ordinary Tensor indexing, and the source's
packed coefficient-row checks. No result or successful execution is assumed. -/
structure Admitted (c : Call) : Prop where
  metadata : (KernelBinding.metadata c.inputs c.geometry).check=Bool.true
  rowStride : RowStride c
  cosineStride : 0<c.geometry.half → c.cosineStride=c.geometry.half
  sineStride : 0<c.geometry.half → c.sineStride=c.geometry.half

def pair := Spec.pair
def mathematical := Spec.result

theorem pair_adjoint (a b u v cosine sine : ℝ) :
    (pair Bool.false a b cosine sine).1*u+(pair Bool.false a b cosine sine).2*v=
      a*(pair Bool.true u v cosine sine).1+b*(pair Bool.true u v cosine sine).2 :=
  Spec.adjoint a b u v cosine sine

/-- Exact same typed AST as the retained four-head source theorem. -/
theorem source_binding (c : Call) : c.source=Source.source c.parameters c.backward := rfl

theorem q_size (c : Call) (h : Admitted c) : c.inputs.q.metadata.numel=c.geometry.size := by
  have shape := (KernelBinding.admitted_shapes c.inputs c.geometry h.metadata).1
  simp [VeriTile.Torch.TensorMetadata.numel,shape,Request.size,Nat.mul_assoc]

theorem active_stride (c : Call) (h : Admitted c) (row : Nat)
    (hr : row<c.geometry.rows) (positive : 0<c.geometry.heads*c.geometry.dimension) :
    row*c.qStride=row*(c.geometry.heads*c.geometry.dimension) := by
  have bound : row*(c.geometry.heads*c.geometry.dimension)<c.inputs.q.metadata.numel := by
    rw [q_size c h]
    simpa [Request.size,Nat.mul_assoc] using Nat.mul_lt_mul_of_pos_right hr positive
  have contiguous := c.inputs.qLaws.contiguous
    (KernelBinding.payload_facts _ (KernelBinding.admitted_payload c.inputs c.geometry h.metadata).1).1
    _ bound
  have actual := h.rowStride row 0 hr positive
  simp only [Nat.add_zero] at actual
  omega

theorem singleton_stride (c : Call) (h : Admitted c) (rows : c.geometry.rows=1)
    (stride : Nat) :
    ∀ row,row<c.geometry.rows → row*stride=row*(c.geometry.heads*c.geometry.dimension) := by
  intro row hr
  have zero : row=0 := by omega
  simp [zero]

#axiomsClean source_binding
#axiomsClean active_stride
#axiomsClean pair_adjoint
end
end VeriTile.Bench.Repairs.FastRopeEmbedding.KernelContract
