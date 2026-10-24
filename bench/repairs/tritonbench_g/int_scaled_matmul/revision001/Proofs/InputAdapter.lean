import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.TensorBinding
import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.FiniteGrid

namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.InputAdapter
open VeriTile.Triton FP.Structural VeriTile.Torch
open Contract TensorBinding

def launch (a b : TensorBinding.Tensor) (p : Contract.Config) : FiniteExecution.Launch :=
  ⟨rows a.metadata,columns b.metadata,columns a.metadata,p,inputA,inputB,outputC⟩
abbrev Index (a b : TensorBinding.Tensor) := Fin (rows a.metadata) × Fin (columns b.metadata)
def address (a b : TensorBinding.Tensor) (i : Index a b) := i.1.val*columns b.metadata+i.2.val

theorem member_a (scaled : Bool) (a b c s : TensorMetadata) : a∈tensors scaled a b c s := by
  cases scaled <;> simp [tensors]
theorem member_b (scaled : Bool) (a b c s : TensorMetadata) : b∈tensors scaled a b c s := by
  cases scaled <;> simp [tensors]
theorem member_c (scaled : Bool) (a b c s : TensorMetadata) : c∈tensors scaled a b c s := by
  cases scaled <;> simp [tensors]
theorem member_scale (a b c s : TensorMetadata) : s∈tensors Bool.true a b c s := by simp [tensors]

theorem address_injective (a b : TensorBinding.Tensor) : Function.Injective (address a b) := by
  intro i j he
  have hr : i.1.val=j.1.val := by
    by_contra h
    rcases lt_or_gt_of_ne h with lt | gt
    · have hm := Nat.mul_le_mul_right (columns b.metadata) (show i.1.val + 1 ≤ j.1.val by omega)
      have hi := i.2.isLt
      unfold address at he
      nlinarith
    · have hm := Nat.mul_le_mul_right (columns b.metadata) (show j.1.val + 1 ≤ i.1.val by omega)
      have hj := j.2.isLt
      unfold address at he
      nlinarith
  apply Prod.ext
  · exact Fin.ext hr
  · apply Fin.ext
    unfold address at he
    rw [hr] at he
    omega

theorem int_geometry (a b : TensorBinding.Tensor) (p : Contract.Config) (legal : ConfigContract p) :
    Geometry.Valid (BlockWord.geometry (FiniteExecution.blockParameters (launch a b p))) := by
  obtain ⟨hm,hn,_,hg⟩ := positive_tiles p legal
  refine ⟨hm,hn,hg,?_⟩
  intro i j he
  apply address_injective a b
  simpa only [Geometry.address,BlockWord.geometry,FiniteExecution.blockParameters,launch,address,Nat.mul_one]
    using he

theorem scaled_geometry (a b : TensorBinding.Tensor) (p : Contract.Config) (legal : ConfigContract p) :
    Geometry.Valid (ScaledWord.geometry (FiniteExecution.scaledParameters (launch a b p) scales)) :=
  int_geometry a b p legal

theorem linear_bound (r c M N : Nat) (hr : r<M) (hc : c<N) : r*N+c<M*N := by
  have h := Nat.mul_le_mul_right N (show r+1≤M by omega)
  nlinarith

theorem output_metadata (scaled : Bool) (a b c s : TensorBinding.Tensor) (p : Contract.Config)
    (h : InputContract scaled a.metadata b.metadata c.metadata s.metadata p) :
    c.metadata.shape=[rows a.metadata,columns b.metadata] ∧ c.metadata.dtype=.other "int32" ∧
    c.metadata.device=a.metadata.device ∧ c.metadata.device.kind="cuda" ∧
    c.metadata.isContiguous=Bool.true ∧ c.metadata.isNeg=Bool.false := by
  have member := member_c scaled a.metadata b.metadata c.metadata s.metadata
  exact ⟨h.shapes.2.1,h.dtypes.2.2.1,h.device _ member,h.cuda _ member,
    h.contiguous _ member,h.resolved _ member⟩

#axiomsClean address_injective
#axiomsClean int_geometry
#axiomsClean output_metadata
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.InputAdapter
