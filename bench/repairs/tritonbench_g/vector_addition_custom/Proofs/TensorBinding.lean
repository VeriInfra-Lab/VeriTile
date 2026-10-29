import bench.repairs.tritonbench_g.vector_addition_custom.Proofs.Execution

/-! Specialize the shared Torch FP32 Tensor semantics to this kernel's memory
reader and checked input contract. No Tensor API law is duplicated here. -/
namespace VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous.Binding
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AddExample.Finite
open Native
noncomputable section

abbrev Word := FP.Value .fp32
export VeriTile.Torch.FP32 (Tensor TensorLaws dataView)

def logicalValue {M : Type} (ops : MemoryPrimitives M Word) (s : M)
    (t : Tensor) (laws : TensorLaws t) (i : Nat) : Word :=
  VeriTile.Torch.FP32.logicalValue (ops.read s .fp32) t laws i

theorem direct_read_eq_logical {M : Type} (ops : MemoryPrimitives M Word)
    (t : Tensor) (laws : TensorLaws t) (s : M)
    (contiguous : t.metadata.isContiguous = Bool.true)
    (notNeg : t.metadata.isNeg = Bool.false) (i : Nat) (hi : i < t.metadata.numel) :
    ops.read s .fp32 (address (dataView t laws contiguous) i) = logicalValue ops s t laws i :=
  VeriTile.Torch.FP32.direct_read_eq_logical (ops.read s .fp32) t laws contiguous notNeg i hi

theorem checked_read_x {M : Type} (ops : MemoryPrimitives M Word)
    (x y : Tensor) (laws : TensorLaws x) (h : InputContract x.metadata y.metadata)
    (s : M) (i : Nat) (hi : i < x.metadata.numel) :
    ops.read s .fp32 (address (dataView x laws h.contiguous.1) i) =
      logicalValue ops s x laws i :=
  direct_read_eq_logical ops x laws s h.contiguous.1 h.noNeg.1 i hi

theorem checked_read_y {M : Type} (ops : MemoryPrimitives M Word)
    (x y : Tensor) (laws : TensorLaws y) (h : InputContract x.metadata y.metadata)
    (s : M) (i : Nat) (hi : i < x.metadata.numel) :
    ops.read s .fp32 (address (dataView y laws h.contiguous.2) i) =
      logicalValue ops s y laws i :=
  direct_read_eq_logical ops y laws s h.contiguous.2 h.noNeg.2 i
    (by have hn := same_numel x.metadata y.metadata h
        change x.metadata.numel = y.metadata.numel at hn
        rw [←hn]; exact hi)

/-- Native launch receives the actual data_ptr, including a nonzero offset. -/
theorem actual_pointer (t : Tensor) (laws : TensorLaws t)
    (contiguous : t.metadata.isContiguous = Bool.true) (nonempty : 0 < t.metadata.numel) :
    (view (dataView t laws contiguous) 0).pointer = t.dataPtr :=
  VeriTile.Torch.FP32.actual_pointer t laws contiguous nonempty

#axiomsClean direct_read_eq_logical
#axiomsClean checked_read_x
#axiomsClean checked_read_y
#axiomsClean actual_pointer
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous.Binding
