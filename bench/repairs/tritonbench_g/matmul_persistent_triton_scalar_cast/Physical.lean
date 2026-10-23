import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Execution
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Tensor

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Physical
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.GridExecution (Post)
open Core
noncomputable section
set_option maxHeartbeats 7000000
variable {α : Type} [Inhabited α]

/-- Ordinary region-to-backing interpretation at the pointer origin. -/
structure Bindings (p : Parameters) (b : Buffers) (initial : State α) (entry : Memory α) : Prop where
  inputs : Inputs p b initial entry
  output : ∀ i,readCell p.problem.profile (initial.mem "C" i)=entry (b.C.storage,b.C.offset+i)

/-- Transport the output region to its physical storage. This definition
contains no arithmetic target and preserves every other physical storage. -/
def materialize (p : Parameters) (b : Buffers) (entry : Memory α) (final : State α) : Memory α :=
  fun point => if point.1=b.C.storage ∧ b.C.offset≤point.2 then
    readCell p.problem.profile (final.mem "C" (point.2-b.C.offset)) else entry point

theorem materialize_read (p : Parameters) (b : Buffers) (entry : Memory α)
    (final : State α) (row column : Nat) (strides : Strides p b) :
    materialize p b entry final (b.C.address row column)=
      readCell p.problem.profile (final.mem "C" (p.cm*row+p.cn*column)) := by
  simp [materialize,Tensor.address,strides.cm,strides.cn,Nat.mul_comm,Nat.add_assoc]

theorem materialize_frame (P : Source.Arithmetic α α) (p : Parameters) (b : Buffers)
    (entry : Memory α) (initial final : State α) (strides : Strides p b)
    (bindings : Bindings p b initial entry)
    (post : Post "C" (logicalAddress p) (referenceValue P p b entry) (fun _ => True) initial final) :
    TensorBinding.PhysicalFrame p.problem b entry (materialize p b entry final) := by
  intro point outside
  unfold materialize
  split_ifs with hit
  · rw [post.2 "C" (point.2-b.C.offset) (.inr ?_),bindings.output]
    · congr 1
      exact Prod.ext hit.1.symm (by omega)
    · intro x hx equal
      apply outside x.1.val x.1.isLt x.2.val x.2.isLt
      apply Prod.ext hit.1.symm
      simp only [Tensor.address,logicalAddress,strides.cm,strides.cn] at equal ⊢
      rw [Nat.mul_comm x.1.val,Nat.mul_comm x.2.val]
      omega
  · rfl

theorem materialize_values (P : Source.Arithmetic α α) (p : Parameters) (b : Buffers)
    (entry : Memory α) (initial final : State α) (strides : Strides p b)
    (post : Post "C" (logicalAddress p) (referenceValue P p b entry) (fun _ => True) initial final) :
    ∀ row<p.problem.M,∀ column<p.problem.N,
      materialize p b entry final (b.C.address row column)=Targets.result P p.problem b entry row column := by
  intro row hr column hc
  rw [materialize_read p b entry final row column strides]
  have read := post.1 (⟨row,hr⟩,⟨column,hc⟩) trivial
  change final.mem "C" (p.cm*row+p.cn*column)=_ at read
  rw [read]
  exact read_pack _ _

theorem tensor_read (P : Source.Arithmetic α α) (p : Parameters) (args : TensorBinding.Arguments)
    (entry : Memory α) (initial final : State α) (layout : Layout p.problem args.descriptors)
    (strides : Strides p args.descriptors)
    (post : Post "C" (logicalAddress p) (referenceValue P p args.descriptors entry) (fun _ => True) initial final)
    (row column : Nat) (hr : row<p.problem.M) (hc : column<p.problem.N) :
    TensorBinding.logicalValue (materialize p args.descriptors entry final) args.output
      (row*p.problem.N+column)=Targets.result P p.problem args.descriptors entry row column := by
  rw [← layout.cShape.2,TensorBinding.logical_read _ _ _ args.outputLaws row column
    (layout.cShape.1 ▸ hr) (layout.cShape.2 ▸ hc)]
  exact materialize_values P p args.descriptors entry initial final strides post row hr column hc

#axiomsClean materialize_frame
#axiomsClean materialize_values
#axiomsClean tensor_read
#specNonCircular Bindings avoiding [Targets.result,Source.run,executeGrid]
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Physical
