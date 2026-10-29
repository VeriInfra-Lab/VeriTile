import bench.repairs.tritonbench_g.matmul_triton1.Proofs.IOResults
import bench.repairs.tritonbench_g.matmul_triton1.Proofs.References

/-! Concrete connection between the existing physical Tensor binding and the
new logical IO. This reuses the ordinary Tensor input laws and the same source
execution foundation; it does not assume any output or compiler result. -/
namespace VeriTile.Bench.Repairs.MatmulTriton1.IOTensorBinding
open VeriTile.Triton FP.Structural Revision001 Kernel
open SharedContract
noncomputable section
variable {α : Type} [Inhabited α]

def values (x : Launch) (read : Memory Word) (decode : Word → α) :
    ChannelValues α (MatmulIO.inputs x.parameters) :=
  Fin.cases (fun j => decode (VeriTile.Torch.FP32.logicalValue read x.a x.la
      (j.1.val*columns x.a.metadata+j.2.1.val)))
    (Fin.cases (fun j => decode (VeriTile.Torch.FP32.logicalValue read x.b x.lb
      (j.1.val*columns x.b.metadata+j.2.1.val))) (fun i => Fin.elim0 i))

theorem input_binding (rest : Algebra α) (P : Core.Primitives α)
    (x : Launch) (read : Memory Word) (decode : Word → α) (s : State α)
    (bound : Binding x read decode s) :
    (MatmulIO.io rest P x.parameters).InputBinding (values x read decode) s := by
  intro i
  refine Fin.cases ?_ (Fin.cases ?_ (fun i => Fin.elim0 i)) i
  · intro j _
    change (s.mem x.ra (j.1.val*columns x.a.metadata+j.2.1.val*1)).read .real=
      decode (VeriTile.Torch.FP32.logicalValue read x.a x.la
        (j.1.val*columns x.a.metadata+j.2.1.val))
    rw [Nat.mul_one]
    apply TensorBinding.logical_left x.a x.b x.la x.lb Bool.true x.defaultFP32
      x.admitted x.ra x.rb read decode s bound.inputs
    rw [matrix_numel x.a.metadata x.admitted.rank.1]
    exact FiniteIndex.linear_bound _ _ _ _ j.1.isLt j.2.1.isLt
  · intro j _
    change (s.mem x.rb (j.1.val*columns x.b.metadata+j.2.1.val*1)).read .real=
      decode (VeriTile.Torch.FP32.logicalValue read x.b x.lb
        (j.1.val*columns x.b.metadata+j.2.1.val))
    rw [Nat.mul_one]
    apply TensorBinding.logical_right x.a x.b x.la x.lb Bool.true x.defaultFP32
      x.admitted x.ra x.rb read decode s bound.inputs
    rw [matrix_numel x.b.metadata x.admitted.rank.2,←x.admitted.dimensions]
    exact FiniteIndex.linear_bound _ _ _ _ j.1.isLt j.2.1.isLt

theorem precision_reference (rest : Algebra α) (P : Core.Primitives α)
    (x : Launch) (read : Memory Word) (decode : Word → α) (s : State α)
    (bound : Binding x read decode s) (i : Index x) :
    MatmulIO.precisionTarget P x.parameters (values x read decode) (0 : Fin 1) (i.1,i.2,())=
      Kernel.precisionTarget P x read decode i := by
  exact (IOResults.precision_target rest P x.parameters (values x read decode) s
    (input_binding rest P x read decode s bound) i.1 i.2 (Core.steps x.parameters)).symm.trans
    (Kernel.precision_reference P x read decode s bound i)

theorem real_reference (rest : Algebra ℝ) (x : Launch) (read : Memory Word)
    (decode : Word → ℝ) (s : State ℝ) (bound : Binding x read decode s) (i : Index x) :
    MatmulIO.realTarget x.parameters (values x read decode) (0 : Fin 1) (i.1,i.2,())=
      Kernel.realTarget x read decode i := by
  have input := input_binding rest Core.realPrimitives x read decode s bound
  have raw := Kernel.real_reference x read decode s bound i
  rw [Core.logical_realMatmul _ _ _ _ (by norm_num [Launch.parameters])] at raw
  exact (IOResults.real_target rest x.parameters (values x read decode) s input i.1 i.2).symm.trans raw

#axiomsClean input_binding
#axiomsClean precision_reference
#axiomsClean real_reference
end
end VeriTile.Bench.Repairs.MatmulTriton1.IOTensorBinding
