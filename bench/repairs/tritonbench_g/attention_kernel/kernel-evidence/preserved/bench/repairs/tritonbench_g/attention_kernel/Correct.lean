import bench.repairs.tritonbench_g.attention_kernel.Kernels
import bench.repairs.tritonbench_g.attention_kernel.Proofs.Contract
import bench.repairs.tritonbench_g.attention_kernel.Proofs.AcceptedReal
namespace VeriTile.Bench.Repairs.AttentionKernel.Correct
open VeriTile Triton FP.Structural
open Whole001 Source Whole Machine
noncomputable section
abbrev target := RealReturned.target
specification attention_real (p : Parameters) (bases : Bases)
    (args : Contract.Arguments) (admitted : Contract.InputContract p bases args) (before : RealReturned.InputValues) (storage : TensorStorage p)
    (fits : storage.capacity=before.capacity)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    Spec.Real (Contract.check p bases args=.ok () ∧ ∃ result,Physical.invoke (fun _=>Real.algebra) (fun _=>Real.dotModel) p (Contract.abi p args admitted.1) (before.store p) work=some result ∧
      result.result=() ∧ result.memory.capacity=before.capacity ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<p.dim),
        ∃ index : Fin (result.memory.capacity p.Out),index.val=logicalAddress p b h n d ∧
          result.memory.cells p.Out index=.mk p.storage.tile
            (ofFloat p.storage.float (some (RealReturned.target p before.logical b h n d)))) ∧
      (∀ r (i : Fin (before.capacity r)),Physical.Exterior p r i.val →
        ∃ j : Fin (result.memory.capacity r),j.val=i.val ∧ result.memory.cells r j=(before.store p).cells r i) ∧
      (Live p → Access.Safe p storage) ∧ Closed p bases) := by
  refine ⟨(Contract.check_iff p bases args).mpr admitted,?_⟩
  exact RealReturned.returned_real p bases (Contract.domain p bases args admitted) (Contract.abi p args admitted.1) (before.store p) storage fits (Contract.positive_block p bases args admitted) before.logical
    (RealReturned.input_values_reads p before) work legal

#specNonCircular target avoiding [Physical.invoke,Whole.invoke,Whole.answer,FP.Structural.exec]
#axiomsClean attention_real
end
end VeriTile.Bench.Repairs.AttentionKernel.Correct
