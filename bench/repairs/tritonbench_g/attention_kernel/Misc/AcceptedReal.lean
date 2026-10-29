import bench.repairs.tritonbench_g.attention_kernel.Proofs.RealReturned
namespace VeriTile.Bench.Repairs.AttentionKernel.Whole002
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionKernel.Whole001 Source Whole Machine
noncomputable section
/-- Complete exact-Real interpretation: original storage tags, actual caller
output, finite memory, arbitrary legal CTA order, all logical cells, full
exterior frame and Python None. This is not a zero-rounding-error claim. -/
theorem whole_returned_real (p : Parameters) (bases : Bases) (domain : Domain p bases)
    (abi : InputABI p) (before : RealReturned.InputValues) (storage : TensorStorage p)
    (fits : storage.capacity=before.capacity) (hbm : 0<p.bm)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    ∃ result,Physical.invoke (fun _=>Real.algebra) (fun _=>Real.dotModel) p abi (before.store p) work=some result ∧
      result.result=() ∧ result.memory.capacity=before.capacity ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<p.dim),
        ∃ index : Fin (result.memory.capacity p.Out),index.val=logicalAddress p b h n d ∧
          result.memory.cells p.Out index=.mk p.storage.tile
            (ofFloat p.storage.float (some (RealReturned.target p before.logical b h n d)))) ∧
      (∀ r (i : Fin (before.capacity r)),Physical.Exterior p r i.val →
        ∃ j : Fin (result.memory.capacity r),j.val=i.val ∧ result.memory.cells r j=(before.store p).cells r i) ∧
      (Live p → Access.Safe p storage) ∧ Closed p bases :=
  RealReturned.returned_real p bases domain abi (before.store p) storage fits hbm before.logical
    (RealReturned.input_values_reads p before) work legal

#axiomsClean whole_returned_real
end
end VeriTile.Bench.Repairs.AttentionKernel.Whole002
