import bench.repairs.tritonbench_g.attn_fwd_triton_tile_checked.Proofs.Physical

namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.RawNative
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open AttnFwdTriton TypedExecution
open VeriTile.Bench.Repairs.LayerNormFwd
open TypedPhysical (Physical pull)
open Verified.PhysicalCells (Layout Role FreshBytes)
open Graph
noncomputable section

/-- Complete production-division kernel grid from raw byte memory to native
BF16 cells. The independent two-loop target retains the existing parameterized
TF32 score dot, FP16 value dot, reductions and selected FMAs. No Tensor admission
or numerical rewrite assumption occurs in this interface. -/
theorem whole_native (ops : Operations) (p : Config) (a : Layout) (base : Role→Nat)
    (views : Views a p base) (memory : Checked.ByteMemory) (template : State Word)
    (stage : p.stage=3) (nonoverlap : NonOverlapping p)
    (fresh : FreshBytes a (fun k=>Window p k.1 k.2)) :
    ∃ final, ∃ graph : Physical Expr a.toLayout, ∃ physical : Physical Word a.toLayout,
      Nonempty (GridRun backend p (HostShape.blockD p.headDim) Bool.false
        (Physical.initial a memory template) final) ∧
      pull a.toLayout graph final=final ∧ physical=Physical.evaluatePhysical ops graph ∧
      (∀ c : Coordinate p,physical (outputCell a p base views c)=
        Physical.reference ops p a memory template c) ∧
      (∀ cell : Fin a.capacity,(∀ k,Window p k.1 k.2 → a.address k≠cell.val) →
        physical cell=Physical.callerCells a memory cell) :=
  Physical.grid ops p a base views memory template stage nonoverlap fresh

#axiomsClean whole_native
end
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.RawNative
