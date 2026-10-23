import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.Scope
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.HostResults
namespace VeriTile.Bench.Repairs.LayerNormOps.Scope
open VeriTile.Triton

/-- Active allocation regions only. Optional returned dResidual aliases DX
under the selected default dtype and does not create an allocation entry. -/
def forwardOutputs (p : ForwardParameters) : List RegionName :=
  [p.y,p.rstd] ++ (if p.rms then [] else [p.mean]) ++
    (if p.storeResidual then [p.residualOut] else [])
def backwardOutputs (p : BackwardParameters) (dw db : RegionName) : List RegionName :=
  [p.dx,p.dw,dw] ++ (if p.hasBias then [p.db,db] else []) ++
    (if p.recompute then [p.y] else []) ++ (if p.storeDres then [p.dresIn] else [])

/-- The ordinary allocator and saved-tensor lifetime boundary. Input regions
are allowed to alias each other. Fresh allocations cannot overwrite caller-live
storage, and backward allocations cannot overwrite forward's saved outputs. -/
structure Allocations (fp : ForwardParameters) (bp : BackwardParameters)
    (dw db : RegionName) (callerLive : List RegionName) : Prop where
  forwardDistinct : (forwardOutputs fp).Nodup
  backwardDistinct : (backwardOutputs bp dw db).Nodup
  forwardFresh : ∀ r ∈ forwardOutputs fp, r ∉ callerLive
  backwardFresh : ∀ r ∈ backwardOutputs bp dw db, r ∉ callerLive ++ forwardOutputs fp
  xLive : fp.x ∈ callerLive
  weightLive : fp.weight ∈ callerLive
  biasLive : fp.hasBias=Bool.true → fp.bias ∈ callerLive
  residualLive : fp.hasResidual=Bool.true → fp.residual ∈ callerLive
  dyLive : bp.dy ∈ callerLive
  dresLive : bp.hasDres=Bool.true → bp.dres ∈ callerLive

/-- Freshness yields the complete operation's caller-memory frame premises. -/
theorem caller_outside (fp : ForwardParameters) (bp : BackwardParameters)
    (dw db : RegionName) (live : List RegionName) (h : Allocations fp bp dw db live)
    (r : RegionName) (hr : r∈live) :
    (r≠fp.y ∧ (fp.storeResidual=Bool.true → r≠fp.residualOut) ∧
      (fp.rms=Bool.false → r≠fp.mean) ∧ r≠fp.rstd) ∧
    Backward.RegionOutside bp r ∧ r≠dw ∧ (bp.hasBias=Bool.true → r≠db) := by
  have hf : ∀ out ∈ forwardOutputs fp, r≠out := by
    intro out hout he
    exact h.forwardFresh out hout (he ▸ hr)
  have hb : ∀ out ∈ backwardOutputs bp dw db, r≠out := by
    intro out hout he
    exact h.backwardFresh out hout (List.mem_append_left _ (he ▸ hr))
  have f : r≠fp.y ∧ (fp.storeResidual=Bool.true → r≠fp.residualOut) ∧
      (fp.rms=Bool.false → r≠fp.mean) ∧ r≠fp.rstd := by
    refine ⟨hf _ (by simp [forwardOutputs]),?_,?_,hf _ (by simp [forwardOutputs])⟩
    · intro hs; exact hf _ (by simp [forwardOutputs,hs])
    · intro hs; exact hf _ (by simp [forwardOutputs,hs])
  have b : Backward.RegionOutside bp r := by
    refine ⟨hb _ (by simp [backwardOutputs]),hb _ (by simp [backwardOutputs]),?_,?_,?_⟩
    · intro hs; exact hb _ (by simp [backwardOutputs,hs])
    · intro hs; exact hb _ (by simp [backwardOutputs,hs])
    · intro hs; exact hb _ (by simp [backwardOutputs,hs])
  refine ⟨f,b,hb _ (by simp [backwardOutputs]),?_⟩
  intro hs
  exact hb _ (by simp [backwardOutputs,hs])

#axiomsClean caller_outside
end VeriTile.Bench.Repairs.LayerNormOps.Scope
