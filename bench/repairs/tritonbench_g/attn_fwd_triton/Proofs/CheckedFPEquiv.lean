import bench.repairs.tritonbench_g.attn_fwd_triton.Contract
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.NativeGraphHost

namespace VeriTile.Bench.Repairs.AttnFwdTriton.Checked
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.LayerNormFwd
open Verified.PhysicalCells (Role Layout width FreshBytes)
open TypedPhysical (Physical pull)
open TypedExecution NativeGraph
noncomputable section
set_option maxHeartbeats 1000000

def typedGeometry (p : TypedExecution.Config) : Geometry where
  batches:=p.batches
  heads:=p.heads
  context:=p.context
  headDim:=p.headDim
  stride
    | .query=>⟨p.queryBatch,p.queryHead,p.queryRow,p.queryDim⟩
    | .key=>⟨p.keyBatch,p.keyHead,p.keyRow,p.keyDim⟩
    | .value=>⟨p.valueBatch,p.valueHead,p.valueRow,p.valueDim⟩
    | .output=>⟨p.outputBatch,p.outputHead,p.outputRow,p.outputDim⟩
    | _=>⟨0,0,1,0⟩

/-- Complete native division graph directly on caller memory. No resolve,
packing, reciprocal rewrite, numerical-to-Real equality or output premise is
assumed. The local Operations are the selected native instruction semantics. -/
theorem whole_native_cells (ops : Operations) (p : TypedExecution.Config) (a : Layout)
    (base : Role→Nat) (views : TypedExecution.Views a p base)
    (q k v qs ks : TensorMetadata) (hinput : inputChecks (typedGeometry p) q k v qs ks=Bool.true)
    (hl : launchChecks (typedGeometry p) qs ks=Bool.true)
    (hp : pointerChecks (typedGeometry p) base=Bool.true)
    (caller : Physical Word a.toLayout) (template : State Word)
    (stage : p.stage=3) (nonoverlap : NonOverlapping p)
    (fresh : FreshBytes a (fun k=>Window p k.1 k.2)) :
    inputChecks (typedGeometry p) q k v qs ks=Bool.true ∧ p.stage=3 ∧
    NativeLaunch.CudaLaunchable (launchShape p) (HostShape.blockD p.headDim) ∧
    (∀ r n, n<envelope (typedGeometry p) → base r%width r=0 ∧ base r+width r*n+width r<2^64) ∧
    ∃ final, ∃ graphPhysical : Physical Expr a.toLayout, ∃ physical : Physical Word a.toLayout,
      Nonempty (GridRun backend p (HostShape.blockD p.headDim) Bool.false
        (pull a.toLayout (liftPhysical caller) (liftState template)) final) ∧
      pull a.toLayout graphPhysical final=final ∧
      physical=evaluatePhysical ops graphPhysical ∧
      (∀ c : Coordinate p, physical (outputCell a p base views c)=
        NativeGraph.reference ops p (HostShape.blockD p.headDim) a caller template c) ∧
      (∀ cell : Fin a.capacity,
        (∀ k, Window p k.1 k.2 → a.address k≠cell.val) → physical cell=caller cell) := by
  refine ⟨hinput,stage,launchable (typedGeometry p) qs ks hl,?_,?_⟩
  · intro r n hn
    have h:=formed_pointer (typedGeometry p) base hp r n hn
    exact ⟨h.1,h.2.1⟩
  · obtain ⟨final,graph,hg,hpull,hread,hframe⟩:=physical_division_grid backend p
      (HostShape.blockD p.headDim) (support _) a base views (liftPhysical caller) (liftState template)
      (HostShape.blockD_covers _) nonoverlap fresh
    refine ⟨final,graph,evaluatePhysical ops graph,hg,hpull,rfl,?_,?_⟩
    · intro c
      exact congrArg (evaluateCell ops) (hread c)
    · intro cell hc
      change evaluateCell ops (graph cell)=_
      rw [hframe cell hc]
      exact evaluate_lift ops (caller cell)

#axiomsClean whole_native_cells
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Checked
