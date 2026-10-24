import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardFlatLoop
import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardFlatPrefix

/-! Full backward source finite-allocation execution bridge. The source is
unchanged; the final six real pointer subtractions still execute. The bridge
concludes complete memory and launch observation equality, not false equality
of dead final pointer registers. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardFlatExecution
open VeriTile.Triton BackwardSource
noncomputable section
set_option maxHeartbeats 1400000
set_option linter.unusedSimpArgs false

theorem executes (A : FlatAlloc) (ha : A.Disjoint)
    (hc : ∀ region, region ∉ A.regions → A.extent region = 0)
    (p : Layout) (r : BackwardRegions) (hasSeed swapped : Bool)
    (initial final : BlockState) (hd : BackwardReal.Disjoint r)
    (hb : BackwardSafety.Bounds p (launch initial) r hasSeed A.extent)
    (hz : initial.undef = fun _ _ => 0)
    (he : exec (kernel p r hasSeed swapped).toAlgKernel initial = some final) :
    ∃ flat, exec (A.flattenKernel (kernel p r hasSeed swapped).toAlgKernel)
      (A.flattenState initial) = some flat ∧ flat.ObsAgree (A.flattenState final) := by
  obtain ⟨middle,hpre,hm⟩ := BackwardFlatPrefix.executes p r hasSeed swapped initial hd
  have hzero : middle.undef = fun _ _ => 0 :=
    (stepStmts_undef _ initial middle (BackwardFlatPrefix.flatten_ok p r hasSeed swapped) hpre).trans hz
  have hflatpre := A.stepStmts_flatten ha hc _ initial
    (BackwardFlatPrefix.safe p r hasSeed swapped initial hd A.extent hb)
    (BackwardFlatPrefix.flatten_ok p r hasSeed swapped) hz
  rw [hpre,Option.map_some] at hflatpre
  rw [exec,BackwardFlatPrefix.split,stepStmts.append_some hpre] at he
  obtain ⟨last,hl,hnil⟩ := TraceProofs.cons_inv he
  rw [stepStmts.nil] at hnil
  cases hnil
  rw [stepForRangeAux.forRange_unfold] at hl
  obtain ⟨flat,hflat,hobs,_⟩ := BackwardFlatLoop.executes A ha hc p (launch initial) initial r
    hasSeed hd hb middle final hm hzero hl
  refine ⟨flat,?_,hobs⟩
  rw [exec,FlatAlloc.flattenKernel,BackwardFlatPrefix.split,A.flattenStmts_append,
    stepStmts.append_some hflatpre]
  have hstmt : stepStmt (.forRange "_t" 0 p.T 1 (A.flattenStmts (secondBody p)))
      (A.flattenState middle) = some flat := by
    rw [stepForRangeAux.forRange_unfold]
    exact hflat
  change stepStmts [.forRange "_t" 0 p.T 1 (A.flattenStmts (secondBody p))]
    (A.flattenState middle) = some flat
  rw [stepStmts.cons_some hstmt,stepStmts.nil]

#axiomsClean executes
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardFlatExecution
