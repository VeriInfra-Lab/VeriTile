import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperReturnViews

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper NativeMachine
noncomputable section
set_option maxHeartbeats 3000000
variable {α Host : Type} [Inhabited α] {caller : Address → Prop}

/-- The fallback returns the original handle while the native X pointer may
refer to a materialized copy. Both represent the same logical coordinate after
all benchmarks and the selected execution; this equality is derived. -/
theorem dispatch_fallback_readback
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (hh : HostDescriptors store) (hl : LogicalReadLocal store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (hp : RequestPresent store q initial)
    (h : LaunchStateBound store tensor q initial p args current) (hv : ValidatedFacts q p.validated)
    (hi : InputsDense p.inputs) (ho : OutputsDense p.outputs)
    (c : Config α block) (hc : PreparedPlacement store q p args current c)
    (hsep : Domain.ReadWriteSeparated c.flags c.regions)
    (hgeo : NativeGeometry.Geometry c.flags c.regions c.strides c.M c.N)
    (benchmarks : List (DispatchEvent block)) (selected : Schedule block) (template : State α)
    (row col : Nat) (hr : row<p.validated.M) (hcol : col<p.validated.N) :
    let native := {current with memory:=(dispatch c benchmarks selected current.memory).memory}
    store.read native p.inputs.originalX [row,col]=
      (view c native.memory template).mem preparedRegions.x (row*(argumentStrides args).x+col) := by
  have hindex := coordinate_matrix q.x p.validated.M p.validated.N row col hv.x.1 hr hcol
  have hprior := dispatch_prior_read store hm hh hl tensor q initial current p args h hv hi ho c hc
    benchmarks selected q.x hp.x [row,col] hindex
  obtain ⟨source,hsource,hvalue⟩ := prepared_input_original store hm tensor q initial current p args h hv hi ho
    c hc.allocations template row col hr .x trivial hcol trivial
  have hsource : q.x=source := Option.some.inj hsource
  rw [←hsource] at hvalue
  have hread : Domain.readRegion c.flags c.regions preparedRegions.x := by
    rw [hc.regions]
    exact Or.inl rfl
  have hframe := dispatch_reads c hsep hgeo benchmarks selected current.memory template
    preparedRegions.x hread (row*(argumentStrides args).x+col)
  dsimp only
  rw [h.inputs.original]
  exact hprior.trans (hvalue.symm.trans hframe.symm)

#axiomsClean dispatch_fallback_readback
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
