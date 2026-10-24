import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperRoleReadback

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper
noncomputable section
set_option maxHeartbeats 3000000
variable {α Host : Type} {caller : Address → Prop}

/-- The currently implemented native storage profile on public request
metadata. No finiteness, sign, nonzero size or numeric accuracy condition is
imposed. Other floating storage profiles remain unfinished program work. -/
structure SelectedRequest (q : Request) : Prop where
  inputs : ∀ role, sourceInputRole role → ∀ tensor,
    originalRoleTensor q role=some tensor → tensor.dtype=DType.fp32
  output : q.outDtype.getD DType.fp32=DType.fp32
  residualDefault : q.residual=none → q.residualDtype.getD DType.fp32=DType.fp32

theorem selected_request_storage
    (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current) (hv : ValidatedFacts q p.validated)
    (hp : SelectedRequest q) : SelectedStorage q p := by
  have hx : p.inputs.x.dtype=DType.fp32 := h.inputs.x.dtype.trans (hp.inputs .x trivial q.x rfl)
  have hy : p.outputs.y.dtype=DType.fp32 := by
    have hd := h.outputs.1.dtype
    simpa only [outputRequests,hx,hp.output] using hd
  have hr : p.validated.residualDtype.getD DType.fp32=DType.fp32 := by
    rw [hv.residualDtype]
    cases he : q.residual with
    | none => exact hp.residualDefault he
    | some value => exact hp.inputs .residual trivial value he
  intro role _ value he
  by_cases hs : sourceInputRole role
  · obtain ⟨source,hsource,hmaterialized⟩ := role_materialized store tensor q initial current p args h role hs value he
    have hd := hmaterialized.dtype.trans (hp.inputs role hs source hsource)
    cases role <;> simp_all [sourceInputRole,roleDType]
  · cases role with
    | x | weight | residual | x1 | bias | weight1 | bias1 | rowscale | rowscale1 =>
      exact False.elim (hs True.intro)
    | y =>
      have heq : p.outputs.y=value := Option.some.inj he
      exact heq ▸ hy
    | y1 =>
      have hd := (allocated_of_some h.outputs.2.1 value he).dtype
      simpa only [outputRequests,roleDType,hy] using hd
    | residualOut =>
      have hd := (allocated_of_some h.outputs.2.2.1 value he).dtype
      simpa only [outputRequests,roleDType,hx,hr] using hd
    | rstd =>
      have heq : p.outputs.rstd=value := Option.some.inj he
      exact heq ▸ h.outputs.2.2.2.2.1.dtype
    | mean => exact (allocated_of_some h.outputs.2.2.2.1 value he).dtype
    | seeds => exact (allocated_of_some h.outputs.2.2.2.2.2.1 value he).dtype
    | seeds1 => exact (allocated_of_some h.outputs.2.2.2.2.2.1 value he).dtype
    | mask => exact (allocated_of_some h.outputs.2.2.2.2.2.2 value he).dtype
    | mask1 => exact (allocated_of_some h.outputs.2.2.2.2.2.2 value he).dtype

#axiomsClean selected_request_storage
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
