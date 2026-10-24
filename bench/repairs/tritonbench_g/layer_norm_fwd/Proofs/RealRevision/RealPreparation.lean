import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.RealRevision.RealMemory

/-! Local shadow transports compose along exactly the actual successful prefix.
The generic preparation proof is reused; no original-to-prepared equation is
assumed at the wrapper interface. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.RealRevision
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper WrapperNative
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 12000
variable {α Host : Type} {caller : Address → Prop}

theorem shadow_prefix
    (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (ρ : HostState α Host → Address → ℝ) (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (laws : Transport store ρ tensor) (device : DispatchAPI (world caller))
    (hs : StorageAPI (shadowStore store ρ) (revalueTensor store ρ tensor laws))
    (hf : FreshAllocationAPI store tensor)
    (hd : DispatchStorage (shadowStore store ρ) (revalueDispatch device))
    (q : Request) (initial : HostState α Host) (hp : RequestPresent store q initial)
    (p : Prepared) (args : LaunchArguments) (route : LaunchPrefix tensor device q initial p args) :
    LaunchStateBound (shadowStore store ρ) (revalueTensor store ρ tensor laws) q initial p args
      (launchState tensor device q initial p args) := by
  let rt := revalueTensor store ρ tensor laws
  let rd := revalueDispatch device
  have ep : preparedState rt q initial=preparedState tensor q initial :=
    congrArg Outcome.state (prepare_run store ρ tensor laws q initial)
  have ee : enteredState rd p (preparedState rt q initial)=enteredState device p (preparedState tensor q initial) := by
    rw [ep]; rfl
  have ea : argumentsState rd q p (enteredState rd p (preparedState rt q initial))=
      argumentsState device q p (enteredState device p (preparedState tensor q initial)) := by
    unfold argumentsState
    rw [ee,arguments_run]
  have hroute : LaunchPrefix rt rd q initial p args := by
    refine ⟨?_,?_,?_,?_⟩
    · rw [prepare_run]; exact route.prepared
    · change ((device.enter p.inputs.x.device).run (preparedState rt q initial)).2=.ok ()
      rw [ep]; exact route.entered
    · rw [arguments_run,ee]; exact route.arguments
    · change ((device.admit args).run (argumentsState rd q p (enteredState rd p (preparedState rt q initial)))).2=.ok ()
      rw [ea]; exact route.admitted
  have hf' : FreshAllocationAPI (shadowStore store ρ) rt := ⟨hf.fresh⟩
  have hbound := prefix_launch_state (shadowStore store ρ) rt rd hs hf' hd q initial ⟨hp.x,hp.weight,hp.bias,hp.residual,hp.x1,hp.weight1,hp.bias1,hp.rowscale⟩ p args hroute
  rw [launchState_eq] at hbound
  exact hbound

/-- Geometry determines this physical role read for every Real memory. -/
theorem role_read
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current) (hv : ValidatedFacts q p.validated)
    (hi : InputsDense p.inputs) (ho : OutputsDense p.outputs) (memory : RealMemory)
    (row col : Nat) (hr : row<p.validated.M) (role : LayerNormFwd.Address.Role)
    (ha : LayerNormFwd.Address.active (p.flags q) role) (hl : NativeAccess.laneActive p.validated.N col role)
    (value : Tensor) (he : roleTensor p role=some value) :
    logicalRead store current memory value (roleIndex p.validated.M row col role)=
      AliasMemory.read (PhysicalDomain.layout (preparedAllocations store current p)) memory
        (LayerNormFwd.Address.region preparedRegions role,
          LayerNormFwd.Address.offset (argumentStrides args) p.validated.M p.validated.N row col role) := by
  obtain ⟨actual,heq,hpresent,hpositive,hcoordinate,hoffset⟩ :=
    active_role_coordinate store hm tensor q initial current p args h hv hi ho row col hr role ha hl
  have hav : actual=value := Option.some.inj (heq.symm.trans he)
  subst actual
  have hd := preparedAllocation_role store current p role value he
  have hb := hm.bounds current value hpresent _ hcoordinate
  have hbound := (LayerNormFwd.Address.byte_address (store.descriptor current value) hb).2.2.1
  have hLive : (PhysicalDomain.layout (preparedAllocations store current p)).live
      (LayerNormFwd.Address.region preparedRegions role,
        LayerNormFwd.Address.offset (argumentStrides args) p.validated.M p.validated.N row col role) := by
    change _<(preparedAllocations store current p _).extent
    rw [hd,hoffset]
    exact hb
  simp only [logicalRead,if_pos hcoordinate,hpositive,Bool.false_eq_true,ite_false,rawRead,AliasMemory.read,dif_pos hLive]
  congr 1
  apply Fin.ext
  change tensorAddress store current value (roleIndex p.validated.M row col role) % 2^64 = _
  unfold tensorAddress
  rw [Nat.mod_eq_of_lt hbound]
  change LayerNormFwd.Address.byteAddress _ _=LayerNormFwd.Address.byteAddress _ _
  rw [hd,hoffset]

/-- Role selection is reused from the actual request; only the generic value
carrier of the existing Materialized proof changes. -/
theorem shadow_role_materialized
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (ρ : HostState α Host → Address → ℝ)
    (q : Request) (initial current : HostState α Host) (p : Prepared)
    (h : InputsMaterialized (shadowStore store ρ) q initial p.inputs current)
    (role : LayerNormFwd.Address.Role) (hi : sourceInputRole role)
    (value : Tensor) (he : roleTensor p role=some value) :
    ∃ source, originalRoleTensor q role=some source ∧ Materialized (shadowStore store ρ) source initial value current := by
  cases role <;> simp only [sourceInputRole] at hi
  all_goals try contradiction
  case x =>
    have hv : p.inputs.x=value := Option.some.inj he
    exact ⟨q.x,rfl,hv ▸ h.x⟩
  case weight =>
    have hv : p.inputs.weight=value := Option.some.inj he
    exact ⟨q.weight,rfl,hv ▸ h.weight⟩
  case residual => exact materialized_of_some h.residual value he
  case x1 => exact materialized_of_some h.x1 value he
  case bias => exact materialized_of_some h.bias value he
  case weight1 => exact materialized_of_some h.weight1 value he
  case bias1 => exact materialized_of_some h.bias1 value he
  case rowscale => exact materialized_of_some h.rowscale value he
  case rowscale1 => exact materialized_of_some h.rowscale value he

theorem input_original
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (ρ : HostState α Host → Address → ℝ) (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current) (hv : ValidatedFacts q p.validated)
    (hi : InputsDense p.inputs) (ho : OutputsDense p.outputs)
    (hs : InputsMaterialized (shadowStore store ρ) q initial p.inputs current)
    (row col : Nat) (hr : row<p.validated.M) (role : LayerNormFwd.Address.Role)
    (ha : LayerNormFwd.Address.active (p.flags q) role) (hl : NativeAccess.laneActive p.validated.N col role)
    (hrole : sourceInputRole role) :
    ∃ source, originalRoleTensor q role=some source ∧
      AliasMemory.read (PhysicalDomain.layout (preparedAllocations store current p)) (interpret ρ current)
        (LayerNormFwd.Address.region preparedRegions role,
          LayerNormFwd.Address.offset (argumentStrides args) p.validated.M p.validated.N row col role)=
        (shadowStore store ρ).read initial source (roleIndex p.validated.M row col role) := by
  obtain ⟨value,he,_⟩ := active_role_coordinate store hm tensor q initial current p args h hv hi ho row col hr role ha hl
  obtain ⟨source,hsource,hmat⟩ := shadow_role_materialized store ρ q initial current p hs role hrole value he
  exact ⟨source,hsource,(role_read store hm tensor q initial current p args h hv hi ho
    (interpret ρ current) row col hr role ha hl value he).symm.trans (hmat.values _)⟩

#axiomsClean shadow_prefix
#axiomsClean role_read
#axiomsClean input_original
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.RealRevision
