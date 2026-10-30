import bench.tritonbench_g.matmul_dequant_int4.Proofs.Source
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.Source
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 18000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
attribute [local simp 10000] FP.Structural.eval_intShiftRight FP.Structural.eval_intBitAnd

def safetyPrefix (f : Profile) (p : Parameters) := (source f p).surfaceBody.take 11
def safetySuffix (f : Profile) (p : Parameters) := (source f p).surfaceBody.drop 11

/-- Only source-produced address and mask registers cross the split. Values
loaded later remain completely arbitrary, including packed/native payloads. -/
structure SafetyControls {α : Type} (p : Parameters) (initial s : State α) : Prop where
  weight : s.regs .nat [p.blockRows,p.blockColumns] "b_offs"=some (weightOffset p initial)
  zero : s.regs .nat [p.blockRows,p.blockColumns] "bzp_offs"=some (zeroOffset p initial)
  scale : s.regs .nat [p.blockRows,p.blockColumns] "bs_offs"=some (scaleOffset p initial)
  output : s.regs .nat [p.blockRows,p.blockColumns] "fpb_offs"=some (outputOffset p initial)
  mask : s.regs .bool [p.blockRows,p.blockColumns] "mask"=some (fun i => decide (active p initial i))

theorem safety_prefix_run {α : Type} [Inhabited α] (f : Profile) (M : Algebra α)
    (p : Parameters) (initial : State α) :
    ∃ s,run M (safetyPrefix f p) initial=some s ∧ SafetyControls p initial s := by
  cases f <;>
    simp [safetyPrefix,source,Profile.dtype,
      VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.typed,ComputeKernel.surfaceBody,
      run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      ComputeDType.eraseDType,numeric,bop,natLt,numericLt,numericEq,Region.cast,
      State.setReg,integralFloorDiv,integralMod,FloatDType.numericDType,
      FloatDType.toTileDType,ofFloat,toFloat]
  all_goals
    constructor <;> simp [SafetyControls,weightOffset,zeroOffset,scaleOffset,outputOffset,
      active,row,column,bop,Broadcast.leftIndex,Broadcast.rightIndex,
      integralFloorDiv,integralMod,and_comm]
  all_goals funext i
  all_goals simp [weightOffset,zeroOffset,scaleOffset,outputOffset,active,row,column,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Bool.and_comm]
  all_goals rfl

theorem safety_prefix_safe {α : Type} [Inhabited α] (f : Profile) (M : Algebra α)
    (p : Parameters) (s : State α) (bounds : RegionName → Nat) :
    Safety.traceSafeList M bounds (safetyPrefix f p) s := by
  cases f <;>
    simp [safetyPrefix,source,Profile.dtype,
      VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.typed,
      ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
      Safety.maskActive_unfold,run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,
      numericEq,Region.cast,State.setReg,integralFloorDiv,integralMod,
      FloatDType.numericDType,FloatDType.toTileDType,ofFloat,toFloat]

def safetyBody (f : Profile) (p : Parameters) := ((source f p).surfaceBody.drop 11).take 6
def safetyStore (f : Profile) (p : Parameters) := (source f p).surfaceBody.drop 17

private theorem controls_setReg {α : Type} (p : Parameters) (initial s : State α)
    (h : SafetyControls p initial s) (name : RegName) (d : TileDType) (sh : TileShape)
    (vals : Values α d sh) (names : name≠"b_offs" ∧ name≠"bzp_offs" ∧ name≠"bs_offs" ∧
      name≠"fpb_offs" ∧ name≠"mask") : SafetyControls p initial (s.setReg name d sh vals) := by
  rcases h with ⟨hw,hz,hs,ho,hm⟩
  constructor <;> simp [State.setReg,names.1,names.2.1,names.2.2.1,names.2.2.2.1,
    names.2.2.2.2,Ne.symm names.1,Ne.symm names.2.1,Ne.symm names.2.2.1,
    Ne.symm names.2.2.2.1,Ne.symm names.2.2.2.2,hw,hz,hs,ho,hm]

private theorem assignment_preserves {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (initial s t : State α) (h : SafetyControls p initial s)
    (name : RegName) (d : TileDType) (sh : TileShape) (e : ComputeExpr d sh)
    (names : name≠"b_offs" ∧ name≠"bzp_offs" ∧ name≠"bs_offs" ∧
      name≠"fpb_offs" ∧ name≠"mask")
    (ran : step M (.assign d sh name e) s=some t) : SafetyControls p initial t := by
  simp only [step] at ran
  cases hx : FP.Structural.evalExpr M e s with
  | none => simp [hx] at ran
  | some vals =>
    simp [hx] at ran
    subst t
    exact controls_setReg p initial s h name d sh vals names

private theorem body_preserves {α : Type} [Inhabited α] (f : Profile) (M : Algebra α)
    (p : Parameters) (initial s t : State α) (h : SafetyControls p initial s)
    (st : ComputeStmt) (member : st∈safetyBody f p) (ran : step M st s=some t) :
    SafetyControls p initial t := by
  cases f <;>
    simp only [safetyBody,source,Profile.dtype,
      VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.typed,
      ComputeKernel.surfaceBody,List.drop,List.take,List.mem_cons,List.not_mem_nil,or_false] at member
  all_goals rcases member with rfl|rfl|rfl|rfl|rfl|rfl
  all_goals exact assignment_preserves M p initial s t h _ _ _ _ (by decide) ran

private theorem body_step_safe {α : Type} [Inhabited α] (f : Profile) (M : Algebra α)
    (p : Parameters) (initial s : State α) (bounds : RegionName → Nat)
    (controls : SafetyControls p initial s)
    (weight : ∀ i : Index p,active p initial i → weightOffset p initial i<bounds p.weight)
    (zero : ∀ i : Index p,active p initial i → zeroOffset p initial i<bounds p.zero)
    (scale : ∀ i : Index p,active p initial i → scaleOffset p initial i<bounds p.scale)
    (st : ComputeStmt) (member : st∈safetyBody f p) : Safety.stepSafe M bounds st s := by
  rcases controls with ⟨hw,hz,hs,ho,hm⟩
  cases f <;>
    simp only [safetyBody,source,Profile.dtype,
      VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.typed,
      ComputeKernel.surfaceBody,List.drop,List.take,List.mem_cons,List.not_mem_nil,or_false] at member
  all_goals rcases member with rfl|rfl|rfl|rfl|rfl|rfl
  all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,
    Safety.addressSafe_unfold,Safety.maskActive_unfold,evalOp_unfold,
    hw,hz,hs,hm,Region.cast]
  all_goals intro i j u active
  all_goals first | exact weight (i,j,u) active | exact zero (i,j,u) active | exact scale (i,j,u) active

private theorem body_safe_preserves {α : Type} [Inhabited α] (f : Profile) (M : Algebra α)
    (p : Parameters) (initial : State α) (bounds : RegionName → Nat)
    (weight : ∀ i : Index p,active p initial i → weightOffset p initial i<bounds p.weight)
    (zero : ∀ i : Index p,active p initial i → zeroOffset p initial i<bounds p.zero)
    (scale : ∀ i : Index p,active p initial i → scaleOffset p initial i<bounds p.scale)
    (stmts : List ComputeStmt) (members : ∀ st∈stmts,st∈safetyBody f p)
    (s : State α) (controls : SafetyControls p initial s) :
    Safety.traceSafeList M bounds stmts s ∧
      ∀ t,run M stmts s=some t → SafetyControls p initial t := by
  induction stmts generalizing s with
  | nil =>
    refine ⟨Safety.traceSafeList_nil M bounds s,?_⟩
    intro t h
    simp only [run] at h
    cases h
    exact controls
  | cons st rest ih =>
    have stmem := members st (by simp)
    have head := body_step_safe f M p initial s bounds controls weight zero scale st stmem
    have restmem : ∀ st∈rest,st∈safetyBody f p := fun st h => members st (by simp [h])
    constructor
    · apply Safety.traceSafeList_cons_intro head
      intro t ran
      exact (ih restmem t (body_preserves f M p initial s t controls st stmem ran)).1
    · intro t ran
      simp only [run] at ran
      cases hx : step M st s with
      | none => simp [hx] at ran
      | some middle =>
        simp only [hx,Option.bind_some] at ran
        exact (ih restmem middle (body_preserves f M p initial s middle controls st stmem hx)).2 t ran

private theorem store_safe {α : Type} [Inhabited α] (f : Profile) (M : Algebra α)
    (p : Parameters) (initial s : State α) (bounds : RegionName → Nat)
    (controls : SafetyControls p initial s)
    (output : ∀ i : Index p,active p initial i → outputOffset p initial i<bounds p.output) :
    Safety.traceSafeList M bounds (safetyStore f p) s := by
  rcases controls with ⟨hw,hz,hs,ho,hm⟩
  cases f <;>
    simp [safetyStore,source,Profile.dtype,
      VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.typed,
      ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
      Safety.maskActive_unfold,evalOp_unfold,ho,hm,Region.cast]
  all_goals
    refine ⟨fun i j u active => output (i,j,u) active,?_⟩
    split <;> trivial

theorem safety_suffix_safe {α : Type} [Inhabited α] (f : Profile) (M : Algebra α)
    (p : Parameters) (initial s : State α) (bounds : RegionName → Nat)
    (controls : SafetyControls p initial s)
    (weight : ∀ i : Index p,active p initial i → weightOffset p initial i<bounds p.weight)
    (zero : ∀ i : Index p,active p initial i → zeroOffset p initial i<bounds p.zero)
    (scale : ∀ i : Index p,active p initial i → scaleOffset p initial i<bounds p.scale)
    (output : ∀ i : Index p,active p initial i → outputOffset p initial i<bounds p.output) :
    Safety.traceSafeList M bounds (safetySuffix f p) s := by
  have split : safetySuffix f p=safetyBody f p++safetyStore f p := by
    change (source f p).surfaceBody.drop 11 =
      ((source f p).surfaceBody.drop 11).take 6 ++ (source f p).surfaceBody.drop 17
    have h := List.take_append_drop 6 ((source f p).surfaceBody.drop 11)
    simpa only [List.drop_drop] using h.symm
  rw [split,Safety.traceSafeList_append]
  have body := body_safe_preserves f M p initial bounds weight zero scale
    (safetyBody f p) (fun _ h => h) s controls
  exact ⟨body.1,fun t ran => store_safe f M p initial t bounds (body.2 t ran) output⟩

/-- Exact packed/scale loads and output store under the actual two-axis mask;
the split follows real predecessor states and preserves both precision profiles. -/
theorem trace_safe {α : Type} [Inhabited α] (f : Profile) (M : Algebra α)
    (p : Parameters) (s : State α) (bounds : RegionName → Nat)
    (weight : ∀ i : Index p,active p s i → weightOffset p s i<bounds p.weight)
    (zero : ∀ i : Index p,active p s i → zeroOffset p s i<bounds p.zero)
    (scale : ∀ i : Index p,active p s i → scaleOffset p s i<bounds p.scale)
    (output : ∀ i : Index p,active p s i → outputOffset p s i<bounds p.output) :
    TraceSafe M bounds (source f p) s := by
  unfold TraceSafe
  rw [←List.take_append_drop 11 (source f p).surfaceBody,Safety.traceSafeList_append]
  refine ⟨safety_prefix_safe f M p s bounds,?_⟩
  intro next ran
  obtain ⟨actual,run,controls⟩ := safety_prefix_run f M p s
  change FP.Structural.run M (safetyPrefix f p) s=some next at ran
  rw [run] at ran
  cases ran
  exact safety_suffix_safe f M p s next bounds controls weight zero scale output

#axiomsClean trace_safe
end
end VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.Source
