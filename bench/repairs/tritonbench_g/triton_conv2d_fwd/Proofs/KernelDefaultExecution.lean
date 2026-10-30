import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelDefaultGraph
namespace VeriTile.Bench.Repairs.Conv2d.KernelDefault
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

structure Context {α : Type} (p : Parameters) (initial current : State α) : Prop where
  memory : current.mem=initial.mem
  undefined : current.undef=initial.undef
  group : current.regs .nat [] "group_pid" = some (fun _ => initial.pids 2)
  ig : current.regs .nat [] "in_group_dim" = some (fun _ => igd p)
  og : current.regs .nat [] "out_group_dim" = some (fun _ => ogd p)
  ba : current.regs .nat [BHW p] "batch_offset" = some (fun i => batch p initial i.1)
  fe : current.regs .nat [OF p] "output_feat_offset" = some (fun i => feat p initial i.1)
  he : current.regs .nat [BHW p] "output_height_offset" = some (fun i => oh p initial i.1)
  wi : current.regs .nat [BHW p] "output_width_offset" = some (fun i => ow p initial i.1)
  ip : current.regs .ptr [BHW p,1] "input_pointer" = some (fun i => (p.Input,inputBase p initial i.1))
  wp : current.regs .ptr [1,OF p] "weight_pointer" = some (fun i => (p.Weight,weightBase p initial i.2.1))

def HasAcc {α : Type} (p : Parameters) (a : CIndex p → α) (s : State α) : Prop :=
  s.regs .real [BHW p,OF p] "accum" = some a

def preludeExecution {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (s : State α) :
    {t : State α // run (backend M D) (prelude p) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [prelude, source, defaultBody, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
    bop, natLt, numericEq, numericLe, numericLt, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType,
    Region.cast, Option.bind, integralFloorDiv, integralMod]
  rfl

theorem preludeContext {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (s : State α) :
    Context p s (preludeExecution M D p s).val ∧
      HasAcc p (fun _ => M.literal (some .fp32) .real 0) (preludeExecution M D p s).val := by
  constructor
  · constructor <;> simp [preludeExecution, backend, igd, ogd, batch, bh, bhw, feat,
      oh, ow, inputBase, weightBase, HasAcc, bop, Broadcast.leftIndex, Broadcast.rightIndex]
    all_goals try rfl
    all_goals funext i
    all_goals simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]
  · simp [HasAcc, preludeExecution, backend]

def bodyExecution {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (initial current : State α) (h w c : Nat) (a : CIndex p → α)
    (ctx : Context p initial current) (ha : HasAcc p a current)
    (hh : current.regs .nat [] "h" = some (fun _ => h))
    (hw : current.regs .nat [] "w" = some (fun _ => w)) :
    {t : State α // run (backend M D) (cBody p)
      (current.setReg "c" .nat [] (fun _ => c)) = some t} := by
  simp only [HasAcc] at ha
  refine ⟨?_, ?_⟩
  swap
  simp [cBody, wBody, hBody, source, defaultBody, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
    bop, natLt, numericEq, numericLe, numericLt, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType,
    Region.cast, Option.bind, integralFloorDiv, integralMod,
    ctx.memory, ctx.undefined, ctx.group, ctx.ig, ctx.og, ctx.ba, ctx.fe, ctx.he,
    ctx.wi, ctx.ip, ctx.wp, ha, HasAcc, hh, hw, backend]
  rfl

theorem bodyContext {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (initial current : State α) (h w c : Nat) (a : CIndex p → α)
    (ctx : Context p initial current) (ha : HasAcc p a current)
    (hh : current.regs .nat [] "h" = some (fun _ => h))
    (hw : current.regs .nat [] "w" = some (fun _ => w)) :
    let t := (bodyExecution M D p initial current h w c a ctx ha hh hw).val
    Context p initial t ∧ HasAcc p (update M D p initial h w c a) t ∧
      t.regs .nat [] "h" = some (fun _ => h) ∧
      t.regs .nat [] "w" = some (fun _ => w) := by
  dsimp only
  refine ⟨?_, ?_, ?_, ?_⟩
  · constructor <;> simp [bodyExecution, ctx.memory, ctx.undefined, ctx.group, ctx.ig,
      ctx.og, ctx.ba, ctx.fe, ctx.he, ctx.wi, ctx.ip, ctx.wp]
  · simp [HasAcc, bodyExecution, update, inputValue, weightValue, inputOffset, weightOffset,
      inputMask, weightMask, ih, iw, backend, bop, Broadcast.leftIndex, Broadcast.rightIndex]
    change (fun i => M.binary (some .fp32) .real .add (a i)
      (M.cast (some .fp32) .fp32 .real
        (D.apply (some .fp32) {allowTF32 := some p.tf32} (batch := [])
          (sourceInputValue M p initial h w c) (weightValue M p initial h w c) i))) =
      update M D p initial h w c a
    rw [sourceInputValue_eq]
    rfl
  · simp [bodyExecution, hh]
  · simp [bodyExecution, hw]

#axiomsClean preludeExecution
#axiomsClean preludeContext
#axiomsClean bodyExecution
#axiomsClean bodyContext
end
end VeriTile.Bench.Repairs.Conv2d.KernelDefault
