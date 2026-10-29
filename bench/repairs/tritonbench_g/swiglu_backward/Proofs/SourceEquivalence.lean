import bench.repairs.tritonbench_g.swiglu_backward.Proofs.IOImplements
import bench.tritonbench_g.swiglu_backward.Proofs.FP

namespace VeriTile.Bench.Repairs.SwigluBackward.SourceEquivalence
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward
open Whole FPProof
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Exact natural pointer arithmetic, independent of every floating primitive. -/
theorem base_step {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (r : RegionName) (name : RegName) (stride : Nat) :
    step M (VeriTile.Bench.TritonBenchG.SwigluFwd.FPProof.base Bool.false r name stride) s =
      step M (VeriTile.Bench.TritonBenchG.SwigluFwd.FPProof.base Bool.true r name stride) s := by
  cases hs : s.regs .nat [] "row" with
  | none => simp [VeriTile.Bench.TritonBenchG.SwigluFwd.FPProof.base,step,
      FP.Structural.evalExpr,evalOp_unfold,hs]
  | some value =>
    simp [VeriTile.Bench.TritonBenchG.SwigluFwd.FPProof.base,step,
      FP.Structural.evalExpr,evalOp_unfold,hs,FP.Structural.numeric,FP.Structural.bop,
      Broadcast.leftIndex,Broadcast.rightIndex,Nat.mul_comm]
    apply congrArg (fun value => s.setReg name .ptr [] value)
    funext i
    change (r,0+value PUnit.unit*stride)=(r,0+stride*value PUnit.unit)
    rw [Nat.mul_comm]

theorem conditional_base_step {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (r : RegionName) (name : RegName) (stride : Nat) (R : Bool) :
    step M (.ifThen (.constBool R) [VeriTile.Bench.TritonBenchG.SwigluFwd.FPProof.base Bool.false r name stride]) s =
      step M (.ifThen (.constBool R) [VeriTile.Bench.TritonBenchG.SwigluFwd.FPProof.base Bool.true r name stride]) s := by
  cases R
  · simp [step,evalOp_unfold,run]
  · simpa [step,evalOp_unfold,run] using base_step M s r name stride

theorem prefix_run {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (X Y DOUT OUT DX DY : RegionName) (sx sy sd so sdx sdy B : Nat) (R : Bool)
    (after : List ComputeStmt) :
    run M (sourcePrefix Bool.false X Y DOUT OUT DX DY sx sy sd so sdx sdy B R ++ after) s =
      run M (sourcePrefix Bool.true X Y DOUT OUT DX DY sx sy sd so sdx sdy B R ++ after) s := by
  rw [run_append,run_append]
  apply congrArg (fun (result : Option (State α)) => result.bind (run M after))
  simp only [sourcePrefix,run,base_step,conditional_base_step]

theorem decomposition (v : Bool) (X Y DOUT OUT DX DY : RegionName)
    (sx sy sd so sdx sdy N B : Nat) (R : Bool) :
    (typedKernel .fp32 .fp32 .fp32 .fp32 v X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R).surfaceBody =
      sourcePrefix v X Y DOUT OUT DX DY sx sy sd so sdx sdy B R ++
        (typedKernel .fp32 .fp32 .fp32 .fp32 Bool.false X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R).surfaceBody.drop 9 := by
  cases v <;> rfl

theorem source_run {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (X Y DOUT OUT DX DY : RegionName) (sx sy sd so sdx sdy N B : Nat) (R : Bool) :
    exec M (kernel X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R) s =
      exec M (commutedKernel X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R) s := by
  change run M (typedKernel .fp32 .fp32 .fp32 .fp32 Bool.false X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R).surfaceBody s =
    run M (typedKernel .fp32 .fp32 .fp32 .fp32 Bool.true X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R).surfaceBody s
  rw [decomposition Bool.false,decomposition Bool.true]
  exact prefix_run M s X Y DOUT OUT DX DY sx sy sd so sdx sdy B R _

/-- Run equality lifts through the entire actual launch order. -/
theorem schedule_run {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (B : Nat) (R : Bool) (tiles : List (Nat × Nat)) (s : State α) :
    runSchedule (tiles.map (IO.invocation M c B R)) s =
      runSchedule (tiles.map (IO.commutedInvocation M c B R)) s := by
  induction tiles generalizing s with
  | nil => rfl
  | cons tile rest ih =>
    change ((IO.invocation M c B R tile).run s).bind _ =
      ((IO.commutedInvocation M c B R tile).run s).bind _
    have head : (IO.invocation M c B R tile).run s =
        (IO.commutedInvocation M c B R tile).run s :=
      source_run M _ c.X c.Y c.DOUT c.OUT c.DX c.DY
        c.sx c.sy c.sd c.so c.sdx c.sdy c.cols B R
    rw [head]
    cases (IO.commutedInvocation M c B R tile).run s <;>
      simp only [Option.bind_none,Option.bind_some,ih]

/-- The index-commuted prefix has exactly the same active backing obligations. -/
theorem candidate_safe {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (B row tile : Nat) (R : Bool) (s : State α) (bounds : RegionBounds)
    (hx : ∀ j : Fin B, tile*B+j.val < c.cols → row*c.sx+(tile*B+j.val) < bounds c.X)
    (hy : ∀ j : Fin B, tile*B+j.val < c.cols → row*c.sy+(tile*B+j.val) < bounds c.Y)
    (hd : ∀ j : Fin B, tile*B+j.val < c.cols → row*c.sd+(tile*B+j.val) < bounds c.DOUT)
    (ho : ∀ ch, enabled R ch → ∀ j : Fin B, tile*B+j.val < c.cols →
      row*c.stride ch+(tile*B+j.val) < bounds (c.output ch)) :
    TraceSafe M bounds
      (commutedKernel c.X c.Y c.DOUT c.OUT c.DX c.DY
        c.sx c.sy c.sd c.so c.sdx c.sdy c.cols B R)
      (tileState M c B row tile s) := by
  cases R
  all_goals
    simp (maxSteps := 1400000) [TraceSafe,commutedKernel,sourceKernel,typedKernel,template,
      typedLoad,typedStore,loadAddress,storeAddress,recomputedValue,
      ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
      Safety.maskActive_unfold,Safety.safeAlg_unfold,run,step,
      FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,
      store,Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,tileState,
      ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,Nat.add_assoc,Nat.mul_comm]
  · exact ⟨by simpa only [Nat.mul_comm] using hx,
      by simpa only [Nat.mul_comm] using hy,
      by simpa only [Nat.mul_comm] using hd,
      by simpa only [Nat.mul_comm] using ho .dx (by simp [enabled]),
      by simpa only [Nat.mul_comm] using ho .dy (by simp [enabled])⟩
  · exact ⟨by simpa only [Nat.mul_comm] using hx,
      by simpa only [Nat.mul_comm] using hy,
      by simpa only [Nat.mul_comm] using hd,
      by simpa only [Nat.mul_comm] using ho .dx (by simp [enabled]),
      by simpa only [Nat.mul_comm] using ho .dy (by simp [enabled]),
      by simpa only [Nat.mul_comm] using ho .out (by simp [enabled])⟩

#axiomsClean source_run
end
end VeriTile.Bench.Repairs.SwigluBackward.SourceEquivalence
