import bench.tritonbench_g.rms_matmul_rbe.Proofs.Arithmetic

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 10000000
set_option linter.unusedSimpArgs false

def storing (outputFP8 : Bool) (p : Arguments) :=
  (epilogue Bool.false outputFP8 p).drop 4

theorem storing_source (fp8 outputFP8 : Bool) (p : Arguments) :
    (epilogue fp8 outputFP8 p).drop 4=storing outputFP8 p := by
  cases fp8 <;> cases outputFP8 <;> rfl

def outputOffset (p : Arguments) (pb pm pn : Nat) (i : TileIndex [16,64]) : Nat :=
  pb*p.sob+(pm*16+i.1.val)*p.som+(pn*64+i.2.1.val)*p.son

def outputActive (p : Arguments) (pm pn : Nat) (i : TileIndex [16,64]) : Prop :=
  pm*16+i.1.val<p.M ∧ pn*64+i.2.1.val<p.N

instance (p : Arguments) (pm pn : Nat) : DecidablePred (outputActive p pm pn) :=
  fun _ => inferInstanceAs (Decidable (_ ∧ _))

def outputCell (a : Arithmetic α) (outputFP8 : Bool) (v : α) : Cell α :=
  if outputFP8 then .mk .f8e5 (a.cast .real .f8e5 v)
  else .mk .fp16 (a.cast .real .fp16 v)

/-- Independent masked write over the actual existing destination. No
allocation or successful-store assumption appears in this target. -/
def tileWrite (a : Arithmetic α) (outputFP8 : Bool) (p : Arguments)
    (pb pm pn : Nat) (values : Values α .real [16,64]) (s : State α) : State α :=
  (TileShape.allIndices [16,64]).foldl (fun t i =>
    if outputActive p pm pn i then
      t.write p.OUT (outputOffset p pb pm pn i) (outputCell a outputFP8 (values i))
    else t) s

def storingExecution {α : Type} [Inhabited α] (a : Arithmetic α)
    (outputFP8 : Bool) (p : Arguments) (s : State α) (pb pm pn : Nat)
    (accumulator : Values α .real [16,64])
    (hb : s.regs .nat [] "pid_batch"=some (fun _ => pb))
    (hm : s.regs .nat [] "pid_m"=some (fun _ => pm))
    (hn : s.regs .nat [] "pid_n"=some (fun _ => pn))
    (ha : s.regs .real [16,64] "accumulator"=some accumulator) :
    {next : State α // run a.model (storing outputFP8 p) s=some next} := by
  cases outputFP8
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [storing,epilogue,source,TypedSource.kernel,TypedSource.halfHalf,TypedSource.halfFP8,TypedSource.fp8Half,TypedSource.fp8FP8,ComputeKernel.surfaceBody,run,
      FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,
      numeric,bop,natLt,numericLt,ComputeDType.eraseDType,ofFloat,toFloat,
      Region.cast,Option.bind,integralFloorDiv,integralMod,Arithmetic.model,hb,hm,hn,ha,
      store,Broadcast.leftIndex,Broadcast.rightIndex]
    rfl

#axiomsClean storing_source
#guard_msgs (drop info) in
#auditModuleAxioms
#specNonCircular tileWrite avoiding [storingExecution,run,store]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
