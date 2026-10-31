import bench.tritonbench_g.attn_fwd_triton.Proofs.TypedPreservation

namespace VeriTile.Bench.Repairs.AttnFwdTriton.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedPreservation
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def outputAddress (p : Config) (BD pm ph : Nat) (i : TileIndex [128,BD]) : Nat :=
  (ph/p.heads)*p.outputBatch+(ph%p.heads)*p.outputHead +
    (pm*128+i.1.val)*p.outputRow+i.2.1.val*p.outputDim

def outputActive (p : Config) (BD pm : Nat) (i : TileIndex [128,BD]) : Bool :=
  decide (pm*128+i.1.val<p.context) && decide (i.2.1.val<p.headDim)

def AddressReady {α : Type} (p : Config) (BD pm ph : Nat) (s : State α) : Prop :=
  s.regs .int [] "n_ctx"=some (fun _ => (p.context : Int)) ∧
  s.regs .int [128] "offs_m"=some (fun i => ((pm*128+i.1.val : Nat) : Int)) ∧
  s.regs .ptr [128,BD] "O_block_ptr"=some (fun i => (p.Out,outputAddress p BD pm ph i))

def middleCode (p : Config) (BD : Nat) := ((code p BD).drop 22).take 19

theorem middle_pure (p : Config) (BD : Nat) :
    Pure ["n_ctx","offs_m","O_block_ptr"] (middleCode p BD) := by
  cases hstage : decide (p.stage=2)
  all_goals simp only [middleCode,code,source,Bool.false_eq_true,↓reduceIte,
    TypedSource.repaired,ComputeKernel.surfaceBody,ControlView.compute,List.drop_succ_cons,List.drop_zero,
    List.take_succ_cons,List.take_zero,hstage,Bool.true_eq,List.take_append,List.take_succ_cons,List.take_zero,List.nil_append]
  all_goals repeat first | exact Pure.nil | apply Pure.assign | apply Pure.branch | apply Pure.when | apply Pure.range | apply Pure.signedRange | decide

theorem header (A : Algebra α) [Inhabited α] (p : Config) (BD : Nat) (s : State α) :
    ∃ t, run (engine A) ((code p BD).take 22) s=some t ∧
      PointerReady BD t ∧ AddressReady p BD (s.pids 0) (s.pids 1) t ∧ t.mem=s.mem := by
  simp only [code,source,Bool.false_eq_true,↓reduceIte,
    TypedSource.repaired,ComputeKernel.surfaceBody,ControlView.compute,List.take_succ_cons,List.take_zero]
  simp [run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,bop,
    natLt,numericLt,Cell.read,ofFloat,toFloat,Region.cast,State.setReg,PointerReady,
    AddressReady,outputAddress]
  constructor <;> funext i <;> simp [bop,Int.natCast_nonneg,not_lt.mpr (Int.natCast_nonneg (s.pids 1))]
  all_goals rw [←Int.emod_def,←Int.natCast_emod]
  all_goals simp only [←Int.natCast_ediv,←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast]
  all_goals norm_cast

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.TypedExecution
