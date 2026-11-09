import bench.tritonbench_g.attn_fwd_triton.Proofs.ControlView
import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Triton.Float.Control

/-! Successful typed execution with arbitrary opaque floating arithmetic.
This layer is independent of the Real interpretation. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

structure Config where
  (Q K V QS KS Out : RegionName)
  (queryBatch queryHead queryRow queryDim : Nat)
  (keyBatch keyHead keyRow keyDim : Nat)
  (valueBatch valueHead valueRow valueDim : Nat)
  (outputBatch outputHead outputRow outputDim : Nat)
  (batches heads context headDim stage : Nat)
  validStage : stage=2 ∨ stage=3

def source (p : Config) (BD : Nat) (_reciprocal : Bool) : ComputeKernel :=
  TypedSource.repaired
    p.Q p.K p.V p.QS p.KS p.Out
    p.queryBatch p.queryHead p.queryRow p.queryDim
    p.keyBatch p.keyHead p.keyRow p.keyDim
    p.valueBatch p.valueHead p.valueRow p.valueDim
    p.outputBatch p.outputHead p.outputRow p.outputDim
    p.batches p.heads p.context p.headDim 128 64 BD p.stage p.validStage

def code (p : Config) (BD : Nat) := (source p BD Bool.false).surfaceBody

def workName (second : Bool) (role : String) : RegName :=
  match role with
  | "acc" => if second then "__veritile_statement_jit_19" else "__veritile_statement_jit_0"
  | "l_i" => if second then "__veritile_statement_jit_20" else "__veritile_statement_jit_1"
  | "m_i" => if second then "__veritile_statement_jit_21" else "__veritile_statement_jit_2"
  | "q" => if second then "__veritile_statement_jit_22" else "__veritile_statement_jit_3"
  | "q_scale" => if second then "__veritile_statement_jit_23" else "__veritile_statement_jit_4"
  | "K_ptrs" => if second then "__veritile_statement_jit_24" else "__veritile_statement_jit_5"
  | "K_scale_ptr" => if second then "__veritile_statement_jit_25" else "__veritile_statement_jit_6"
  | "V_ptrs" => if second then "__veritile_statement_jit_26" else "__veritile_statement_jit_7"
  | "start_m" => if second then "__veritile_statement_jit_27" else "__veritile_statement_jit_8"
  | "stride_kn" => if second then "__veritile_statement_jit_28" else "__veritile_statement_jit_9"
  | "stride_vk" => if second then "__veritile_statement_jit_29" else "__veritile_statement_jit_10"
  | "offs_m" => if second then "__veritile_statement_jit_30" else "__veritile_statement_jit_11"
  | "offs_n" => if second then "__veritile_statement_jit_31" else "__veritile_statement_jit_12"
  | "N_CTX" => if second then "__veritile_statement_jit_32" else "__veritile_statement_jit_13"
  | other => (if second then "__veritile_statement_jit_local_33_" else "__veritile_statement_jit_local_14_") ++ other

/-- Exact source ranges, including all argument captures and tuple returns. -/
def parts (p : Config) (BD : Nat) (second : Bool) : Source.IntRange :=
  (Source.intRange? (code p BD) (workName second "start_n")).getD
    ⟨[], .constInt 0, .constInt 0, .constNat 0, [], []⟩
def preloop (p : Config) (BD : Nat) := (parts p BD Bool.false).before
def body (p : Config) (BD : Nat) (second : Bool) := (parts p BD second).body
def finish (p : Config) (BD : Nat) (second : Bool) := (parts p BD second).after.take 6
def setup (p : Config) (BD : Nat) :=
  ((Source.intRange? (parts p BD Bool.false).after (workName Bool.true "start_n")).getD
    ⟨[], .constInt 0, .constInt 0, .constNat 0, [], []⟩).before.drop 6
def tail (p : Config) (BD : Nat) (_r : Bool) := (parts p BD Bool.true).after.drop 6

theorem code_run {α : Type} [Inhabited α] (A : Algebra α) (p : Config) (BD : Nat)
    (r : Bool) (s : State α) :
    run (engine A) (code p BD) s=run (engine A) (source p BD r).surfaceBody s := by
  rfl

def Ready {α : Type} (BD : Nat) (s : State α) : Prop :=
  (∃ v, s.regs .int [] "n_ctx" = some v) ∧
  (∃ v, s.regs .int [] "stride_kn64" = some v) ∧
  (∃ v, s.regs .int [] "stride_vk64" = some v) ∧
  (∃ v, s.regs .int [] "start_m" = some v) ∧
  (∃ v, s.regs .int [128] "offs_m" = some v) ∧
  (∃ v, s.regs .int [64] "offs_n" = some v) ∧
  (∃ v, s.regs .ptr [BD,64] "K_ptrs" = some v) ∧
  (∃ v, s.regs .ptr [] "K_scale_ptr" = some v) ∧
  (∃ v, s.regs .ptr [64,BD] "V_ptrs" = some v) ∧
  (∃ v, s.regs .ptr [128,BD] "O_block_ptr" = some v) ∧
  (∃ v, s.regs .real [128,BD] "q" = some v) ∧
  (∃ v, s.regs .real [] "q_scale" = some v) ∧
  (∃ v, s.regs .real [128] "m_i" = some v) ∧
  (∃ v, s.regs .real [128] "l_i" = some v) ∧
  (∃ v, s.regs .real [128,BD] "acc" = some v)

def keyName (second : Bool) := workName second "K_ptrs"
def scaleName (second : Bool) := workName second "K_scale_ptr"
def valueName (second : Bool) := workName second "V_ptrs"
def lowerName (second : Bool) := workName second "lo"
def upperName (second : Bool) := workName second "hi"

def lowerOp (_p : Config) (second : Bool) : Op .int [] :=
  .ref .int [] (lowerName second)

def lowerValue {α : Type} (_p : Config) (second : Bool) (s : State α) : Option (Values α .int []) :=
  s.regs .int [] (lowerName second)

theorem eval_lower {α : Type} [Inhabited α] (A : Algebra α) (p : Config) (second : Bool) (s : State α) :
    evalOp A none (lowerOp p second) s = lowerValue p second s := by
  rfl

def CapturedReady {α : Type} (BD : Nat) (second : Bool) (s : State α) : Prop :=
  (∃ v, s.regs .int [] (workName second "N_CTX")=some v) ∧
  (∃ v, s.regs .int [] (workName second "stride_kn")=some v) ∧
  (∃ v, s.regs .int [] (workName second "stride_vk")=some v) ∧
  (∃ v, s.regs .int [] (workName second "start_m")=some v) ∧
  (∃ v, s.regs .int [128] (workName second "offs_m")=some v) ∧
  (∃ v, s.regs .int [64] (workName second "offs_n")=some v) ∧
  (∃ v, s.regs .real [128,BD] (workName second "q")=some v) ∧
  (∃ v, s.regs .real [] (workName second "q_scale")=some v) ∧
  (∃ v, s.regs .real [128] (workName second "m_i")=some v) ∧
  (∃ v, s.regs .real [128] (workName second "l_i")=some v) ∧
  (∃ v, s.regs .real [128,BD] (workName second "acc")=some v)

def LoopReady {α : Type} (p : Config) (BD : Nat) (second : Bool) (s : State α) : Prop :=
  (∃ v, s.regs .ptr [BD,64] (keyName second)=some v) ∧
  (∃ v, s.regs .ptr [] (scaleName second)=some v) ∧
  (∃ v, s.regs .ptr [64,BD] (valueName second)=some v) ∧
  (∃ v, lowerValue p second s=some v) ∧
  (∃ v, s.regs .int [] (upperName second)=some v) ∧
  CapturedReady BD second s

def CoordinateReady {α : Type} (s : State α) : Prop :=
  (∃ v, s.regs .int [] "n_ctx"=some v) ∧
  (∃ v, s.regs .int [] "stride_kn64"=some v) ∧
  (∃ v, s.regs .int [] "stride_vk64"=some v) ∧
  (∃ v, s.regs .int [] "start_m"=some v) ∧
  (∃ v, s.regs .int [] "off_hz"=some v) ∧
  (∃ v, s.regs .int [] "off_z"=some v) ∧
  (∃ v, s.regs .int [] "off_h"=some v) ∧
  (∃ v, s.regs .int [] "qvk_offset"=some v) ∧
  (∃ v, s.regs .int [] "k_offset"=some v) ∧
  (∃ v, s.regs .int [] "v_offset"=some v) ∧
  (∃ v, s.regs .int [] "o_offset"=some v) ∧
  (∃ v, s.regs .int [] "q_scale_offset"=some v) ∧
  (∃ v, s.regs .int [] "k_scale_offset"=some v)

def coordinateCode (p : Config) (BD : Nat) := (code p BD).take 13

theorem coordinates_success {α : Type} [Inhabited α] (A : Algebra α)
    (p : Config) (BD : Nat) (s : State α) :
    ∃ t, run (engine A) (coordinateCode p BD) s=some t ∧ CoordinateReady t := by
  by_cases hd : p.stage=2
  all_goals simp only [coordinateCode,code,source,Bool.false_eq_true,↓reduceIte,
    TypedSource.repaired,originalKernel,hd,decide_true,decide_false,
    ComputeKernel.surfaceBody,List.take_succ_cons,List.take_zero]
  all_goals simp [run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,bop,
    natLt,numericLt,Cell.read,ofFloat,toFloat,Region.cast,State.setReg,CoordinateReady]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.TypedExecution
