import bench.tritonbench_g.attn_fwd_triton.Proofs.FP
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

def source (p : Config) (BD : Nat) (reciprocal : Bool) : ComputeKernel :=
  (if reciprocal then TypedSource.reciprocal else TypedSource.repaired)
    p.Q p.K p.V p.QS p.KS p.Out
    p.queryBatch p.queryHead p.queryRow p.queryDim
    p.keyBatch p.keyHead p.keyRow p.keyDim
    p.valueBatch p.valueHead p.valueRow p.valueDim
    p.outputBatch p.outputHead p.outputRow p.outputDim
    p.batches p.heads p.context p.headDim 128 64 BD p.stage p.validStage

def code (p : Config) (BD : Nat) := (source p BD Bool.false).surfaceBody
def preloop (p : Config) (BD : Nat) := (code p BD).take 33
def body (p : Config) (BD : Nat) (second : Bool) :=
  ((Source.dynamicRange? (code p BD) "start_n" (if second then 1 else 0)).map (·.body)).getD []
def setup (p : Config) (BD : Nat) := ((code p BD).drop 34).take 6
def tail (p : Config) (BD : Nat) (r : Bool) := ((source p BD r).surfaceBody.drop 41)

def Ready {α : Type} (BD : Nat) (s : State α) : Prop :=
  (∃ v, s.regs .nat [] "n_ctx" = some v) ∧
  (∃ v, s.regs .nat [] "stride_kn64" = some v) ∧
  (∃ v, s.regs .nat [] "stride_vk64" = some v) ∧
  (∃ v, s.regs .nat [] "start_m" = some v) ∧
  (∃ v, s.regs .nat [128] "offs_m" = some v) ∧
  (∃ v, s.regs .nat [64] "offs_n" = some v) ∧
  (∃ v, s.regs .ptr [BD,64] "K_ptrs" = some v) ∧
  (∃ v, s.regs .ptr [] "K_scale_ptr" = some v) ∧
  (∃ v, s.regs .ptr [64,BD] "V_ptrs" = some v) ∧
  (∃ v, s.regs .ptr [128,BD] "O_block_ptr" = some v) ∧
  (∃ v, s.regs .real [128,BD] "q" = some v) ∧
  (∃ v, s.regs .real [] "q_scale" = some v) ∧
  (∃ v, s.regs .real [128] "m_i" = some v) ∧
  (∃ v, s.regs .real [128] "l_i" = some v) ∧
  (∃ v, s.regs .real [128,BD] "acc" = some v)

def keyName (second : Bool) := if second then "K_ptrs_second" else "K_ptrs_first"
def scaleName (second : Bool) := if second then "K_scale_ptr_second" else "K_scale_ptr_first"
def valueName (second : Bool) := if second then "V_ptrs_second" else "V_ptrs_first"
def lowerName (second : Bool) := if second then "second_lo" else "first_lo"
def upperName (second : Bool) := if second then "second_hi" else "first_hi"

def LoopReady {α : Type} (BD : Nat) (second : Bool) (s : State α) : Prop :=
  (∃ v, s.regs .ptr [BD,64] (keyName second)=some v) ∧
  (∃ v, s.regs .ptr [] (scaleName second)=some v) ∧
  (∃ v, s.regs .ptr [64,BD] (valueName second)=some v) ∧
  (∃ v, s.regs .nat [] (lowerName second)=some v) ∧
  (∃ v, s.regs .nat [] (upperName second)=some v)

def CoordinateReady {α : Type} (s : State α) : Prop :=
  (∃ v, s.regs .nat [] "n_ctx"=some v) ∧
  (∃ v, s.regs .nat [] "stride_kn64"=some v) ∧
  (∃ v, s.regs .nat [] "stride_vk64"=some v) ∧
  (∃ v, s.regs .nat [] "start_m"=some v) ∧
  (∃ v, s.regs .nat [] "off_hz"=some v) ∧
  (∃ v, s.regs .nat [] "off_z"=some v) ∧
  (∃ v, s.regs .nat [] "off_h"=some v) ∧
  (∃ v, s.regs .nat [] "qvk_offset"=some v) ∧
  (∃ v, s.regs .nat [] "k_offset"=some v) ∧
  (∃ v, s.regs .nat [] "v_offset"=some v) ∧
  (∃ v, s.regs .nat [] "o_offset"=some v) ∧
  (∃ v, s.regs .nat [] "q_scale_offset"=some v) ∧
  (∃ v, s.regs .nat [] "k_scale_offset"=some v)

def coordinateCode (p : Config) (BD : Nat) := (code p BD).take 13

theorem coordinates_success {α : Type} [Inhabited α] (A : Algebra α)
    (p : Config) (BD : Nat) (s : State α) :
    ∃ t, run (engine A) (coordinateCode p BD) s=some t ∧ CoordinateReady t := by
  simp only [coordinateCode,code,source,Bool.false_eq_true,↓reduceIte,
    TypedSource.repaired,ComputeKernel.surfaceBody,List.take_succ_cons,List.take_zero]
  simp [run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,bop,
    natLt,numericLt,Cell.read,ofFloat,toFloat,Region.cast,State.setReg,CoordinateReady]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.TypedExecution
