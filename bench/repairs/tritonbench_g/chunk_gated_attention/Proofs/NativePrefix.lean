import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.NativeValue

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.NativePrefix
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaSimple
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (scatter_memory masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

structure Shape where
  T : Nat
  S : Nat
  BT : Nat
  BS : Nat
  BH : Nat

abbrev Index (c : Shape) := TileIndex [c.BT,c.BS]

def source (c : Shape) (base : Nat) : ComputeKernel :=
  NativeSource.cumOriginal .fp32 .fp32 "s" "o" (c.T*c.S) c.S 1 c.T c.S c.BT c.BS c.BH base

def linear (base : Nat) (s : State NativeGraph.Expr) := s.pids 0+base
def enabled (c : Shape) (base : Nat) (s : State NativeGraph.Expr) : Prop := linear base s<c.BH*(c.T/c.BT)*(c.S/c.BS)
instance (c : Shape) (base : Nat) (s : State NativeGraph.Expr) : Decidable (enabled c base s) := inferInstanceAs (Decidable (_<_))
def featureBlock (c : Shape) (base : Nat) (s : State NativeGraph.Expr) := linear base s%(c.S/c.BS)
def timeBlock (c : Shape) (base : Nat) (s : State NativeGraph.Expr) := linear base s/(c.S/c.BS)%(c.T/c.BT)
def head (c : Shape) (base : Nat) (s : State NativeGraph.Expr) := linear base s/((c.S/c.BS)*(c.T/c.BT))
def offset (c : Shape) (base : Nat) (s : State NativeGraph.Expr) (i : Index c) : Nat :=
  head c base s*(c.T*c.S)+timeBlock c base s*c.BT*c.S+featureBlock c base s*c.BS+i.1.val*c.S+i.2.1.val

def triangle (c : Shape) (i : TileIndex [c.BT,c.BT]) : NativeGraph.Expr :=
  .cast (some .fp32) .real .real (if i.2.1.val ≤ i.1.val then .literal (some .fp32) .real 1 else .literal (some .fp32) .real 0)
def input (c : Shape) (base : Nat) (s : State NativeGraph.Expr) (i : Index c) : NativeGraph.Expr :=
  .cast (some .fp32) .fp32 .real ((s.mem "s" (offset c base s i)).read .fp32)
def value (c : Shape) (base : Nat) (s : State NativeGraph.Expr) (i : Index c) : NativeGraph.Expr :=
  .cast (some .fp32) .real .fp32 (.pendingDot (some .fp32) NativeGraph.options [] c.BT c.BT c.BS
    (triangle c) (input c base s) i)

def written (c : Shape) (base : Nat) (s : State NativeGraph.Expr) : State NativeGraph.Expr :=
  if enabled c base s then (TileShape.allIndices [c.BT,c.BS]).foldl
    (fun t i=>t.write "o" (offset c base s i) (.mk .fp32 (value c base s i))) s else s

def execution (c : Shape) (base : Nat) (s : State NativeGraph.Expr) :
    {t : State NativeGraph.Expr // FP.Structural.exec (engine NativeValue.algebra) (source c base) s=some t} := by
  by_cases h : enabled c base s
  all_goals
    have condition:=h
    unfold enabled linear at condition
    refine ⟨?_,?_⟩
    swap
    simp (config := {maxSteps:=150000}) [source,NativeSource.cumOriginal,
      ComputeKernel.surfaceBody,FP.Structural.exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,
      State.setReg,integralFloorDiv,integralMod,FP.Structural.store,condition,
      NativeValue.algebra,ScalarGraph.algebra,NativeGraph.algebra]
    rfl

theorem memory (c : Shape) (base : Nat) (s : State NativeGraph.Expr) :
    (execution c base s).val.mem=(written c base s).mem := by
  by_cases h : enabled c base s
  all_goals
    have condition:=h
    unfold enabled linear at condition
    simp [execution,written,h,condition,enabled,linear,offset,head,timeBlock,featureBlock,value,triangle,input,
      scatter_memory,masked_scatter_memory,NativeGraph.options,FP.Structural.bop,
      Broadcast.leftIndex,Broadcast.rightIndex] <;>
      norm_num [triangle,input,offset,head,timeBlock,featureBlock,linear] <;> rfl

theorem source_memory (c : Shape) (base : Nat) (s : State NativeGraph.Expr) :
    ∃ t,FP.Structural.exec (engine NativeValue.algebra) (source c base) s=some t ∧ t.mem=(written c base s).mem :=
  ⟨(execution c base s).val,(execution c base s).property,memory c base s⟩

#specNonCircular value avoiding [FP.Structural.exec,FP.Structural.run]
#axiomsClean source_memory
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.NativePrefix
