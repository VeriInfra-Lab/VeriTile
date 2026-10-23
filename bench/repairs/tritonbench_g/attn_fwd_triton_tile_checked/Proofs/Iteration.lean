import bench.repairs.tritonbench_g.attn_fwd_triton_tile_checked.Proofs.Model

namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.Model
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open AttnFwdTriton.TypedExecution Graph
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def Bound (f : Fixed BD) (c : Carry BD) (s : State Expr) : Prop :=
  s.regs .nat [] "n_ctx"=some (fun _=>f.context) ∧
  s.regs .nat [] "stride_kn64"=some (fun _=>f.keyStride) ∧
  s.regs .nat [] "stride_vk64"=some (fun _=>f.valueStride) ∧
  s.regs .nat [] "start_m"=some (fun _=>f.startM) ∧
  s.regs .nat [128] "offs_m"=some f.rows ∧
  s.regs .nat [64] "offs_n"=some f.keys ∧
  s.regs .ptr [BD,64] "K_ptrs"=some f.keyBase ∧
  s.regs .ptr [] "K_scale_ptr"=some (fun _=>f.scaleBase) ∧
  s.regs .ptr [64,BD] "V_ptrs"=some f.valueBase ∧
  s.regs .ptr [128,BD] "O_block_ptr"=some f.output ∧
  s.regs .real [128,BD] "q"=some f.query ∧
  s.regs .real [] "q_scale"=some (fun _=>f.queryScale) ∧
  s.regs .real [128] "m_i"=some c.max ∧
  s.regs .real [128] "l_i"=some c.denom ∧
  s.regs .real [128,BD] "acc"=some c.accum

def LoopBound (second : Bool) (ptr : Cursor BD) (lo hi : Nat) (s : State Expr) : Prop :=
  s.regs .ptr [BD,64] (keyName second)=some ptr.key ∧
  s.regs .ptr [] (scaleName second)=some (fun _=>ptr.scale) ∧
  s.regs .ptr [64,BD] (valueName second)=some ptr.value ∧
  s.regs .nat [] (lowerName second)=some (fun _=>lo) ∧
  s.regs .nat [] (upperName second)=some (fun _=>hi)

def reads (memory : RegionName→Nat→Cell Expr) (ptr : Ptr) : Expr :=
  (memory ptr.1 ptr.2).read .real

theorem iteration (p : Config) (BD : Nat) (stage : p.stage=3)
    (second : Bool) (f : Fixed BD) (old : Carry BD) (ptr : Cursor BD)
    (lo hi cur : Nat) (s : State Expr) (hb : Bound f old s) (hl : LoopBound second ptr lo hi s) :
    ∃ t, run (engine backend) (body p BD second)
        (s.setReg "start_n" .nat [] (fun _=>cur))=some t ∧
      Bound f (update p BD (reads s.mem) f cur ptr second old) t ∧
      LoopBound second (advance f ptr) lo hi t ∧ t.mem=s.mem := by
  have zero : (0.0 : ℝ)=0 := by norm_num
  rcases hb with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14⟩
  rcases hl with ⟨hkp,hks,hvp,hlo,hhi⟩
  cases second
  all_goals simp only [keyName,scaleName,valueName,lowerName,upperName,
    Bool.false_eq_true,Bool.true_eq,↓reduceIte] at hkp hks hvp hlo hhi
  all_goals simp only [body,code,source,Bool.false_eq_true,Bool.true_eq,↓reduceIte,
    AttnFwdTriton.TypedSource.repaired,ComputeKernel.surfaceBody,Source.dynamicRange?,
    Option.map_some,Option.getD_some,beq_self_eq_true]
  all_goals simp [stage,run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,bop,
    natLt,natLe,numericLt,numericLe,Cell.read,ofFloat,toFloat,Region.cast,State.setReg,
    Bound,LoopBound,keyName,scaleName,valueName,lowerName,upperName,TileShape.axisDim,
    h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,hkp,hks,hvp,hlo,hhi,
    engine,Algebra.withDefaultPrecision,resolvePrecision,backend,
    update,scores,nextMax,probability,alpha,keyTile,valueTile,advance,reads,
    Graph.cast,nativeDType,literal,zero]
  all_goals repeat' first | constructor | funext i
  all_goals simp only [bop,Function.comp_def,online_update]
  all_goals try dsimp only [update,scores,nextMax,probability,alpha,keyTile,valueTile,advance,reads]
  all_goals simp [consume,nativeDType,Cell.read,zero]

  all_goals simp [funext_iff,scores,keyTile,valueTile,reads,nextMax,probability,alpha,Cell.read]

  all_goals delta keyTile scores probability nextMax alpha valueTile reads
  all_goals simp [Cell.read,funext_iff]

#axiomsClean iteration
end
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.Model
