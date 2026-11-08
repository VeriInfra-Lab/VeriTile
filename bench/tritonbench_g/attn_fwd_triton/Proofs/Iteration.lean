import bench.tritonbench_g.attn_fwd_triton.Proofs.Model

namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.Model
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open AttnFwdTriton.TypedExecution Graph
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def Bound (f : Fixed BD) (c : Carry BD) (s : State Expr) : Prop :=
  s.regs .int [] "n_ctx"=some (fun _=>(f.context : Int)) ∧
  s.regs .int [] "stride_kn64"=some (fun _=>(f.keyStride : Int)) ∧
  s.regs .int [] "stride_vk64"=some (fun _=>(f.valueStride : Int)) ∧
  s.regs .int [] "start_m"=some (fun _=>(f.startM : Int)) ∧
  s.regs .int [128] "offs_m"=some (fun i => (f.rows i : Int)) ∧
  s.regs .int [64] "offs_n"=some (fun i => (f.keys i : Int)) ∧
  s.regs .ptr [BD,64] "K_ptrs"=some f.keyBase ∧
  s.regs .ptr [] "K_scale_ptr"=some (fun _=>f.scaleBase) ∧
  s.regs .ptr [64,BD] "V_ptrs"=some f.valueBase ∧
  s.regs .ptr [128,BD] "O_block_ptr"=some f.output ∧
  s.regs .real [128,BD] "q"=some f.query ∧
  s.regs .real [] "q_scale"=some (fun _=>f.queryScale) ∧
  s.regs .real [128] "m_i"=some c.max ∧
  s.regs .real [128] "l_i"=some c.denom ∧
  s.regs .real [128,BD] "acc"=some c.accum

/-- Caller registers survive a hygienic helper invocation. The current
online state lives in the helper's captured parameters until its return. -/
def CapturedBound (f : Fixed BD) (c : Carry BD) (second : Bool) (s : State Expr) : Prop :=
  s.regs .int [] (workName second "N_CTX")=some (fun _=>(f.context : Int)) ∧
  s.regs .int [] (workName second "stride_kn")=some (fun _=>(f.keyStride : Int)) ∧
  s.regs .int [] (workName second "stride_vk")=some (fun _=>(f.valueStride : Int)) ∧
  s.regs .int [] (workName second "start_m")=some (fun _=>(f.startM : Int)) ∧
  s.regs .int [128] (workName second "offs_m")=some (fun i=>(f.rows i : Int)) ∧
  s.regs .int [64] (workName second "offs_n")=some (fun i=>(f.keys i : Int)) ∧
  s.regs .real [128,BD] (workName second "q")=some f.query ∧
  s.regs .real [] (workName second "q_scale")=some (fun _=>f.queryScale) ∧
  s.regs .real [128] (workName second "m_i")=some c.max ∧
  s.regs .real [128] (workName second "l_i")=some c.denom ∧
  s.regs .real [128,BD] (workName second "acc")=some c.accum

def HelperBound (f : Fixed BD) (caller current : Carry BD) (second : Bool) (s : State Expr) : Prop :=
  Bound f caller s ∧ CapturedBound f current second s

def LoopBound (second : Bool) (ptr : Cursor BD) (lo hi : Nat) (s : State Expr) : Prop :=
  s.regs .ptr [BD,64] (keyName second)=some ptr.key ∧
  s.regs .ptr [] (scaleName second)=some (fun _=>ptr.scale) ∧
  s.regs .ptr [64,BD] (valueName second)=some ptr.value ∧
  s.regs .int [] (lowerName second)=some (fun _=>(lo : Int)) ∧
  s.regs .int [] (upperName second)=some (fun _=>(hi : Int))

def reads (memory : RegionName→Nat→Cell Expr) (ptr : Ptr) : Expr :=
  (memory ptr.1 ptr.2).read .real

private theorem tail_comparison (key context cur : Nat) :
    ((key : Int)<(context : Int)-(cur : Int)) ↔ key<context-cur := by
  omega

theorem iteration (p : Config) (BD : Nat) (stage : p.stage=3)
    (second : Bool) (f : Fixed BD) (old : Carry BD) (ptr : Cursor BD)
    (lo hi cur : Nat) (s : State Expr) {caller : Carry BD} (hb : HelperBound f caller old second s) (hl : LoopBound second ptr lo hi s) :
    ∃ t, run (engine backend) (body p BD second)
        (s.setReg (workName second "start_n") .int [] (fun _=>(cur : Int)))=some t ∧
      HelperBound f caller (update p BD (reads s.mem) f cur ptr second old) second t ∧
      LoopBound second (advance f ptr) lo hi t ∧ t.mem=s.mem := by
  have zero : (0.0 : ℝ)=0 := by norm_num
  rcases hb with ⟨hb,hc⟩
  rcases hc with ⟨hc0,hc1,hc2,hc3,hc4,hc5,hc6,hc7,hc8,hc9,hc10⟩
  rcases hb with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14⟩
  rcases hl with ⟨hkp,hks,hvp,hlo,hhi⟩
  cases second
  all_goals simp [workName,keyName,scaleName,valueName,lowerName,upperName,
    Bool.false_eq_true,Bool.true_eq,↓reduceIte] at hkp hks hvp hlo hhi hc0 hc1 hc2 hc3 hc4 hc5 hc6 hc7 hc8 hc9 hc10
  all_goals simp only [body,parts,code,source,stage,show ¬(3:Nat)=2 by decide,decide_false,workName,Bool.false_eq_true,Bool.true_eq,↓reduceIte,
    AttnFwdTriton.TypedSource.repaired,AttnFwdTriton.originalKernel,ComputeKernel.surfaceBody,AttnFwdTriton.ControlView.compute,Source.intRange?,
    Option.map_some,Option.getD_some,beq_self_eq_true,List.drop_succ_cons,List.drop_zero]
  all_goals simp [stage,Source.intRange?,AttnFwdTriton.ControlView.compute,run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,bop,
    natLt,natLe,numericLt,numericLe,Cell.read,ofFloat,toFloat,Region.cast,State.setReg,
    HelperBound,CapturedBound,Bound,workName,LoopBound,keyName,scaleName,valueName,lowerName,upperName,TileShape.axisDim,
    h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,hkp,hks,hvp,hlo,hhi,hc0,hc1,hc2,hc3,hc4,hc5,hc6,hc7,hc8,hc9,hc10,
    engine,Algebra.withDefaultPrecision,resolvePrecision,backend,
    update,scores,nextMax,probability,alpha,keyTile,valueTile,advance,reads,
    Graph.cast,nativeDType,literal,zero,-Nat.cast_add,-Nat.cast_mul,←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast,
    AttnFwdTriton.ControlView.signed_tail,tail_comparison]
  all_goals repeat' first | constructor | funext i
  all_goals simp only [bop,Function.comp_def,online_update]
  all_goals try dsimp only [update,scores,nextMax,probability,alpha,keyTile,valueTile,advance,reads]
  all_goals simp [consume,nativeDType,Cell.read,zero]

  all_goals simp [funext_iff,scores,keyTile,valueTile,reads,nextMax,probability,alpha,Cell.read]

  all_goals delta keyTile scores probability nextMax alpha valueTile reads
  all_goals simp [Cell.read,funext_iff]

  all_goals simp only [tail_comparison]

#axiomsClean iteration
end
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.Model
