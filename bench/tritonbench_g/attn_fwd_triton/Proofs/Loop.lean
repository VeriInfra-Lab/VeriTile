import bench.tritonbench_g.attn_fwd_triton.Proofs.Iteration

namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.Model
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open AttnFwdTriton.TypedExecution Graph
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def lower (f : Fixed BD) (second : Bool) := if second then f.startM*128 else 0
def upper (f : Fixed BD) (second : Bool) :=
  if second then min ((f.startM+1)*128) f.context else f.startM*128

def startCursor (f : Fixed BD) (second : Bool) : Cursor BD :=
  if second then
    ⟨fun i=>((f.keyBase i).1,(f.keyBase i).2+f.keyStride*(f.startM*128)),
      (f.scaleBase.1,f.scaleBase.2+f.startM*128/64),
      fun i=>((f.valueBase i).1,(f.valueBase i).2+f.valueStride*(f.startM*128))⟩
  else ⟨f.keyBase,f.scaleBase,f.valueBase⟩

private theorem upper_projection (start context : Nat) :
    (if ((start : Int)+1)*128<(context : Int) then ((start : Int)+1)*128 else (context : Int)) =
      min (((start+1)*128 : Nat) : Int) (context : Int) := by
  omega

theorem setup_exact (p : Config) (BD : Nat) (stage : p.stage=3) (second : Bool)
    (f : Fixed BD) (old : Carry BD) (s : State Expr) (hb : Bound f old s) :
    ∃ t, run (engine backend) (setupCode p BD second) s=some t ∧
      HelperBound f old old second t ∧ LoopBound second (startCursor f second) (lower f second) (upper f second) t ∧
      t.mem=s.mem := by
  rcases hb with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14⟩
  cases second
  all_goals simp only [setupCode,setup,preloop,parts,workName,code,source,stage,show ¬(3:Nat)=2 by decide,decide_false,Source.intRange?,Option.map_some,Option.getD_some,beq_self_eq_true,Bool.false_eq_true,Bool.true_eq,↓reduceIte,
    AttnFwdTriton.TypedSource.repaired,AttnFwdTriton.originalKernel,ComputeKernel.surfaceBody,AttnFwdTriton.ControlView.compute,List.drop_succ_cons,List.drop_zero,
    List.take_succ_cons,List.take_zero]
  all_goals simp [Source.intRange?,stage,run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,bop,
    natLt,natLe,numericLt,numericLe,Cell.read,ofFloat,toFloat,Region.cast,State.setReg,
    HelperBound,CapturedBound,Bound,workName,LoopBound,keyName,scaleName,valueName,lowerName,upperName,
    h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,
    startCursor,lower,upper,-Nat.cast_add,-Nat.cast_mul,←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast,
    AttnFwdTriton.ControlView.signed_div,AttnFwdTriton.ControlView.signed_min]
  all_goals repeat first | constructor | funext i | rfl
  all_goals exact upper_projection f.startM f.context

theorem fold_exec (p : Config) (BD : Nat) (stage : p.stage=3)
    (second : Bool) (f : Fixed BD) (lo hi : Nat)
    (cur : Nat) (ptr : Cursor BD) (old : Carry BD) (s : State Expr)
    {caller : Carry BD} (hb : HelperBound f caller old second s) (hl : LoopBound second ptr lo hi s) :
    ∃ t, rangeInt (engine backend) (workName second "start_n") (cur : Int) (hi : Int) 64 (body p BD second) s=some t ∧
      HelperBound f caller (fold p BD (reads s.mem) f hi second cur ptr old) second t ∧ t.mem=s.mem := by
  generalize hk : hi-cur=k
  induction k using Nat.strong_induction_on generalizing cur ptr old s with
  | h k ih =>
    by_cases hlt : cur<hi
    · obtain ⟨t,ht,hb',hl',hm⟩ := iteration p BD stage second f old ptr lo hi cur s hb hl
      obtain ⟨u,hu,hbu,hmu⟩ := ih (hi-(cur+64)) (by omega) (cur+64) (advance f ptr)
        (update p BD (reads s.mem) f cur ptr second old) t hb' hl' rfl
      rw [hm] at hbu
      refine ⟨u,?_,?_,hmu.trans hm⟩
      · rw [AttnFwdTriton.ControlView.signed_nat_step (engine backend) (workName second "start_n") cur hi 64 (body p BD second) s (by decide) hlt,ht]
        exact hu
      · rw [fold,dif_pos hlt]
        exact hbu
    · refine ⟨s,AttnFwdTriton.ControlView.signed_nat_done (engine backend) (workName second "start_n") cur hi 64 (body p BD second) s (by omega),?_,rfl⟩
      rw [fold,dif_neg hlt]
      exact hb

/-- Python discards the second invocation's third result with `_`. -/
def returned (second : Bool) (caller result : Carry BD) : Carry BD :=
  {result with max:=if second then caller.max else result.max}

theorem finish_exact (p : Config) (BD : Nat) (stage : p.stage=3) (second : Bool)
    (f : Fixed BD) (caller current : Carry BD) (s : State Expr)
    (hb : HelperBound f caller current second s) :
    ∃ t, run (engine backend) (finish p BD second) s=some t ∧
      Bound f (returned second caller current) t ∧ t.mem=s.mem := by
  rcases hb with ⟨hb,hc⟩
  rcases hb with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14⟩
  rcases hc with ⟨hc0,hc1,hc2,hc3,hc4,hc5,hc6,hc7,hc8,hc9,hc10⟩
  cases second
  all_goals simp [workName] at hc8 hc9 hc10
  all_goals simp only [finish,parts,code,source,AttnFwdTriton.TypedSource.repaired,
    AttnFwdTriton.originalKernel,stage,show ¬(3:Nat)=2 by decide,decide_false,
    workName,Bool.false_eq_true,Bool.true_eq,↓reduceIte,ComputeKernel.surfaceBody,
    Source.intRange?,Option.map_some,Option.getD_some,beq_self_eq_true,
    List.take_succ_cons,List.take_zero]
  all_goals simp [Source.intRange?,run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,bop,Cell.read,ofFloat,toFloat,
    Region.cast,State.setReg,Bound,returned,
    engine,Algebra.withDefaultPrecision,resolvePrecision,backend,Graph.cast,nativeDType,consume,
    h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,hc8,hc9,hc10]

theorem loop_exact (p : Config) (BD : Nat) (stage : p.stage=3)
    (second : Bool) (f : Fixed BD) (old : Carry BD) (s : State Expr)
    (hb : HelperBound f old old second s)
    (hl : LoopBound second (startCursor f second) (lower f second) (upper f second) s) :
    ∃ t, run (engine backend) (loopCode p BD second) s=some t ∧
      Bound f (returned second old (fold p BD (reads s.mem) f (upper f second) second
        (lower f second) (startCursor f second) old)) t ∧ t.mem=s.mem := by
  obtain ⟨t,ht,hb',hm⟩ := fold_exec p BD stage second f (lower f second) (upper f second)
    (lower f second) (startCursor f second) old s hb hl
  obtain ⟨u,hu,hbu,hmu⟩ := finish_exact p BD stage second f old _ t hb'
  refine ⟨u,?_,hbu,hmu.trans hm⟩
  rcases hl with ⟨_,_,_,hlo,hhi⟩
  have loEval : evalOp (engine backend) none (lowerOp p second) s =
      some (fun _ => (lower f second : Int)) := by
    rw [eval_lower]
    exact hlo
  simp only [loopCode,run,step]
  rw [loEval]
  simp only [evalOp_unfold,hhi,bind,Option.bind]
  rw [ht]
  exact hu

#axiomsClean setup_exact
#axiomsClean fold_exec
#axiomsClean finish_exact
#axiomsClean loop_exact
end
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.Model
