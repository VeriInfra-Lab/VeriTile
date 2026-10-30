import bench.repairs.tritonbench_g.attn_fwd_triton_tile_checked.Proofs.Iteration

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

theorem setup_exact (p : Config) (BD : Nat) (stage : p.stage=3) (second : Bool)
    (f : Fixed BD) (old : Carry BD) (s : State Expr) (hb : Bound f old s) :
    ∃ t, run (engine backend) (setupCode p BD second) s=some t ∧
      Bound f old t ∧ LoopBound second (startCursor f second) (lower f second) (upper f second) t ∧
      t.mem=s.mem := by
  rcases hb with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14⟩
  cases second
  all_goals simp only [setupCode,code,source,Bool.false_eq_true,Bool.true_eq,↓reduceIte,
    AttnFwdTriton.TypedSource.repaired,ComputeKernel.surfaceBody,List.drop_succ_cons,List.drop_zero,
    List.take_succ_cons,List.take_zero]
  all_goals simp [stage,run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,bop,
    natLt,natLe,numericLt,numericLe,Cell.read,ofFloat,toFloat,Region.cast,State.setReg,
    Bound,LoopBound,keyName,scaleName,valueName,lowerName,upperName,
    h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,
    startCursor,lower,upper]
  all_goals repeat first | constructor | funext i | rfl
  by_cases hlt : (f.startM+1)*128<f.context
  · rw [if_pos hlt,Nat.min_eq_left (Nat.le_of_lt hlt)]
  · rw [if_neg hlt,Nat.min_eq_right (Nat.le_of_not_gt hlt)]

theorem fold_exec (p : Config) (BD : Nat) (stage : p.stage=3)
    (second : Bool) (f : Fixed BD) (lo hi : Nat)
    (cur : Nat) (ptr : Cursor BD) (old : Carry BD) (s : State Expr)
    (hb : Bound f old s) (hl : LoopBound second ptr lo hi s) :
    ∃ t, range (engine backend) "start_n" cur hi 64 (body p BD second) s=some t ∧
      Bound f (fold p BD (reads s.mem) f hi second cur ptr old) t ∧ t.mem=s.mem := by
  generalize hk : hi-cur=k
  induction k using Nat.strong_induction_on generalizing cur ptr old s with
  | h k ih =>
    by_cases hlt : cur<hi
    · obtain ⟨t,ht,hb',hl',hm⟩ := iteration p BD stage second f old ptr lo hi cur s hb hl
      obtain ⟨u,hu,hbu,hmu⟩ := ih (hi-(cur+64)) (by omega) (cur+64) (advance f ptr)
        (update p BD (reads s.mem) f cur ptr second old) t hb' hl' rfl
      rw [hm] at hbu
      refine ⟨u,?_,?_,hmu.trans hm⟩
      · rw [range_step (engine backend) "start_n" cur hi 64 (body p BD second) s (by decide) hlt,ht]
        exact hu
      · rw [fold,dif_pos hlt]
        exact hbu
    · refine ⟨s,range_done (engine backend) "start_n" cur hi 64 (body p BD second) s (by omega),?_,rfl⟩
      rw [fold,dif_neg hlt]
      exact hb

theorem loop_exact (p : Config) (BD : Nat) (stage : p.stage=3)
    (second : Bool) (f : Fixed BD) (old : Carry BD) (s : State Expr)
    (hb : Bound f old s)
    (hl : LoopBound second (startCursor f second) (lower f second) (upper f second) s) :
    ∃ t, run (engine backend) (loopCode p BD second) s=some t ∧
      Bound f (fold p BD (reads s.mem) f (upper f second) second
        (lower f second) (startCursor f second) old) t ∧ t.mem=s.mem := by
  obtain ⟨t,ht,hb',hm⟩ := fold_exec p BD stage second f (lower f second) (upper f second)
    (lower f second) (startCursor f second) old s hb hl
  refine ⟨t,?_,hb',hm⟩
  rcases hl with ⟨_,_,_,hlo,hhi⟩
  simpa [loopCode,run,step,evalOp_unfold,hlo,hhi] using ht

#axiomsClean setup_exact
#axiomsClean fold_exec
#axiomsClean loop_exact
end
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.Model
