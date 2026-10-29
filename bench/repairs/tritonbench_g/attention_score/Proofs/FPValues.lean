import bench.repairs.tritonbench_g.attention_score.Proofs.FPBody

namespace VeriTile.Bench.Repairs.AttentionScore.FPValues
open VeriTile Triton FP.Structural FP.ProfiledRewrite FPEngine FPContext
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def query (M : Algebra α) (B D : Nat) (p : RealSource.Parameters) (s : State α)
    (row : Nat) (i : TileIndex [B,D]) : α :=
  M.fp32Load (if p.evenM || decide (row+i.1.val<p.queryContext) then
    (s.mem p.Q (baseQ p s+(row+i.1.val)*p.queryRow+i.2.1.val*p.queryDim)).read .real
    else zero M)

def normalizer (M : Algebra α) (B : Nat) (p : RealSource.Parameters) (s : State α)
    (row : Nat) (i : Fin B) : α :=
  M.fp32Load (if row+i.val<p.queryContext then (s.mem p.M (baseM p s+row+i.val)).read .real else zero M)

def raw (M : Algebra α) (dot : Dot α) (B D : Nat) (p : RealSource.Parameters) (s : State α)
    (row : Nat) (i : TileIndex [B,B]) : α :=
  M.unary (some .fp32) .exp2
    (M.binary (some .fp32) .real .sub
      (M.binary (some .fp32) .real .mul
        (M.binary (some .fp32) .real .add (zero M)
          (dot.apply (some .fp32) .real .real .real {} (batch:=[])
            (query M B D p s row) (FPContext.key M B D p s) i)) (FPContext.scale M p))
      (normalizer M B p s row i.1))

def distance (B : Nat) (p : RealSource.Parameters) (s : State α) (row : Nat)
    (i : TileIndex [B,B]) : Int :=
  (i.1.val : Int)-i.2.1.val+row-(s.pids 0 : Int)*B+p.offset

def probability (M : Algebra α) (dot : Dot α) (B D : Nat) (p : RealSource.Parameters) (s : State α)
    (row : Nat) (i : TileIndex [B,B]) : α :=
  let windowed := if p.window then
    if (if p.complement then p.size≤distance B p s row i
      else 0≤distance B p s row i ∧ distance B p s row i<p.size) then raw M dot B D p s row i else zero M
    else raw M dot B D p s row i
  if p.evenM || decide (row+i.1.val<p.queryContext) then windowed else zero M

/-- Independent ordered primitive recurrence. This contains no interpreter,
successful execution predicate or observed output memory. -/
def update (M : Algebra α) (dot : Dot α) (B D : Nat) (p : RealSource.Parameters) (s : State α)
    (row : Nat) (acc : Fin B → α) (col : Fin B) : α :=
  M.binary (some .fp32) .real .add (acc col)
    (M.reduceSum (some .fp32) ⟨0,by simp⟩ Bool.false (probability M dot B D p s row) (col,()))

theorem body_context (M : Algebra α) (dot : Dot α) (B D : Nat) (p : RealSource.Parameters)
    (initial current : State α) (row : Nat) (acc : Fin B → α)
    (ctx : Context M B D p initial row acc current) :
    Context M B D p initial (row+B) (update M dot B D p initial row acc)
      (FPBody.executes M dot B D p initial current row acc ctx).val := by
  by_cases hm : p.evenM = Bool.true
  all_goals by_cases hw : p.window = Bool.true
  all_goals by_cases hc : p.complement = Bool.true
  all_goals
    constructor
    all_goals simp [FPBody.executes,hm,hw,hc,ctx.memory,ctx.startN,ctx.batch,ctx.head,
      ctx.query,ctx.normalizer,ctx.key,ctx.scale,ctx.output,FPContext.queryPointer,
      update,probability,raw,query,normalizer,distance,FPContext.zero,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,
      Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
    all_goals funext i
    all_goals simp [FPValues.probability,FPValues.raw,FPValues.query,FPValues.normalizer,FPValues.distance,
      FPContext.zero,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,hm,hw,hc,
      Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
    all_goals congr 2
    all_goals funext j
    all_goals simp [FPValues.probability,FPValues.raw,FPValues.query,FPValues.normalizer,FPValues.distance,
      FPContext.zero,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,hm,hw,hc,
      Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
    all_goals delta FPValues.query
    all_goals simp [hm,FPContext.zero,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
    all_goals try rfl

#specNonCircular update avoiding [FP.Structural.run,FP.Structural.exec,FPBody.executes]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScore.FPValues
