import bench.tritonbench_g.attention_score.Proofs.Reused.NativeStaging
import bench.tritonbench_g.attention_score.Proofs.Reused.FPLoop

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.NativeFormula
open VeriTile Triton FP.Structural NativeInstructions NativeStaging
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

def zero32 : Graph := .literal32 0

def query (B D : Nat) (p : RealSource.Parameters) (s : State Expr) (row : Nat) (i : TileIndex [B,D]) : Graph :=
  if p.evenM || decide (row+i.1.val<p.queryContext) then
    lower ((s.mem p.Q (FPContext.baseQ p s+(row+i.1.val)*p.queryRow+i.2.1.val*p.queryDim)).read .real)
  else zero32

def key (B D : Nat) (p : RealSource.Parameters) (s : State Expr) (i : TileIndex [D,B]) : Graph :=
  if p.evenN || decide (s.pids 0*B+i.2.1.val<p.keyContext) then
    lower ((s.mem p.K (FPContext.baseK p s+i.1.val*p.keyDim+(s.pids 0*B+i.2.1.val)*p.keyRow)).read .real)
  else zero32

def normalizer (B : Nat) (p : RealSource.Parameters) (s : State Expr) (row : Nat) (i : Fin B) : Graph :=
  if row+i.val<p.queryContext then lower ((s.mem p.M (FPContext.baseM p s+row+i.val)).read .real) else zero32

def scale (p : RealSource.Parameters) : Graph :=
  .mulRN (.literal32 p.scale) (.literal32 1.4426950408889634)

/-- The original dot seed, ordered TF32 MMA chain, scale-first fused multiply
subtract and approximate FTZ exponential, all expressed in fixed ISA nodes. -/
def raw (B D : Nat) (p : RealSource.Parameters) (s : State Expr) (row : Nat) (i : TileIndex [B,B]) : Graph :=
  .ex2ApproxFTZ (.fmaRN (scale p)
    (dot D i.1.val i.2.1.val (fun d => query B D p s row (i.1,d,()))
      (fun d => key B D p s (d,i.2.1,())) zero32)
    (.negateSign (normalizer B p s row i.1)))

def probability (B D : Nat) (p : RealSource.Parameters) (s : State Expr) (row : Nat) (i : TileIndex [B,B]) : Graph :=
  let dist : Int := (i.1.val : Int)-i.2.1.val+row-(s.pids 0 : Int)*B+p.offset
  let windowed := if p.window then
    if (if p.complement then p.size≤dist else 0≤dist ∧ dist<p.size) then raw B D p s row i else zero32
    else raw B D p s row i
  if p.evenM || decide (row+i.1.val<p.queryContext) then windowed else zero32

def contribution (B D : Nat) (p : RealSource.Parameters) (s : State Expr) (row : Nat) (col : Fin B) : Graph :=
  (planFor B).value NativeInstructions.zero
    (fun r => if h : r<B then probability B D p s row (⟨r,h⟩,col,()) else NativeInstructions.zero)
    Graph.move Graph.addRN

def update (B D : Nat) (p : RealSource.Parameters) (s : State Expr) (row : Nat) (acc : Fin B → Graph) (col : Fin B) : Graph :=
  .addRN (acc col) (contribution B D p s row col)

def accumulated (B D : Nat) (p : RealSource.Parameters) (s : State Expr) : Fin B → Graph :=
  foldRange (update B D p s) 0 p.queryContext B (fun _ => zero32)

@[simp] theorem lower_ite (P : Prop) [Decidable P] (a b : Expr) :
    lower (if P then a else b)=(if P then lower a else lower b) := by
  by_cases h : P <;> simp [h]

theorem query_lower (B D : Nat) (p : RealSource.Parameters) (s : State Expr) (row : Nat) (i : TileIndex [B,D]) :
    lower (FPValues.query algebra B D p s row i)=query B D p s row i := by
  simp [FPValues.query,query,algebra,FPContext.zero,lower,zero32]

theorem key_lower (B D : Nat) (p : RealSource.Parameters) (s : State Expr) (i : TileIndex [D,B]) :
    lower (FPContext.key algebra B D p s i)=key B D p s i := by
  simp [FPContext.key,key,algebra,FPContext.zero,lower,zero32]

theorem normalizer_lower (B : Nat) (p : RealSource.Parameters) (s : State Expr) (row : Nat) (i : Fin B) :
    lower (FPValues.normalizer algebra B p s row i)=normalizer B p s row i := by
  simp [FPValues.normalizer,normalizer,algebra,FPContext.zero,lower,zero32]

theorem raw_lower (B D : Nat) (p : RealSource.Parameters) (s : State Expr) (row : Nat) (i : TileIndex [B,B]) :
    lower (FPValues.raw algebra dotModel B D p s row i)=raw B D p s row i := by
  change Graph.ex2ApproxFTZ (Graph.fmaRN (scale p)
    (dot D i.1.val i.2.1.val
      (fun d => lower (FPValues.query algebra B D p s row (i.1,d,())))
      (fun d => lower (FPContext.key algebra B D p s (d,i.2.1,()))) zero32)
    (Graph.negateSign (lower (FPValues.normalizer algebra B p s row i.1))))=raw B D p s row i
  simp only [query_lower,key_lower,normalizer_lower,raw]

@[simp] theorem zero_lower : lower (FPContext.zero algebra)=zero32 := rfl

theorem probability_lower (B D : Nat) (p : RealSource.Parameters) (s : State Expr) (row : Nat) (i : TileIndex [B,B]) :
    lower (FPValues.probability algebra dotModel B D p s row i)=probability B D p s row i := by
  simp only [FPValues.probability,probability,FPValues.distance,lower_ite,raw_lower,zero_lower]

theorem update_lower (B D : Nat) (p : RealSource.Parameters) (s : State Expr) (row : Nat) (acc : Fin B → Expr) (col : Fin B) :
    lower (FPValues.update algebra dotModel B D p s row acc col)=
      update B D p s row (fun i => lower (acc i)) col := by
  change Graph.addRN (lower (acc col))
    ((planFor B).value NativeInstructions.zero
      (fun r => if h : r<B then lower (FPValues.probability algebra dotModel B D p s row (⟨r,h⟩,col,()))
        else NativeInstructions.zero) Graph.move Graph.addRN)=_
  simp only [probability_lower,update,contribution]

#specNonCircular accumulated avoiding [run,FP.Structural.exec,FPValues.update,FPLoop.accumulated]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.NativeFormula
