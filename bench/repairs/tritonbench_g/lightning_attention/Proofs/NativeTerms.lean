import bench.repairs.tritonbench_g.lightning_attention.Proofs.NativeInstructions
namespace VeriTile.Bench.Repairs.LightningAttention.Native
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 6000000

/-- Source expression syntax, before compiler fusion. The source's complete
Model/Grid execution instantiates this algebra; dot is a constructor, never a
numerical oracle. Casts and loads are explicit materialization boundaries. -/
inductive Expr where
  | zero
  | bits : Word → Expr
  | add : Expr → Expr → Expr
  | cast : Expr → Expr
  | load : Expr → Expr
  | dot (m k n row col : Nat) : (Fin k → Expr) → (Fin k → Expr) → Expr
instance : Inhabited Expr := ⟨.zero⟩

/-- Only zero literals, FP32 casts and addition occur in the selected model.
Unused generic algebra operations are totalized; the exact source/Model
operator inventory is checked separately. There is no half precision cast. -/
def algebra : Algebra Expr where
  literal := fun _ _ _ => .zero
  negInf := .bits 0xff800000
  binary := fun _ _ op a b => match op with | .add => .add a b | _ => .bits 0
  unary := fun _ _ _ => .bits 0
  cast := fun _ _ _ a => .cast a
  fromNat := fun _ _ => .zero
  fromInt := fun _ _ => .zero
  fp32Bits := fun b => .bits b.bits
  fp32Load := .load
  reduceMax := fun _ {_} _ _ _ _ => .bits 0
  reduceSum := fun _ {_} _ _ _ _ => .bits 0

def dotValues (l r out : FloatDType) : (batch : TileShape) → {m k n : Nat} →
    Values Expr l.toTileDType (batch++[m,k]) → Values Expr r.toTileDType (batch++[k,n]) →
    Values Expr out.toTileDType (batch++[m,n])
  | [],m,k,n,a,b,i => ofFloat out (.dot m k n i.1.val i.2.1.val
      (fun e => toFloat l (a (i.1,e,()))) (fun e => toFloat r (b (e,i.2.1,()))))
  | _::ds,_,_,_,a,b,i => dotValues l r out ds (fun j => a (i.1,j)) (fun j => b (i.1,j)) i.2

def dots : DotPrimitive Expr where
  apply := fun _ l r out _ {batch} {_} {_} {_} a b => dotValues l r out batch a b

def seedLive (plan : Plan) (m n : Nat) : Expr → Bool
  | .zero => plan.zeroSeedLive m n
  | _ => Bool.true

/-- Source-specific immediate dot/add fusion. RIGHT-dot fusion wins when
both operands are pending dots (forward output). LEFT-dot fusion covers the
three loaded-gradient updates. Completed inner deltas remain add-headed, so
the outer carry addition is retained. No floating algebraic equality is used. -/
def lower (plan : Plan) : Expr → Graph
  | .zero => .bits 0
  | .bits x => .bits x
  | .add seed (.dot m k n row col a b) =>
    dotGraph plan m k n row col (fun e => lower plan (a e)) (fun e => lower plan (b e)) (seedLive plan m n seed) (lower plan seed)
  | .add (.dot m k n row col a b) seed =>
    dotGraph plan m k n row col (fun e => lower plan (a e)) (fun e => lower plan (b e)) (seedLive plan m n seed) (lower plan seed)
  | .add a b => .add (lower plan a) (lower plan b)
  | .cast a | .load a => lower plan a
  | .dot m k n row col a b =>
    dotGraph plan m k n row col (fun e => lower plan (a e)) (fun e => lower plan (b e)) Bool.false (.bits 0)

def Pending : Expr → Prop
  | .dot .. => True
  | _ => False

theorem consume_right (plan : Plan) (m k n row col : Nat) (a b : Fin k → Expr) (seed : Expr) :
    lower plan (.add seed (.dot m k n row col a b)) =
      dotGraph plan m k n row col (fun e => lower plan (a e)) (fun e => lower plan (b e)) (seedLive plan m n seed) (lower plan seed) := by
  cases seed <;> rfl

theorem consume_left (plan : Plan) (m k n row col : Nat) (a b : Fin k → Expr)
    (seed : Expr) (h : ¬Pending seed) :
    lower plan (.add (.dot m k n row col a b) seed) =
      dotGraph plan m k n row col (fun e => lower plan (a e)) (fun e => lower plan (b e)) (seedLive plan m n seed) (lower plan seed) := by
  cases seed <;> simp_all [Pending,lower]

theorem completed_add (plan : Plan) (a b : Expr) (ha : ¬Pending a) (hb : ¬Pending b) :
    lower plan (.add a b) = .add (lower plan a) (lower plan b) := by
  cases a <;> cases b <;> simp_all [Pending,lower]

theorem cast_barrier (plan : Plan) (a b : Expr) :
    lower plan (.add (.cast a) (.cast b)) = .add (lower plan a) (lower plan b) := rfl

theorem query_delta_completed (p : Parameters) (head block : Nat) (input : Memory Expr)
    (i : TileIndex [p.E,p.D]) : ¬Pending (queryBlockDelta algebra dots p head block input i) := by
  unfold queryBlockDelta
  cases h : p.NC with
  | zero => simp [foldRange_zero,zero,algebra,Pending]
  | succ n =>
    rw [foldRange_succ]
    simp [add,algebra,Pending]

theorem key_delta_completed (p : Parameters) (head block : Nat) (input : Memory Expr)
    (i : TileIndex [p.D,p.E]) : ¬Pending (keyValueBlockDelta algebra dots p head block input i) := by
  unfold keyValueBlockDelta
  cases h : p.NC with
  | zero => simp [foldRange_zero,zero,algebra,Pending]
  | succ n =>
    rw [foldRange_succ]
    simp [add,algebra,Pending]

theorem query_carry_completed (p : Parameters) (head block : Nat) (input : Memory Expr)
    (i : TileIndex [p.E,p.D]) : ¬Pending (queryCarry algebra dots p head input block i) := by
  cases block <;> simp [queryCarry,zero,add,algebra,Pending]

theorem key_carry_completed (p : Parameters) (head block : Nat) (input : Memory Expr)
    (i : TileIndex [p.D,p.E]) : ¬Pending (keyValueCarry algebra dots p head input block i) := by
  cases block <;> simp [keyValueCarry,zero,add,algebra,Pending]

theorem query_outer_add (plan : Plan) (p : Parameters) (head block : Nat) (input : Memory Expr)
    (i : TileIndex [p.E,p.D]) :
    lower plan (queryCarry algebra dots p head input (block+1) i) =
      .add (lower plan (queryCarry algebra dots p head input block i))
        (lower plan (queryBlockDelta algebra dots p head block input i)) :=
  completed_add plan _ _ (query_carry_completed _ _ _ _ _) (query_delta_completed _ _ _ _ _)

theorem key_outer_add (plan : Plan) (p : Parameters) (head block : Nat) (input : Memory Expr)
    (i : TileIndex [p.D,p.E]) :
    lower plan (keyValueCarry algebra dots p head input (block+1) i) =
      .add (lower plan (keyValueCarry algebra dots p head input block i))
        (lower plan (keyValueBlockDelta algebra dots p head block input i)) :=
  completed_add plan _ _ (key_carry_completed _ _ _ _ _) (key_delta_completed _ _ _ _ _)

#axiomsClean consume_right
#axiomsClean consume_left
#axiomsClean completed_add
#axiomsClean query_outer_add
#axiomsClean key_outer_add
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LightningAttention.Native
