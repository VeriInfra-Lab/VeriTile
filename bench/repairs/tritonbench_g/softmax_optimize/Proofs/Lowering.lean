import bench.repairs.tritonbench_g.softmax_optimize.Proofs.TypedSource
import bench.repairs.tritonbench_g.softmax_optimize.Proofs.RowGraph

/-! Ordinary local compiler contract, specialized to production FP32, sm90,
warp32, num_warps=4, num_stages=3 and enabled fusion. Its laws concern only
local expression lowering. No law mentions a loop, memory, successful kernel
execution, a row result, a grid or a host return. Contracts are supplied for
actual typed source fragments. In particular weighted reduction consumes pairs
of expression operands, before product rounding, so FMA remains expressible. -/
namespace VeriTile.Bench.Repairs.SoftmaxOptimize.Lowering
open VeriTile.Triton Graph RowGraph

structure Production where
  storage : FloatDType := .fp32
  accumulation : FloatDType := .fp32
  output : FloatDType := .fp32
  fusion : Bool := Bool.true
  warpSize : Nat := 32
  numWarps : Nat := 4
  numStages : Nat := 3
  architecture : Nat := 90

def production : Production := {}

inductive ScalarABI where | i32 | i64
  deriving DecidableEq, Repr

def scalarValue (abi : ScalarABI) (N : Nat) : Int :=
  match abi with
  | .i32 => (BitVec.ofNat 32 N).toInt
  | .i64 => (BitVec.ofNat 64 N).toInt

structure Compiler (B : Nat) where
  scalarABI : ScalarABI
  configuration : Production
  configuration_law : configuration = production
  profile : Profile B
  update : List ComputeStmt → (Expr Nat × Expr Nat) → Expr Nat → (Expr Nat × Expr Nat)
  maximum : List ComputeStmt → (Fin B → Expr Nat) → Fin B → Expr Nat
  weighted : List ComputeStmt → (Fin B → Expr Nat × Expr Nat) → Fin B → Expr Nat
  normalize : List ComputeStmt → Expr Nat → Expr Nat → Expr Nat → Expr Nat
  update_law : ∀ state score,
    update (TypedSource.guarded B) state score = Graph.update profile.updateOrder state score
  maximum_law : ∀ values lane,
    maximum (TypedSource.maximum B) values lane = (profile.maxPlan lane).expression values
  weighted_law : ∀ values lane,
    weighted (TypedSource.weighted B) values lane = (profile.denominators.plan lane).expression values
  normalize_law : ∀ score maximum denominator,
    normalize (TypedSource.normalize B) score maximum denominator =
      .divide (.exponential (.subtract score maximum)) denominator

/-- Nonempty local implementation; there is no fail-all or output oracle.
Unrecognized fragments are outside this specialized interface's support. -/
def graphCompiler (profile : Profile B) : Compiler B where
  scalarABI := .i64
  configuration := production
  configuration_law := rfl
  profile := profile
  update := fun _ => Graph.update profile.updateOrder
  maximum := fun _ values lane => (profile.maxPlan lane).expression values
  weighted := fun _ values lane => (profile.denominators.plan lane).expression values
  normalize := fun _ score maximum denominator => .divide (.exponential (.subtract score maximum)) denominator
  update_law := fun _ _ => rfl
  maximum_law := fun _ _ => rfl
  weighted_law := fun _ _ => rfl
  normalize_law := fun _ _ _ => rfl

def fullPass (compiler : Compiler B) (N lane count : Nat) : Expr Nat × Expr Nat :=
  (List.range count).foldl
    (fun state index => compiler.update (TypedSource.guarded B) state (score N B index lane))
    (.negInf, .zero)

def firstPass (compiler : Compiler B) (N lane : Nat) : Expr Nat × Expr Nat :=
  compiler.update (TypedSource.guarded B)
    (fullPass compiler N lane (Config.fullTiles N B)) (score N B (Config.fullTiles N B) lane)

theorem fullPass_graph (compiler : Compiler B) (N lane count : Nat) :
    fullPass compiler N lane count = laneAfter compiler.profile.updateOrder N B lane count := by
  unfold fullPass laneAfter
  congr 1
  funext state index
  exact compiler.update_law state _

theorem firstPass_graph (compiler : Compiler B) (N lane : Nat) :
    firstPass compiler N lane = twoLoops compiler.profile.updateOrder N B lane := by
  rw [firstPass, compiler.update_law, fullPass_graph]
  rfl

def reducedMax (compiler : Compiler B) (N : Nat) (lane : Fin B) : Expr Nat :=
  compiler.maximum (TypedSource.maximum B) (fun i => (firstPass compiler N i).1) lane

def weightedTerms (compiler : Compiler B) (N : Nat) (lane : Fin B) : Expr Nat × Expr Nat :=
  (.exponential (.subtract (firstPass compiler N lane).1 (reducedMax compiler N lane)),
    (firstPass compiler N lane).2)

def reducedSum (compiler : Compiler B) (N : Nat) (lane : Fin B) : Expr Nat :=
  compiler.weighted (TypedSource.weighted B) (weightedTerms compiler N) lane

def outputExpr (compiler : Compiler B) (N column : Nat) (lane : Fin B) : Expr Nat :=
  compiler.normalize (TypedSource.normalize B) (.input column)
    (reducedMax compiler N lane) (reducedSum compiler N lane)

theorem reducedMax_graph (compiler : Compiler B) (N : Nat) (lane : Fin B) :
    reducedMax compiler N lane = maxima compiler.profile N lane := by
  simp only [reducedMax, compiler.maximum_law, firstPass_graph, maxima]

theorem reducedSum_graph (compiler : Compiler B) (N : Nat) (lane : Fin B) :
    reducedSum compiler N lane = denominator compiler.profile N lane := by
  simp only [reducedSum, compiler.weighted_law, denominator]
  congr 1
  funext i
  simp only [weightedTerms, factors, firstPass_graph, reducedMax_graph]

/-- Universal loop/expression composition, derived from local compiler laws.
It retains separate maxima/denominators for each output lane and exact FMA edges.
All FP word interpretations are admitted; this theorem uses no numerical law. -/
theorem output_graph (compiler : Compiler B) (N column : Nat) (lane : Fin B) :
    outputExpr compiler N column lane = output compiler.profile N column lane := by
  rw [outputExpr, compiler.normalize_law, reducedMax_graph, reducedSum_graph]
  rfl

/-- The compiler's four local inputs are the exact typed fragments in both
source input/output loops. The source decomposition also fixes all loop bounds,
loads and stores; it is independent of the numerical primitive interpretation. -/
theorem typed_binding (Y X : RegionName) (M N B : Nat) :
    (TypedSource.fullInput X N B).drop 4 = TypedSource.guarded B ∧
    (TypedSource.tailInput X N B).drop 5 = TypedSource.guarded B ∧
    ((TypedSource.fullOutput Y X N B).drop 4).take 2 = TypedSource.normalize B ∧
    ((TypedSource.tailOutput Y X N B).drop 5).take 2 = TypedSource.normalize B :=
  ⟨TypedSource.full_guarded X N B, TypedSource.tail_guarded X N B,
    (TypedSource.output_normalize Y X N B).1, (TypedSource.output_normalize Y X N B).2⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SoftmaxOptimize.Lowering
