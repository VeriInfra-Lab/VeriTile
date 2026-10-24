import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelGeometry
import VeriTile.Triton.Float.UnsignedCast

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel
open VeriTile.Triton FP.Structural
open Math
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- Ordered scan topology. A fused node consumes the original product operands
through the scalar graph carrier, never an already-rounded product word. -/
inductive SumTree (n : Nat) where
  | leaf (index : Fin n)
  | add (left right : SumTree n)
  | fused (index : Fin n) (addend : SumTree n)

def SumTree.leaves {n : Nat} : SumTree n → List Nat
  | .leaf i => [i.val]
  | .add l r => l.leaves ++ r.leaves
  | .fused i r => [i.val] ++ r.leaves

def SumTree.eval {n : Nat} (add fused : α → α → α) (x : Fin n → α) : SumTree n → α
  | .leaf i => x i
  | .add l r => add (l.eval add fused x) (r.eval add fused x)
  | .fused i r => fused (x i) (r.eval add fused x)

structure PrefixPlan (h n : Nat) where
  tree : Fin h → Fin n → SumTree n
  leaves : ∀ i j, (tree i j).leaves.Perm (List.range (j.val+1))

/-- Only reached scalar comparison/minimum and one row-scan interpretation.
No field supplies a kernel result, a memory equation or a successful run. -/
structure Primitives (α : Type) (c : Geometry) where
  model : Algebra α
  lessEqual : α → α → Bool
  minimum : α → α → α
  le_support : model.compareLe (some .fp32) .real = some lessEqual
  minimum_support : model.minimum (some .fp32) .real = some minimum
  fused : α → α → α
  plan : PrefixPlan c.BH c.BC
  scan_support : ∀ x : Values α .real [c.BH,c.BC],
    model.scan (some .fp32) (shape := [c.BH,c.BC]) .sum (1 : Fin 2) .forward x =
      some (fun i => (plan.tree i.1 i.2.1).eval (model.binary (some .fp32) .real .add) fused
        (fun k => x (i.1,k,PUnit.unit)))

@[simp] theorem scatter_regs {α ι : Type} (region : RegionName) (off : ι → Nat)
    (value : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (lanes : List ι) (s : State α) :
    (lanes.foldl (fun t i => if active i then t.write region (off i) (value i) else t) s).regs = s.regs := by
  induction lanes generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons]
    by_cases hi : active i
    · simpa only [hi,if_true,State.write] using ih (s.write region (off i) (value i))
    · simpa only [hi,if_false] using ih s

/-- Successful execution of every statement in the authoritative quotation.
Runtime bounds enter only through initial registers. No clamp substitution
or assumed output equation occurs in this construction. -/
def sourceRun {α : Type} [Inhabited α] (c : Geometry) (p : Primitives α c)
    (hasBias useSoftplus : Bool) (lower upper : α) (l : Launch) (s : State α) :
    {v : State α // FP.Structural.exec (engine p.model) (kernel c hasBias useSoftplus)
      (tileState l (bindBounds s lower upper)) = some v} := by
  cases hasBias <;> cases useSoftplus
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp (maxSteps := 1200000) [FP.Structural.exec,kernel,source,
      ComputeKernel.surfaceBody,run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,store,
      Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,bop,natLt,natLe,numericLt,numericLe,tileState,bindBounds,ofFloat,toFloat,
      integralMod,IntegerExpr.same_left,IntegerExpr.same_right,Op.natToUnsigned,
      TileShape.axisDim,Region.cast,State.setReg,Geometry.layout,scatter_regs,
      p.le_support,p.minimum_support,p.scan_support]
    try rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel
