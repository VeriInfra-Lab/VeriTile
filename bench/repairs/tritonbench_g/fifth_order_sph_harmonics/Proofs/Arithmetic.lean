import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Reference

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Arithmetic
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 64000

/-- Source-order scalar graph. Precision is fp32 at every arithmetic node;
there is no fused operation or reassociation constructor. -/
inductive Expr where
  | input (index : Nat)
  | literal (value : ℝ)
  | binary (op : Binary) (left right : Expr)
  | unary (op : Unary) (operand : Expr)

def symbolic : Algebra Expr where
  literal:=fun _ _ x=>.literal x
  negInf:=.literal 0
  binary:=fun _ _ op a b=>.binary op a b
  unary:=fun _ op a=>.unary op a
  cast:=fun _ _ _=>id
  fromNat:=fun _ n=>.literal n
  fromInt:=fun _ n=>.literal n
  fp32Bits:=fun _=>.literal 0
  fp32Load:=id
  reduceMax:=fun _ _ _ _ _=>fun _=>.literal 0
  reduceSum:=fun _ _ _ _ _=>fun _=>.literal 0

def evaluate (M : Algebra α) (inputs : Nat → α) : Expr → α
  | .input i=>inputs i
  | .literal x=>M.literal (some .fp32) .real x
  | .binary op a b=>M.binary (some .fp32) .real op (evaluate M inputs a) (evaluate M inputs b)
  | .unary op a=>M.unary (some .fp32) op (evaluate M inputs a)

/-- Only the scalar primitive at this node is assumed, on the exact operands
reached by the reference child computations. Children have their own local
obligations. No source execution, index, memory, frame or result equality is a
field of this numerical domain. -/
def LocalNumerics (actual reference : Algebra α) (inputs : Nat → α) : Expr → Prop
  | .input _=>True
  | .literal x=>actual.literal (some .fp32) .real x=reference.literal (some .fp32) .real x
  | .binary op a b=>LocalNumerics actual reference inputs a ∧ LocalNumerics actual reference inputs b ∧
      actual.binary (some .fp32) .real op (evaluate reference inputs a) (evaluate reference inputs b)=
        reference.binary (some .fp32) .real op (evaluate reference inputs a) (evaluate reference inputs b)
  | .unary op a=>LocalNumerics actual reference inputs a ∧
      actual.unary (some .fp32) op (evaluate reference inputs a)=
        reference.unary (some .fp32) op (evaluate reference inputs a)

theorem evaluate_congr (actual reference : Algebra α) (inputs : Nat → α) (e : Expr)
    (laws : LocalNumerics actual reference inputs e) :
    evaluate actual inputs e=evaluate reference inputs e := by
  induction e with
  | input _=>rfl
  | literal _=>exact laws
  | binary op a b ia ib=>
    simp only [evaluate,ia laws.1,ib laws.2.1]
    exact laws.2.2
  | unary op a ih=>
    simp only [evaluate,ih laws.1]
    exact laws.2

def forwardGraph : Fin 11 → Expr :=
  Reference.forwardValues symbolic (.input 0) (.input 1) (.input 2)

def backwardGraph : Fin 3 → Expr :=
  Reference.backwardValues symbolic (.input 0) (.input 1) (.input 2)
    (fun i=>.input (3+i.val)) (fun i=>.input (14+i.val))

theorem forward_graph (M : Algebra α) (inputs : Nat → α) (channel : Fin 11) :
    evaluate M inputs (forwardGraph channel)=
      Reference.forwardValues M (inputs 0) (inputs 1) (inputs 2) channel := by
  fin_cases channel <;> simp [forwardGraph,Reference.forwardValues,
    VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP.channels,symbolic,evaluate]

theorem backward_graph (M : Algebra α) (inputs : Nat → α) (component : Fin 3) :
    evaluate M inputs (backwardGraph component)=
      Reference.backwardValues M (inputs 0) (inputs 1) (inputs 2)
        (fun i=>inputs (3+i.val)) (fun i=>inputs (14+i.val)) component := by
  fin_cases component <;> simp [backwardGraph,Reference.backwardValues,symbolic,evaluate]

#axiomsClean evaluate_congr
#axiomsClean forward_graph
#axiomsClean backward_graph
end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Arithmetic
