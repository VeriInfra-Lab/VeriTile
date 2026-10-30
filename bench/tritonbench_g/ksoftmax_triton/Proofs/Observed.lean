import bench.tritonbench_g.ksoftmax_triton.Proofs.Host
import bench.tritonbench_g.ksoftmax_triton.Proofs.Words
namespace VeriTile.Bench.Repairs.Ksoftmax
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
noncomputable section

/-- Only source-active input words are bound. Computed values are never rebound
as raw leaves; load barriers preserve the actual contraction graph. -/
def InputWords (depth : Host.DepthLibrary) (op : Operation) (x : TensorMetadata)
    (initial : State Graph.Expr) (left right : Nat → Graph.Word) : Prop :=
  (∀ (b : Fin (batches x)) (r : Fin (rows x)) (j : Fin (depth.depth (columns x))),
    Forward.readActive (columns x) r op.causal j.val=Bool.true →
    (initial.mem "left" (Native.flat (rows x) (columns x) b r j.val)).read .real=
      .input 0 (Native.flat (rows x) (columns x) b r j.val) (left (Native.flat (rows x) (columns x) b r j.val))) ∧
  (∀ (b : Fin (batches x)) (r : Fin (rows x)) (j : Fin (depth.depth (columns x))),
    op.readsRight=Bool.true → Forward.readActive (columns x) r op.causal j.val=Bool.true →
    (initial.mem "right" (rightAddress op (rows x) (columns x) b r j.val)).read .real=
      .input 1 (rightAddress op (rows x) (columns x) b r j.val) (right (rightAddress op (rows x) (columns x) b r j.val)))

theorem body_words (rest : Algebra Graph.Expr) (profile : Graph.Profile) (ops : Graph.Operations)
    (depth : Host.DepthLibrary) (op : Operation) (x : TensorMetadata) (rank : x.shape.length=3)
    (initial : State Graph.Expr) (left right : Nat → Graph.Word) (loads : InputWords depth op x initial left right)
    (b : Fin (batches x)) (r : Fin (rows x)) (i : Fin (depth.depth (columns x))) (hi : i.val<columns x) :
    Graph.eval ops (((Host.body (Graph.algebra rest profile) depth op x initial).mem "dst"
      (Native.flat (rows x) (columns x) b r i.val)).read .real)=
    Graph.reference rest profile ops op (rows x) (columns x) b r 0 1 left right i := by
  rw [Host.body_value _ _ _ _ rank initial b r i hi]
  unfold Graph.reference
  apply congrArg (fun c : Cell Graph.Expr => Graph.eval ops (c.read .real))
  exact gridValue_congr _ _ _ _ _ _ _ _ _ _ (loads.1 b r) (loads.2 b r) i

#axiomsClean body_words
end
end VeriTile.Bench.Repairs.Ksoftmax
