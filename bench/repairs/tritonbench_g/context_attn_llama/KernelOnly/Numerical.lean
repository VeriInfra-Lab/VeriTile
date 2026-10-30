import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.NativeGraph

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.Numerical
open NativeGraph
noncomputable section

/-- All-word laws for eight scalar instruction families and the fixed k8
single-output WGMMA family. No arbitrary-size dot/reduction, final accumulator,
source success, logical output, address, memory frame or Real equality law. -/
structure ScalarLaws (actual reference : ScalarOps) : Prop where
  literal32 : ∀ x, actual.literal32 x=reference.literal32 x
  add : ∀ x y, actual.add x y=reference.add x y
  sub : ∀ x y, actual.sub x y=reference.sub x y
  mul : ∀ x y, actual.mul x y=reference.mul x y
  div : ∀ x y, actual.div x y=reference.div x y
  max : ∀ x y, actual.max x y=reference.max x y
  exp2 : ∀ x, actual.exp2 x=reference.exp2 x
  fmaRN : ∀ x y z, actual.fmaRN x y z=reference.fmaRN x y z
  wgmmaTF32 : ∀ width row col a b accumulate seed,
    actual.wgmmaTF32 width row col a b accumulate seed=reference.wgmmaTF32 width row col a b accumulate seed

theorem ScalarLaws.operations_eq (actual reference : ScalarOps)
    (h : ScalarLaws actual reference) : actual=reference := by
  cases actual
  cases reference
  simp only [ScalarOps.mk.injEq]
  exact ⟨funext h.literal32,
    funext (fun x => funext (h.add x)),funext (fun x => funext (h.sub x)),
    funext (fun x => funext (h.mul x)),funext (fun x => funext (h.div x)),
    funext (fun x => funext (h.max x)),funext h.exp2,
    funext (fun x => funext (fun y => funext (h.fmaRN x y))),
    funext (fun width => funext (fun row => funext (fun col =>
      funext (fun a => funext (fun b => funext (fun accumulate =>
        funext (h.wgmmaTF32 width row col a b accumulate)))))))⟩

theorem evaluated_equiv (actual reference : ScalarOps) (h : ScalarLaws actual reference)
    (plan : Plan) (term : Expr) (owner : Owner) :
    evaluate actual plan term owner=evaluate reference plan term owner := by
  rw [h.operations_eq actual reference]

#axiomsClean ScalarLaws.operations_eq
#axiomsClean evaluated_equiv
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.Numerical
