import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.NativeStructure
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.ContextAttnFwd.KernelProfile
open VeriTile.Triton FP.Structural NativeGraph
noncomputable section

/-- Change raw words and arbitrary literal fields, preserving only node shape.
Fixed source zero/sentinel constructors stay fixed. No selector reads the
numerical fields of an input or literal node. -/
def replacePayload (word : Word → Word) (literal : ℝ → ℝ) : Expr → Expr
  | .leaf w => .leaf (word w)
  | .zero => .zero
  | .negInf => .negInf
  | .sentinel => .sentinel
  | .literal p x => .literal p (literal x)
  | .fromNat p n => .fromNat p n
  | .fromInt p n => .fromInt p n
  | .cast p a b x => .cast p a b (replacePayload word literal x)
  | .unary p op x => .unary p op (replacePayload word literal x)
  | .binary p op x y => .binary p op (replacePayload word literal x) (replacePayload word literal y)
  | .product p x y => .product p (replacePayload word literal x) (replacePayload word literal y)
  | .fma p x y z => .fma p (replacePayload word literal x) (replacePayload word literal y)
      (replacePayload word literal z)
  | .reduction p op n trees xs => .reduction p op n trees (fun i => replacePayload word literal (xs i))
  | .dotAcc p left right out options batch m k n a b acc i =>
      .dotAcc p left right out options batch m k n
        (fun j => replacePayload word literal (a j))
        (fun j => replacePayload word literal (b j)) (replacePayload word literal acc) i

theorem route_payload_independent (bm d : Nat) (word : Word → Word) (literal : ℝ → ℝ)
    (e : Expr) (edge : Edge) (owner : Owner) :
    (NativeSM90.plan bm d).route (replacePayload word literal e) edge owner=
      (NativeSM90.plan bm d).route e edge owner := by
  cases e <;> cases edge <;> simp [NativeSM90.plan,NativeSM90.route,replacePayload]
  rename_i p left right out opts batch m k n a b acc i linear
  cases acc <;> rfl

theorem operand_order_payload_independent (bm d : Nat)
    (word : Word → Word) (literal : ℝ → ℝ)
    (p : Option ComputeDType) (x y : Expr) (owner : Owner) :
    (NativeSM90.plan bm d).reverseMul p (replacePayload word literal x)
      (replacePayload word literal y) owner=(NativeSM90.plan bm d).reverseMul p x y owner ∧
    (NativeSM90.plan bm d).reverseFMA p (replacePayload word literal x)
      (replacePayload word literal y) owner=(NativeSM90.plan bm d).reverseFMA p x y owner := by
  cases x <;> cases y <;> exact ⟨rfl,rfl⟩

theorem accumulate_payload_independent (word : Word → Word) (literal : ℝ → ℝ) (e : Expr) :
    accumulate (replacePayload word literal e)=accumulate e := by cases e <;> rfl

/-- In particular a live zero-valued PV seed cannot turn into a cleared QK seed. -/
theorem pv_seed_always_live (p : Option ComputeDType) (old factor : Expr) :
    accumulate (.product p old factor)=Bool.true := rfl

#axiomsClean route_payload_independent
#axiomsClean operand_order_payload_independent
#axiomsClean accumulate_payload_independent
#axiomsClean pv_seed_always_live
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnFwd.KernelProfile
