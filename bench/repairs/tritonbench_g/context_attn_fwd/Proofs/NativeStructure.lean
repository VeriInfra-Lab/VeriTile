import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.NativeSM90

namespace VeriTile.Bench.Repairs.ContextAttnFwd.NativeGraph
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000

/-- Change arbitrary input payload words while preserving source structure. -/
def reword (f : Word → Word) : Expr → Expr
  | .leaf w => .leaf (f w)
  | .zero => .zero | .negInf => .negInf | .sentinel => .sentinel
  | .literal p x => .literal p x | .fromNat p x => .fromNat p x | .fromInt p x => .fromInt p x
  | .cast p a b x => .cast p a b (reword f x)
  | .unary p op x => .unary p op (reword f x)
  | .binary p op x y => .binary p op (reword f x) (reword f y)
  | .product p x y => .product p (reword f x) (reword f y)
  | .fma p x y z => .fma p (reword f x) (reword f y) (reword f z)
  | .reduction p op n trees xs => .reduction p op n trees (fun i => reword f (xs i))
  | .dotAcc p left right out options batch m k n a b acc i =>
    .dotAcc p left right out options batch m k n (fun j => reword f (a j))
      (fun j => reword f (b j)) (reword f acc) i

theorem route_payload_independent (bm d : Nat) (f : Word → Word) (e : Expr) (edge : Edge) (owner : Owner) :
    (NativeSM90.plan bm d).route (reword f e) edge owner=(NativeSM90.plan bm d).route e edge owner := by
  cases e <;> cases edge <;> simp [NativeSM90.plan,NativeSM90.route,reword]
  rename_i p left right out opts batch m k n a b acc i linear
  cases acc <;> rfl

theorem multiply_order_payload_independent (bm d : Nat) (f : Word → Word)
    (p : Option ComputeDType) (x y : Expr) (owner : Owner) :
    (NativeSM90.plan bm d).reverseMul p (reword f x) (reword f y) owner=
      (NativeSM90.plan bm d).reverseMul p x y owner := by
  cases x <;> cases y <;> rfl

theorem fma_order_payload_independent (bm d : Nat) (f : Word → Word)
    (p : Option ComputeDType) (x y : Expr) (owner : Owner) :
    (NativeSM90.plan bm d).reverseFMA p (reword f x) (reword f y) owner=
      (NativeSM90.plan bm d).reverseFMA p x y owner := rfl

/-- Both source dot sites have complete fixed-k8 chains, with no padding on
the checked widths. Instruction shape and count cannot depend on payloads. -/
theorem qk_k8_exact (d : Nat) (hd : d∈({16,32,64,128,256} : Finset Nat)) :
    (d+7)/8=d/8 ∧ 8*(d/8)=d := by
  simp only [Finset.mem_insert,Finset.mem_singleton] at hd
  rcases hd with rfl|rfl|rfl|rfl|rfl <;> decide

theorem pv_k8_exact (bm : Nat) (hb : bm=64 ∨ bm=128) :
    (bm+7)/8=bm/8 ∧ 8*(bm/8)=bm := by
  rcases hb with rfl|rfl <;> decide

#print axioms route_payload_independent
end
end VeriTile.Bench.Repairs.ContextAttnFwd.NativeGraph
