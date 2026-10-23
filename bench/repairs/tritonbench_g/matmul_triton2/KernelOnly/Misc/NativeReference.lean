import bench.repairs.tritonbench_g.matmul_triton2.revision001.Proofs.Native

/-! Optional single-source native reference: padding, selected fixed-k8
instructions and incoming accumulator order remain explicit. -/
namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.Reference
open VeriTile.Triton Revision001
noncomputable section

def left {α : Type} (P : Core.Primitives α) (K B : Nat)
    (a : Nat → Nat → α) (r q : Nat) (k : Fin B) : α :=
  if q*B+k.val<K then a r (q*B+k.val) else P.zero
def right {α : Type} (P : Core.Primitives α) (K B : Nat)
    (b : Nat → Nat → α) (c q : Nat) (k : Fin B) : α :=
  if q*B+k.val<K then b (q*B+k.val) c else P.zero

def accumulation {α : Type} (P : Core.Primitives α) (K B : Nat)
    (a b : Nat → Nat → α) (r c : Nat) : Nat → α
  | 0 => P.zero
  | q+1 => P.dot (left P K B a r q) (right P K B b c q)
      (accumulation P K B a b r c q)

def native (ops : Native.Reference) (choice : Fin 8) (K B : Nat)
    (a b : Nat → Nat → Native.Word) (r c : Nat) : Native.Word :=
  accumulation (Native.primitives (Native.referenceInstructions ops) (Native.family choice))
    K B a b r c ((K+B-1)/B)

#specNonCircular native avoiding [Core.grid,Core.source,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.Reference
