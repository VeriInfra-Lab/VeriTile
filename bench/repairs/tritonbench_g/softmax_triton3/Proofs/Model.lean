import bench.repairs.tritonbench_g.softmax_triton3.Proofs.KernelSource
import bench.repairs.tritonbench_g.softmax_triton1.Proofs.Grid
namespace VeriTile.Bench.Repairs.SoftmaxTriton3
open VeriTile.Triton
open VeriTile.Bench.Repairs.SoftmaxTriton1

abbrev Evaluator (S : Type) := RegionName → RegionName → Option RegionName → Nat → Nat → Nat → Nat → S → Option S
abbrev Formula (V : Type) := Nat → Nat → (Nat → V) → Option (Nat → V) → Nat → V

structure Source {S C V} (a : Access S C V) (evaluate : Evaluator S) (formula : Formula V) : Prop where
  row : ∀ out inp mask rows n B p s t xs ms,
    0<B → n≤B → p<rows →
    (∀ c,c<n → a.read s inp (p*n+c)=xs c) →
    (∀ r,mask=some r → ∀ c,c<n → a.read s r (p*n+c)=(ms.getD (fun _ => a.read s inp 0)) c) →
    (mask.isSome=ms.isSome) →
    evaluate out inp mask rows n B p s=some t →
    (∀ c,c<n → a.read t out (p*n+c)=formula n B xs ms c) ∧
    (∀ r i,(r≠out ∨ ∀ c,c<n → i≠p*n+c) → a.cells t r i=a.cells s r i)

end VeriTile.Bench.Repairs.SoftmaxTriton3
