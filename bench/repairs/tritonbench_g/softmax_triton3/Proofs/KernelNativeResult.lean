import bench.repairs.tritonbench_g.softmax_triton3.Proofs.NativeSource

namespace VeriTile.Bench.Repairs.SoftmaxTriton3.KernelNativeResult
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding blockSize numWarps
open VeriTile.Bench.Repairs.SoftmaxTriton3.NativeGraph
noncomputable section

/-- Independent ordered native graph, with input maximum before subtraction,
optional mask addition, log2(e) multiply, exp2, selected reduction and division. -/
def reference (inst : Instructions) (profile : Profile) := NativeGraph.row inst profile.vector

def negateWord (x : Word) : Word := x ^^^ (BitVec.ofNat 32 (2^31))

/-- Complete native row source/projection on raw word state and lane arrays.
The primitive instruction graph and full exterior Cell/tag frame are retained. -/
theorem softmax_kernel_fp32 (inst : Instructions) (profile : Profile)
    (out inp : RegionName) (mask : Option RegionName) (rows n B p : Nat)
    (positive : 0 < B) (cover : n ≤ B) (valid : p < rows)
    (s : State Word) (xs : Nat → Word) (ms : Option (Nat → Word))
    (input : ∀ c, c < n → Floating.access.read s inp (p*n+c) = xs c)
    (masks : ∀ r, mask = some r → ∀ c, c < n →
      Floating.access.read s r (p*n+c) = (ms.getD (fun _ => Floating.access.read s inp 0)) c)
    (matching : mask.isSome = ms.isSome) :
    ∃ t, NativeGraph.evaluate inst profile.vector out inp mask rows n B p s = some t ∧
      (∀ c, c < n → Floating.access.read t out (p*n+c) = reference inst profile n B xs ms c) ∧
      (∀ r i, (r ≠ out ∨ ∀ c, c < n → i ≠ p*n+c) → t.mem r i = s.mem r i) := by
  obtain ⟨t,run⟩ := NativeGraph.terminates inst profile.vector out inp mask rows n B p positive s
  exact ⟨t,run,(NativeGraph.source inst profile.vector).row out inp mask rows n B p s t xs ms
    positive cover valid input masks matching run⟩

#specNonCircular reference avoiding [softmax_kernel,NativeGraph.evaluate,Floating.evaluate]
#axiomsClean softmax_kernel_fp32
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxTriton3.KernelNativeResult
