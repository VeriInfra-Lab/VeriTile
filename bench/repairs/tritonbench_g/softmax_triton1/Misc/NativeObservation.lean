import bench.repairs.tritonbench_g.softmax_triton1.Proofs.Rewrite
import bench.repairs.tritonbench_g.softmax_triton2.Proofs.NativeIO

/-! Auxiliary native word observation for the original softmax source. The source-to-reciprocal equivalence is in FPEquiv. -/
namespace VeriTile.Bench.Repairs.SoftmaxTriton1.FPEquiv
open VeriTile Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

/-- Independent ordered native graph, with the production blocked layout V1:
register folds, simultaneous butterflies, max/sub/log2(e)/exp2/sum/div.full.
Primitive instruction ports and graph selection are modeling boundaries. -/
def reference (inst : NativeGraph.Instructions) := NativeGraph.row inst 1

def negateWord (x : NativeGraph.Word) : NativeGraph.Word := x ^^^ (BitVec.ofNat 32 (2^31))

/-- Shared strided FP32 channel layout and blocked-thread observation, with
this task's actual source selected explicitly. -/
@[reducible] def nativeIO (inst : NativeGraph.Instructions) (inp out : RegionName)
    (m n ib ob ir ic orow oc B p : Nat) : ProgramIO NativeGraph.Word NativeGraph.Node :=
  { SoftmaxTriton2.NativeIO.io inst out inp m n ib ob ir ic orow oc B p with
    schedule := fun _ => [{
      SoftmaxTriton2.NativeIO.launch inst out inp m n ib ob ir ic orow oc B p with
      kernel := softmax_kernel out inp ib ob ir ic orow oc n B }] }

/-- Actual full source execution, independent ordered native result and exact
exterior cells. Original row coverage, launch and output-injectivity conditions
are retained. Primitive graph selection is a model boundary, not a rewrite. -/
theorem softmax_kernel_fp32 (inst : NativeGraph.Instructions) (inp out : RegionName)
    (m n ib ob ir ic orow oc B p : Nat) (hB : 0 < B) (cover : n ≤ B) (hp : p < m)
    (injective : ∀ r c r' c', r<m → c<n → r'<m → c'<n →
      ob+r*orow+c*oc=ob+r'*orow+c'*oc → r=r' ∧ c=c') :
    (nativeIO inst inp out m n ib ob ir ic orow oc B p).Implements (fun xs =>
      SoftmaxTriton2.NativeIO.pack out B (fun j =>
        reference inst n B (SoftmaxTriton2.NativeIO.words B (xs 0)) j.1.val)) := by
  apply SoftmaxTriton2.NativeIO.implements inst out inp m n ib ob ir ic orow oc B p hB
  intro i j hi hj equal
  apply Fin.ext
  exact (injective p i.val p j.val hp hi hp hj (by simp [equal])).2

#stmtSurfaceSubset softmax_kernel_fp32 ⊆
  [NativeGraph.Instructions, RegionName, NativeGraph.Word, NativeGraph.Node, nativeIO,
    ProgramIO.Implements, SoftmaxTriton2.NativeIO.pack, SoftmaxTriton2.NativeIO.words,
    reference, ProgramIO.Inputs, IOChannel, MemoryIO.inputs, ProgramIO.toMemoryIO,
    List.instNeZeroNatLengthCons, IOChannel.mk, TileDType.real, TileIndex]

#specNonCircular reference avoiding [softmax_kernel,reciprocal_kernel,NativeGraph.algebra]

#axiomsClean softmax_kernel_fp32

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxTriton1.FPEquiv
