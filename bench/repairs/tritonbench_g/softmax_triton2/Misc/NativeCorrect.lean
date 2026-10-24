import bench.repairs.tritonbench_g.softmax_triton2.Proofs.NativeIO
import bench.repairs.tritonbench_g.softmax_triton2.Proofs.KernelNative

/-! Retained single-source native correctness. This module declares ordinary
theorems only; a two-program FP equivalence is not established here. -/

namespace VeriTile.Bench.Repairs.SoftmaxTriton2.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxTriton1 NativeGraph
noncomputable section

/-- Independent ordered native graph: register left folds, simultaneous
butterflies, max/sub/log2(e)/exp2/sum/div.full, at production layout V1. -/
def reference (inst : Instructions) := NativeGraph.row inst 1

/-- Source precision, active strided windows and blocked-thread output ownership
are explicit in the shared public descriptor in IO.lean. -/
@[reducible] def nativeIO (inst : Instructions) (out inp : RegionName)
    (rows cols ib ob ir ic orow oc B p : Nat) : ProgramIO Word Node :=
  NativeIO.io inst out inp rows cols ib ob ir ic orow oc B p

/-- The complete source implements the independent ordered native graph on
all active columns, with source access safety and exact exterior cells.
The existing row coverage, legal row and output-injectivity domain is retained. -/
theorem softmax_kernel_fp32 (inst : Instructions) (out inp : RegionName)
    (rows cols ib ob ir ic orow oc B p : Nat) (hB : 0<B) (cover : cols≤B) (hp : p<rows)
    (injective : ∀ r c r' c', r<rows → c<cols → r'<rows → c'<cols →
      ob+r*orow+c*oc=ob+r'*orow+c'*oc → r=r' ∧ c=c') :
    (nativeIO inst out inp rows cols ib ob ir ic orow oc B p).Implements
      (fun xs => NativeIO.pack out B (fun j =>
        reference inst cols B (NativeIO.words B (xs 0)) j.1.val)) := by
  apply NativeIO.implements inst out inp rows cols ib ob ir ic orow oc B p hB
  intro i j hi hj eq
  apply Fin.ext
  exact (injective p i.val p j.val hp hi hp hj (by simp [eq])).2

#specNonCircular reference avoiding [SoftmaxTriton2.softmax_kernel,KernelNative.evaluate,
  NativeGraph.evaluate,Floating.formula,Floating.rowExpression,NativeGraph.algebra]
#axiomsClean softmax_kernel_fp32
#stmtSurfaceSubset softmax_kernel_fp32 ⊆
  [Instructions,RegionName,Word,Node,TileIndex,nativeIO,ProgramIO.Implements,NativeIO.pack,
   NativeIO.words,reference,ProgramIO.Inputs,IOChannel,MemoryIO.inputs,
   ProgramIO.toMemoryIO,List.instNeZeroNatLengthCons,IOChannel.mk,TileDType.real]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxTriton2.FPEquiv
