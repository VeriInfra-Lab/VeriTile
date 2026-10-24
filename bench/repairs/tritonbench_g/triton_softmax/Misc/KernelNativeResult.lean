import bench.repairs.tritonbench_g.triton_softmax.Proofs.KernelInterface

namespace VeriTile.Bench.Repairs.TritonSoftmax.KernelNativeResult
open VeriTile.Triton
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding softmax_kernel numWarps
open RawKernel
noncomputable section

open FP.Structural

/-- Both typed sources and native observations succeed, with ordered native
outputs and both source-cell and physical-word exterior frames. -/
theorem kernel_native (inst : NativeGraph.Instructions) (x y : View) (B p : Nat)
    (shape : y.rows=x.rows ∧ y.cols=x.cols) (injective : y.Injective)
    (positive : 0<B) (cover : x.cols≤B) (row : p<x.rows)
    (s : State NativeGraph.Word) (values : Nat → Nat → NativeGraph.Word)
    (input : ∀ r c,r<x.rows → c<x.cols → (s.mem x.region (x.index r c)).read .real=values r c) :
    ∃ nodes t,
      Floating.evaluate (NativeGraph.algebra inst 1) x.descriptor y.descriptor B p (NativeGraph.liftState s)=some nodes ∧
      KernelNative.candidateEvaluate (NativeGraph.algebra inst 1) x.descriptor y.descriptor B p (NativeGraph.liftState s)=some nodes ∧
      t=NativeGraph.projectState 1 B y.descriptor p s nodes ∧
      NativeGraph.evaluate inst 1 x.descriptor y.descriptor B p s=some t ∧
      KernelNative.candidate inst 1 x.descriptor y.descriptor B p s=some t ∧
      (∀ c,c<x.cols → (t.mem y.region (y.index p c)).read .real=NativeGraph.row inst 1 x.cols B (values p) c) ∧
      (∀ r i,(∀ c,c<y.cols → ¬(y.region=r ∧ y.index p c=i)) →
        nodes.mem r i=(NativeGraph.liftState s).mem r i ∧ t.mem r i=s.mem r i) := by
  obtain ⟨nodes,t,sourceRun,candidateRun,project,nativeRun,nativeCandidateRun⟩ :=
    KernelNative.complete inst 1 x.descriptor y.descriptor B p positive s
  have nodeInput : Inputs Floating.access x.descriptor (fun r c => NativeGraph.atom (values r c))
      (NativeGraph.liftState s) := by
    intro r c hr hc
    change NativeGraph.atom ((s.mem x.region (x.index r c)).read .real)=_
    exact congrArg NativeGraph.atom (input r c hr hc)
  have typed := (Floating.source (NativeGraph.algebra inst 1)).row x.descriptor y.descriptor B p
    (NativeGraph.liftState s) nodes (fun r c => NativeGraph.atom (values r c))
    ⟨shape.1,shape.2,rfl⟩ injective positive cover row nodeInput sourceRun
  have result := (NativeGraph.source inst 1).row x.descriptor y.descriptor B p s t values
    ⟨shape.1,shape.2,rfl⟩ injective positive cover row input nativeRun
  refine ⟨nodes,t,sourceRun,candidateRun,project,nativeRun,nativeCandidateRun,result.1,?_⟩
  intro r i outside
  have off : OffRows y.descriptor [p] r i := by
    intro q hq c hc
    have same : q=p := by simpa using hq
    subst q
    exact outside c hc
  exact ⟨typed.2 r i off,result.2 r i off⟩

#axiomsClean kernel_native
end
end VeriTile.Bench.Repairs.TritonSoftmax.KernelNativeResult
