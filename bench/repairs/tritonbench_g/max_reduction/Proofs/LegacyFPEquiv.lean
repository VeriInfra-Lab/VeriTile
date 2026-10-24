import bench.repairs.tritonbench_g.max_reduction.Misc.ASTHost

namespace VeriTile.Bench.Repairs.MaxReduction.FPEquiv
open Execution
noncomputable section

/-- All FP32 words, including NaNs, signed zero and infinities. No real
maximum or attaining-index claim is imposed on unordered native comparisons. -/
theorem max_fp32 (ops : NativeOps) (x : VeriTile.Torch.TensorMetadata) (h : GlobalContract x)
    (profile : Host.GlobalProfile x.numel) (s : Heap Word) :
    ∃ result : Host.ScalarResult Word,Host.max (Compute.native ops) x profile s=.ok result ∧
      result.shape=[] ∧ result.dtype=.floating .fp32 ∧ result.device=x.device ∧
      result.heap ("output",0)=.value (globalReference (Compute.native ops)
        (Execution.read negativeInfinity s "input") x.numel (globalBlock x.numel)
        (fun p => (profile.first p).tree) profile.second.tree) ∧
      (∀ q,(q.1≠"middle" ∨ globalMid x.numel≤q.2) → q≠("output",0) → result.heap q=s q) := by
  exact ⟨⟨[],.floating .fp32,x.device,_⟩,Host.max_success _ _ h _ _,rfl,rfl,rfl,
    global_result (Compute.native ops) x.numel (globalBlock x.numel) (globalWidth x.numel)
      (fun p => (profile.first p).tree) profile.second.tree s,
    fun q hm ho => global_frame (Compute.native ops) x.numel (globalBlock x.numel) (globalWidth x.numel)
      (fun p => (profile.first p).tree) profile.second.tree s q hm ho⟩

theorem max_dim_fp32 (ops : NativeOps) (a : DimArguments) (h : DimContract a)
    (trials : List (Host.DimProfile a.N)) (selected : Host.DimProfile a.N) (s : Heap Word) :
    ∃ result : Host.DimensionResult Word,Host.maxDim (Compute.native ops) a trials selected s=.ok result ∧
      result.shape=a.outputShape ∧ result.valuesDtype=.floating .fp32 ∧
      result.indicesDtype=.other "int64" ∧ result.device=a.input.device ∧
      (∀ j,j<a.M*a.K →
        result.heap ("values",j)=.value (nativeDimension ops selected.reduction.tree
          (Execution.read negativeInfinity s "input") a.N a.K (j/a.K) (j%a.K)).1 ∧
        result.heap ("indices",j)=.index (nativeDimension ops selected.reduction.tree
          (Execution.read negativeInfinity s "input") a.N a.K (j/a.K) (j%a.K)).2) ∧
      (∀ q,(q.1≠"values" ∧ q.1≠"indices" ∨ a.M*a.K≤q.2) → result.heap q=s q) := by
  refine ⟨⟨a.outputShape,.floating .fp32,.other "int64",a.input.device,_⟩,
    Host.maxDim_success _ _ h _ _ _,rfl,rfl,rfl,rfl,?_,?_⟩
  · intro j hj
    exact Host.maxDim_values _ _ _ _ _ j hj
  · intro q miss
    exact Host.maxDim_frame _ _ _ _ _ q miss

/-- Only the three scalar comparison/maxnum ports are deferred. Both complete
reduction trees are expanded; no whole-reduction or memory equality is a port. -/
theorem global_local_conformance (ops : NativeOps) (target : Native.Target)
    (leaves : Native.LocalObligations ops target) (input : Nat → Word) (count block : Nat)
    (first : Nat → Tree) (second : Tree) :
    globalReference (Compute.native ops) input count block first second=
      globalReference (Compute.nativeReference target) input count block first second := by
  unfold globalReference
  rw [Compute.native_scalar_equiv ops target leaves]
  apply Compute.scalar_local
  intro p _
  exact Compute.native_scalar_equiv ops target leaves (first p) input count (p*block)

#axiomsClean max_fp32
#axiomsClean max_dim_fp32
#axiomsClean global_local_conformance
end
end VeriTile.Bench.Repairs.MaxReduction.FPEquiv
