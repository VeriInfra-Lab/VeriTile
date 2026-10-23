import bench.repairs.tritonbench_g.triton_argmax.Proofs.NativeSafety
import bench.repairs.tritonbench_g.triton_argmax.Proofs.GlobalBridge
import bench.repairs.tritonbench_g.triton_argmax.Proofs.DimBridge

namespace VeriTile.Bench.Repairs.TritonArgmax.FPEquiv
open Source Memory TensorBinding
noncomputable section

theorem global_tensor (mem : Memory.Primitives M) (before : M)
    (p : NativeGraph.Primitives) (zero : NativeGraph.ZeroClamps p)
    (x : Input) (o : Checked.Observation) (observed : o.input = x.tensor.metadata)
    (h : Checked.InputContract o) (branch : o.dim = none)
    (adapter : RealScratch.Adapter) (dtype : o.dtype.adapter = some adapter)
    (g : NativeGraph.GlobalPrograms adapter.format o)
    (a : GlobalAllocation x o adapter.format callers) :
    ∃ result,globalInvoke mem before p adapter o g a = .ok result ∧
      result.returned = a.output.returned ∧
      result.returned.metadata.shape = Host.outputShape x.tensor.metadata.shape none o.keepdim ∧
      result.returned.dtypeName = "torch.int64" ∧
      result.returned = a.output.allocated ∧ NativeSafety.Global x o a ∧
      result.returned.storage.owner = x.storage.owner ∧
      mem.read 8 result.state (address result.returned.storage 8 0) =
        BitVec.ofNat 64 (NativeGraph.globalTarget p adapter g (Host.logicalWord mem before x.host)) ∧
      (∀ caller,caller ∈ callers → ∀ point,caller.contains point →
        mem.byte result.state point = mem.byte before point) := by
  have hg : Checked.GlobalContract o := by simpa only [branch] using h.2
  let setup := a.setup mem before x o observed hg g
  have native := NativeGraph.global_source p zero adapter g mem a.buffers before
  have readback := HostExecution.global_returned mem (NativeGraph.scalars p adapter) before x.host setup
  change mem.read 8 (GlobalMemory.run mem (NativeGraph.scalars p adapter) g.config a.buffers
      (Host.initializeIndices mem a.buffers g.config.midBlock before))
      (address a.output.returned.storage 8 0) = _ at readback
  refine ⟨⟨a.output.returned,NativeGraph.global p adapter g mem a.buffers before⟩,
    by simp [globalInvoke,(Checked.check_iff o).mpr h],rfl,a.outputShape,a.output.int64,a.directOutput,
    NativeSafety.global x o observed hg adapter dtype a,a.ownerOutput,?_,?_⟩
  · rw [native,readback,NativeGraph.global_target p zero]
    rfl
  · intro caller member point inside
    rw [native]
    exact HostExecution.global_all_caller_frame mem (NativeGraph.scalars p adapter)
      before x.host setup callers (by intro b hb; simp [setup,GlobalAllocation.setup,direct] at hb)
      a.freshMid a.freshIndex a.freshOutput caller member point inside

theorem dimension_tensor (mem : Memory.Primitives M) (before : M)
    (x : Input) (o : Checked.Observation) (observed : o.input = x.tensor.metadata)
    (h : Checked.InputContract o) (dim : Int) (branch : o.dim = some dim)
    (ax : Nat) (normalized : Host.normalizeDim o.input.shape.length dim = some ax)
    (g : NativeGraph.DimensionPrograms) (call : ContiguousCall mem before x callers)
    (prepared : o.prepared = call.result.tensor.metadata)
    (a : DimensionAllocation x o ax call.result callers) :
    ∃ result,dimensionInvoke mem before o ax g call a = .ok result ∧
      result.returned = a.output.returned ∧
      result.returned.metadata.shape = Host.outputShape x.tensor.metadata.shape (some ax) o.keepdim ∧
      result.returned.dtypeName = "torch.int64" ∧
      (o.keepdim = Bool.true → result.returned = a.output.allocated) ∧
      NativeSafety.Dimension o ax call.result result.returned ∧
      result.returned.storage = a.output.allocated.storage ∧
      (Checked.outer o ax = 0 ∨ Checked.inner o ax = 0 → result.state = call.state) ∧
      (∀ row k,row < Checked.outer o ax → k < Checked.inner o ax →
        mem.read 8 result.state (address result.returned.storage 8 (row*Checked.inner o ax+k)) =
          BitVec.ofNat 64 (NativeGraph.dimIndex g (g.config o ax)
            (Host.logicalWord mem before x.host) row k)) ∧
      (∀ caller,caller ∈ callers → ∀ point,caller.contains point →
        mem.byte result.state point = mem.byte before point) := by
  have hd : Checked.DimensionContract o ax := by simpa only [branch,normalized] using h.2
  have valid : ax < x.tensor.metadata.shape.length := by
    rw [←observed]
    exact Checked.normalized_valid _ _ _ normalized
  let setup := a.setup mem before x o observed ax valid g call
  have native := NativeGraph.dimension_source g o ax mem a.buffers call.state
  refine ⟨⟨a.output.returned,NativeGraph.dimension g o ax mem a.buffers call.state⟩,
    by simp [dimensionInvoke,(Checked.check_iff o).mpr h],rfl,a.shape,a.output.int64,a.keptOutput,
    NativeSafety.dimension x o observed h.1 ax valid hd call prepared a,
    a.output.sameStorage,NativeSafety.empty_dimension g o ax mem a.buffers call.state,?_,?_⟩
  · intro row k hr hk
    rw [native,NativeGraph.dimension_index]
    simpa only [setup,DimensionAllocation.setup,HostExecution.dimensionRun,Output.host,
      AllocatedTensor.host,Nat.zero_add] using
      HostExecution.dimension_returned mem NativeGraph.dimScalars before x.host setup row k hr hk
  · intro caller member point inside
    rw [native]
    exact HostExecution.dimension_all_caller_frame mem NativeGraph.dimScalars before x.host setup
      callers call.fresh a.fresh caller member point inside

/-- Reached node comparisons, including post-cast intermediates, are the only
conditions for identifying the native global index with the independent Real
least-maximum target. Dtype rounding can make these conditions false. -/
theorem global_real_bridge (p : NativeGraph.Primitives) (zero : NativeGraph.ZeroClamps p)
    (adapter : RealScratch.Adapter) (o : Checked.Observation) (hg : Checked.GlobalContract o)
    (g : NativeGraph.GlobalPrograms adapter.format o) (bits : Nat → Word) (logical : Nat → ℝ)
    (comparisons : GlobalBridge.LocalNumerics (NativeGraph.scalars p adapter) g.config bits logical) :
    NativeGraph.globalTarget p adapter g bits =
      (Math.argmax o.input.numel hg.2.2.1 (fun i => logical i.val)).val := by
  rw [NativeGraph.global_target p zero]
  exact GlobalBridge.reference_index _ _ (Checked.global_coverage o hg _ _
    (fun pid _ => (g.first pid).covers) g.second.covers) bits logical comparisons

structure ScalarAgreement (p q : NativeGraph.Primitives) : Prop where
  direct : ∀ d,p.direct d = q.direct d
  integer : ∀ k,p.integer k = q.integer k

theorem primitive_agreement (p q : NativeGraph.Primitives) (ports : ScalarAgreement p q) : p = q := by
  have hd := funext ports.direct
  have hi := funext ports.integer
  cases p
  cases q
  cases hd
  cases hi
  rfl

theorem local_refinement (p q : NativeGraph.Primitives) (ports : ScalarAgreement p q)
    (a : RealScratch.Adapter) (g : NativeGraph.GlobalPrograms a.format o) (bits : Nat → Word) :
    NativeGraph.globalTarget p a g bits = NativeGraph.globalTarget q a g bits := by
  rw [primitive_agreement p q ports]

#axiomsClean global_tensor
#axiomsClean dimension_tensor
#axiomsClean global_real_bridge
#axiomsClean local_refinement
#specNonCircular NativeGraph.globalTarget avoiding [NativeGraph.global,globalInvoke,GlobalMemory.run]
end
end VeriTile.Bench.Repairs.TritonArgmax.FPEquiv
