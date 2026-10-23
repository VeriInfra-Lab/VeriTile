import bench.repairs.tritonbench_g.parallel_attention.Proofs.Torch
import bench.repairs.tritonbench_g.parallel_attention.Proofs.BackwardContract
import bench.repairs.tritonbench_g.parallel_attention.Proofs.Materialize
import bench.repairs.tritonbench_g.parallel_attention.Proofs.FPReference

namespace VeriTile.Bench.Repairs.ParallelAttention.FPEquiv
noncomputable section
open HostAPI Checked

/-- Complete physical heap, returned Tensor identities, saved Q/K/V objects,
launch device and restored caller device. Only fixed-arity scalar laws remain. -/
specification fp_equivalent (c : Call) (d : Domain c) (profile : Syntax)
    (r : Public.Run c.shape (forwardBases (inputs c)) 128 forwardKinds)
    (t : ForwardTensors) (eps : FP32Word) (den dst : Nat) (h : Physical.Heap)
    (native : Ops FP32Word) (laws : ScalarAgreement native softwareOps) :
    invoke c d profile r t eps den dst native h=invoke c d profile r t eps den dst softwareOps h := by
  rw [laws.eq]

/-- Actual returned handles read the independent ordered software FP graph.
The specification contains no source or host memory-execution function. -/
specification returned_reference (c : Call) (d : Domain c) (profile : Syntax)
    (r : Public.Run c.shape (forwardBases (inputs c)) 128 forwardKinds)
    (t : ForwardTensors) (eps : FP32Word) (den dst : Nat) (h : Physical.Heap)
    (alloc : Allocations c r.output den dst t)
    (fresh : c.normalize=Bool.true → c.returnBoth=Bool.false → HostNative.NormalizeLayout c.shape r.output den dst)
    (native : Ops FP32Word) (laws : ScalarAgreement native softwareOps)
    (bh : Nat) (hb : bh<c.shape.B*c.shape.H) :
    let f:=invoke c d profile r t eps den dst native h
    f.tensors.Valid c ∧ Spec.Agrees c.shape.T c.shape.V
      ((f.tensors.handle c.shape).read (Physical.observe f.memory) bh)
      (Spec.forward softwareOps (specialization c d profile).compiler c.shape r.trees
        (currentInputs softwareOps c.shape (forwardBases (inputs c)) (Physical.observe h))
        eps c.returnBoth c.normalize bh) := by
  refine ⟨returned_valid c r.output den dst t alloc,?_⟩
  rw [laws.eq]
  change Spec.Agrees _ _ (((returned c t).handle c.shape).read
    (Physical.observe (invoke c d profile r t eps den dst softwareOps h).memory) bh) _
  rw [returned_handle c r.output den dst t alloc,invoke_observe]
  exact ordered_forward softwareOps (specialization c d profile).compiler c.shape (inputs c)
    r.trees r.scratch r.output r.ks r.hs (r.layout d.shape) (Physical.observe h)
    eps den dst c.returnBoth c.normalize fresh bh hb

/-- Full caller byte frame, including padding and sub-word addresses. -/
specification forward_frame (c : Call) (d : Domain c) (profile : Syntax)
    (r : Public.Run c.shape (forwardBases (inputs c)) 128 forwardKinds)
    (t : ForwardTensors) (eps : FP32Word) (den dst : Nat) (h : Physical.Heap)
    (native : Ops FP32Word) (address : Nat)
    (outsideKernel : ∀ ev∈r.ks,tensorAddress r.scratch ev≠address/4)
    (outsideHost : ∀ ev∈r.hs,tensorAddress r.output ev≠address/4)
    (outsideDen : c.normalize=Bool.true → c.returnBoth=Bool.false → ∀ i,i<c.shape.B*c.shape.H*c.shape.T → den+i≠address/4)
    (outsideOutput : c.normalize=Bool.true → c.returnBoth=Bool.false → ∀ i,i<c.shape.B*c.shape.H*c.shape.T*c.shape.V → dst+i≠address/4) :
    (invoke c d profile r t eps den dst native h).memory address=h address := by
  apply Physical.byte_frame
  rw [invoke_observe]
  exact HostNative.forward_frame native (specialization c d profile).compiler c.shape (inputs c)
    r.trees r.scratch r.output r.ks r.hs (Physical.observe h) eps den dst c.returnBoth c.normalize
    (address/4) outsideKernel outsideHost outsideDen outsideOutput

/-- Every byte of all Q/K/V storage allocations is unchanged, including
bytes not part of the active tensor views and any physical input aliasing. -/
specification caller_storage_frame (c : Call) (d : Domain c) (profile : Syntax)
    (r : Public.Run c.shape (forwardBases (inputs c)) 128 forwardKinds)
    (t : ForwardTensors) (eps : FP32Word) (den dst : Nat) (h : Physical.Heap)
    (native : Ops FP32Word) (fresh : CallerStorageFresh c r den dst)
    (input : TensorView) (member : input∈[c.q,c.k,c.v]) (offset : Nat) (inside : offset < input.capacity) :
    (invoke c d profile r t eps den dst native h).memory (input.storageBase+offset)=h (input.storageBase+offset) := by
  apply forward_frame
  · intro ev he;exact avoids_word input _ offset (fresh.kernel input member ev he) inside
  · intro ev he;exact avoids_word input _ offset (fresh.host input member ev he) inside
  · intro hn hb i hi;exact avoids_word input _ offset (fresh.denominator hn hb input member i hi) inside
  · intro hn hb i hi;exact avoids_word input _ offset (fresh.output hn hb input member i hi) inside

/-- Forward or backward kernel plus host reductions: each output byte equals
its finite software operation graph, for all exceptional FP32 input bits. -/
specification kernel_bytes (native : Ops FP32Word) (laws : ScalarAgreement native softwareOps)
    (k : Public.Specialization) (b : Bases) (L : Nat) (kinds : List OutputKind)
    (r : Public.Run k.shape b L kinds) (h : Physical.Heap)
    (kind : OutputKind) (hk : kind∈kinds) (i : Nat) (hi : i<kind.cells k.shape) (byte : Fin 4) :
    Physical.dispatch native k.compiler k.shape b r.trees r.scratch r.output r.ks r.hs h
      (4*(r.output kind+i)+byte.val)=
    Packing.word (nativeWordReference softwareOps k.compiler k.shape kind (r.trees kind)
      (currentInputs softwareOps k.shape b (Physical.observe h)) i) byte := by
  apply Physical.output_byte
  rw [laws.eq,Physical.observe_dispatch,
    HostNative.dispatch_eq _ _ _ _ _ _ _ kinds _ _ (r.layout k.domain)]
  exact compiled_output softwareOps k.compiler k.shape b r.trees r.scratch r.output kinds r.ks r.hs
    (r.layout k.domain).kernelCoverage (r.layout k.domain).hostCoverage (r.layout k.domain).kernelFresh
    (r.layout k.domain).kernelUnique (r.layout k.domain).hostFresh (r.layout k.domain).hostUnique
    (Physical.observe h) kind hk i hi

specification both_backward_bytes (native : Ops FP32Word) (laws : ScalarAgreement native softwareOps)
    (k : Public.Specialization) (x : HostAPI.Inputs) (do_ dz : Tensor)
    (boundary : PreparedBinding k.shape x do_ dz)
    (r : Public.Run k.shape (backwardBases (save softwareOps k.shape x) do_ dz) 64 backwardKinds)
    (ready : Physical.Heap) :
    Physical.backward native k.compiler (save native k.shape x) do_ dz
      r.trees r.scratch r.output r.ks r.hs ready=
    Physical.backward softwareOps k.compiler (save softwareOps k.shape x) do_ dz
      r.trees r.scratch r.output r.ks r.hs ready := by rw [laws.eq]

/-- Normalized/raw single-output backward consumes the actual retained
forward denominator and the later dY view, then executes the byte heap. -/
specification single_backward_bytes (native : Ops FP32Word) (laws : ScalarAgreement native softwareOps)
    (k : Public.Specialization) (f : HostNative.ForwardResult FP32Word) (normalize : Bool)
    (rows : Nat → Nat → TorchHost.Selection k.shape.V) (dy : View) (later : Physical.Heap)
    (dest : OutputKind → Nat)
    (boundary : PreparedBinding f.api.saved.shape f.api.saved.tensors
      (tensor f.api.saved.shape .forward (dest .forward)) (tensor f.api.saved.shape .normalizer (dest .normalizer)))
    (trees : OutputKind → SumPlan) (scratch out : OutputKind → Nat)
    (ks hs : List Event) :
    Physical.singleBackward native k.compiler f normalize (Public.rowTrees k.shape rows) dy later
      dest trees scratch out ks hs=
    Physical.singleBackward softwareOps k.compiler f normalize (Public.rowTrees k.shape rows) dy later
      dest trees scratch out ks hs := by rw [laws.eq]

specification backward_frame (a : Ops FP32Word) (c : CompilerLayout) (ctx : Saved FP32Word)
    (do_ dz : Tensor) (boundary : PreparedBinding ctx.shape ctx.tensors do_ dz)
    (trees : OutputKind → SumPlan) (scratch out : OutputKind → Nat)
    (ks hs : List Event) (h : Physical.Heap) (address : Nat)
    (outsideKernel : ∀ ev∈ks,tensorAddress scratch ev≠address/4)
    (outsideHost : ∀ ev∈hs,tensorAddress out ev≠address/4) :
    Physical.backward a c ctx do_ dz trees scratch out ks hs h address=h address := by
  apply Physical.byte_frame
  rw [Physical.observe_backward]
  exact HostNative.backward_frame a c ctx do_ dz trees scratch out ks hs (Physical.observe h)
    (address/4) outsideKernel outsideHost

specification single_frame (a : Ops FP32Word) (c : CompilerLayout) (f : HostNative.ForwardResult FP32Word)
    (normalize : Bool) (tree : Nat → Nat → SumPlan) (dy : View) (h : Physical.Heap)
    (dest : OutputKind → Nat)
    (boundary : PreparedBinding f.api.saved.shape f.api.saved.tensors
      (tensor f.api.saved.shape .forward (dest .forward)) (tensor f.api.saved.shape .normalizer (dest .normalizer)))
    (trees : OutputKind → SumPlan) (scratch out : OutputKind → Nat)
    (ks hs : List Event) (address : Nat)
    (outsidePropagation : ∀ ev∈events f.api.saved.shape Bool.false forwardKinds,tensorAddress dest ev≠address/4)
    (outsideKernel : ∀ ev∈ks,tensorAddress scratch ev≠address/4)
    (outsideHost : ∀ ev∈hs,tensorAddress out ev≠address/4) :
    Physical.singleBackward a c f normalize tree dy h dest trees scratch out ks hs address=h address := by
  apply Physical.byte_frame
  rw [Physical.observe_singleBackward]
  exact HostNative.single_frame a c f normalize tree dy (Physical.observe h) dest trees scratch out ks hs
    (address/4) outsidePropagation outsideKernel outsideHost

#axiomsClean caller_storage_frame
#axiomsClean backward_frame
#axiomsClean single_frame
#axiomsClean fp_equivalent
#axiomsClean returned_reference
#axiomsClean forward_frame
#axiomsClean kernel_bytes
#axiomsClean both_backward_bytes
#axiomsClean single_backward_bytes
#axiomsClean Physical.observe_singleBackward
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ParallelAttention.FPEquiv
