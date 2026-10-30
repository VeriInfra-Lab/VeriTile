import bench.repairs.tritonbench_g.parallel_attention.Proofs.TorchReal
import bench.repairs.tritonbench_g.parallel_attention.Proofs.BackwardContract

namespace VeriTile.Bench.Repairs.ParallelAttention.Correct
noncomputable section
open HostAPI Public Checked

/-- The actual specialization descriptor is admitted by the pinned structural
compiler checker. Lean fixes its profile parametrically; the ordinary artifact,
ABI and local-instruction boundary is stated in COMPILER-CONTRACT.md. -/
specification forward_correct (k : Specialization) (x : HostAPI.Inputs) (metadata : x.Valid k.shape)
    (r : Run k.shape (forwardBases x) 128 forwardKinds) (entry : Memory ℝ)
    (eps : ℝ) (den dst : Nat) (returnBoth normalize : Bool)
    (fresh : normalize=Bool.true → returnBoth=Bool.false → HostNative.NormalizeLayout k.shape r.output den dst)
    (bh : Nat) (hb : bh<k.shape.B*k.shape.H) :
    let f := forward realOps k x r entry eps den dst returnBoth normalize
    PublicResult.Agrees k.shape.T k.shape.V (f.api.returned.read f.api.memory bh)
      (publicSpec k.shape.T k.shape.K (selectedScale k.shape.K k.shape.useScale) eps
        (inputValues realOps k.shape x entry bh) returnBoth normalize) :=
  HostNative.forward_correct k.compiler k.shape k.domain x metadata r.trees r.scratch r.output r.ks r.hs
    (r.layout k.domain) entry eps den dst returnBoth normalize fresh bh hb

/-- O/Z return priority and the saved forward input/scale context feed the
actual custom backward, including incoming view materialization. -/
specification both_vjp (k : Specialization) (x : HostAPI.Inputs) (metadata : x.Valid k.shape)
    (fRun : Run k.shape (forwardBases x) 128 forwardKinds) (entry : Memory ℝ)
    (eps : ℝ) (den dst : Nat) (normalize : Bool)
    (incoming ready : Memory ℝ) (valid : SavedValid (save realOps k.shape x) entry ready)
    (doView dzView : View) (do_ dz : Tensor)
    (boundary : BackwardBinding k.shape x doView dzView do_ dz)
    (ho : Materialized k.shape .forward incoming ready doView do_)
    (hz : Materialized k.shape .normalizer incoming ready dzView dz)
    (bRun : Run k.shape (backwardBases (save realOps k.shape x) do_ dz) 64 backwardKinds)
    (bh : Nat) (hb : bh<k.shape.B*k.shape.H) (dx : ParallelAttention.Inputs ℝ) :
    let f := forward realOps k x fRun entry eps den dst Bool.true normalize
    let result := HostNative.backward realOps k.compiler f.api.saved do_ dz
      bRun.trees bRun.scratch bRun.output bRun.ks bRun.hs ready
    result.returned.scale=none ∧
    HasDerivAt (fun a : ℝ => lossSpec k.shape.T k.shape.K k.shape.V f.api.saved.scale
      (affineInputs (inputValues realOps k.shape x entry bh) dx a)
      (incomingCotangent realOps k.shape incoming doView dzView bh))
      (result.returned.pairing k.shape bh dx result.memory) 0 :=
  HostNative.returned_context_vjp k.compiler k.shape k.domain x metadata fRun.trees fRun.scratch fRun.output
    fRun.ks fRun.hs entry eps den dst Bool.true normalize incoming ready valid doView dzView do_ dz ho hz
    bRun.trees bRun.scratch bRun.output bRun.ks bRun.hs (bRun.layout k.domain) bh hb dx

specification normalized_vjp (k : Specialization) (x : HostAPI.Inputs) (metadata : x.Valid k.shape)
    (fRun : Run k.shape (forwardBases x) 128 forwardKinds) (entry later : Memory ℝ)
    (eps : ℝ) (den dst : Nat) (fresh : HostNative.NormalizeLayout k.shape fRun.output den dst)
    (dy : View) (rows : Nat → Nat → TorchHost.Selection k.shape.V)
    (preserved : let f := forward realOps k x fRun entry eps den dst Bool.false Bool.true
      RawSaved k.shape f.api.rawOutput (tensor k.shape .normalizer den) f.api.memory later)
    (dest : OutputKind → Nat)
    (boundary : PreparedBinding k.shape x (tensor k.shape .forward (dest .forward))
      (tensor k.shape .normalizer (dest .normalizer)))
    (propLayout : PropagationLayout k.shape dest)
    (propFresh : PropagationFresh k.shape (tensor k.shape .forward (fRun.output .forward))
      (tensor k.shape .normalizer den) dy dest)
    (saved : SavedValid (save realOps k.shape x) entry later)
    (inputFresh : ∀ ev∈events k.shape Bool.false forwardKinds,tensorAddress dest ev∉inputCellsOnly k.shape x)
    (bRun : Run k.shape (backwardBases (save realOps k.shape x)
      (tensor k.shape .forward (dest .forward)) (tensor k.shape .normalizer (dest .normalizer))) 64 backwardKinds)
    (bh : Nat) (hb : bh<k.shape.B*k.shape.H) (dx : ParallelAttention.Inputs ℝ)
    (nonzero : ∀ t,t<k.shape.T → normalizerSpec k.shape.T k.shape.K (selectedScale k.shape.K k.shape.useScale)
      (inputValues realOps k.shape x entry bh) t+eps≠0) :
    let f := forward realOps k x fRun entry eps den dst Bool.false Bool.true
    let result := HostNative.singleBackward realOps k.compiler f Bool.true (rowTrees k.shape rows)
      dy later dest bRun.trees bRun.scratch bRun.output bRun.ks bRun.hs
    result.returned.scale=none ∧
    HasDerivAt (fun a : ℝ => normalizedLoss k.shape.T k.shape.K k.shape.V
      (selectedScale k.shape.K k.shape.useScale) eps
      (affineInputs (inputValues realOps k.shape x entry bh) dx a) (viewRead realOps k.shape k.shape.V dy later bh))
      (result.returned.pairing k.shape bh dx result.memory) 0 :=
  HostNative.single_normalized_vjp k.compiler k.shape k.domain x metadata fRun.trees fRun.scratch fRun.output
    fRun.ks fRun.hs (fRun.layout k.domain) entry later eps den dst fresh dy (rowTrees k.shape rows)
    (fun bh t => (rows bh t).complete) preserved dest propLayout propFresh saved inputFresh
    bRun.trees bRun.scratch bRun.output bRun.ks bRun.hs (bRun.layout k.domain) bh hb dx nonzero

specification raw_vjp (k : Specialization) (x : HostAPI.Inputs) (metadata : x.Valid k.shape)
    (fRun : Run k.shape (forwardBases x) 128 forwardKinds) (entry later : Memory ℝ)
    (eps : ℝ) (den dst : Nat) (dy : View) (rows : Nat → Nat → TorchHost.Selection k.shape.V)
    (dest : OutputKind → Nat)
    (boundary : PreparedBinding k.shape x (tensor k.shape .forward (dest .forward))
      (tensor k.shape .normalizer (dest .normalizer)))
    (propLayout : PropagationLayout k.shape dest)
    (propFresh : PropagationFresh k.shape (tensor k.shape .forward (fRun.output .forward))
      (tensor k.shape .normalizer (fRun.output .normalizer)) dy dest)
    (saved : SavedValid (save realOps k.shape x) entry later)
    (inputFresh : ∀ ev∈events k.shape Bool.false forwardKinds,tensorAddress dest ev∉inputCellsOnly k.shape x)
    (bRun : Run k.shape (backwardBases (save realOps k.shape x)
      (tensor k.shape .forward (dest .forward)) (tensor k.shape .normalizer (dest .normalizer))) 64 backwardKinds)
    (bh : Nat) (hb : bh<k.shape.B*k.shape.H) (dx : ParallelAttention.Inputs ℝ) :
    let f := forward realOps k x fRun entry eps den dst Bool.false Bool.false
    let result := HostNative.singleBackward realOps k.compiler f Bool.false (rowTrees k.shape rows)
      dy later dest bRun.trees bRun.scratch bRun.output bRun.ks bRun.hs
    result.returned.scale=none ∧
    HasDerivAt (fun a : ℝ => lossSpec k.shape.T k.shape.K k.shape.V (selectedScale k.shape.K k.shape.useScale)
      (affineInputs (inputValues realOps k.shape x entry bh) dx a)
      (outputOnlyCotangent (viewRead realOps k.shape k.shape.V dy later bh)))
      (result.returned.pairing k.shape bh dx result.memory) 0 :=
  HostNative.single_raw_vjp k.compiler k.shape k.domain x metadata fRun.trees fRun.scratch fRun.output
    fRun.ks fRun.hs (fRun.layout k.domain) entry later eps den dst dy (rowTrees k.shape rows)
    dest propLayout propFresh saved inputFresh bRun.trees bRun.scratch bRun.output bRun.ks bRun.hs
    (bRun.layout k.domain) bh hb dx


#specNonCircular publicSpec avoiding [HostNative.forward,HostNative.backward,HostNative.singleBackward,compiledRun,nativeWordReference]
#specNonCircular normalizedLoss avoiding [HostNative.forward,HostNative.backward,HostNative.singleBackward,compiledRun,nativeWordReference]


/-- Whole checked public forward call: actual returned allocation metadata,
caller-device restoration, and independent causal quadratic attention values. -/
specification correct_real (c : Call) (d : Domain c) (profile : Syntax)
    (r : Public.Run c.shape (forwardBases (inputs c)) 128 forwardKinds)
    (t : ForwardTensors) (eps : ℝ) (den dst : Nat) (h : Nat → ℝ)
    (alloc : Allocations c r.output den dst t)
    (fresh : c.normalize=Bool.true → c.returnBoth=Bool.false → HostNative.NormalizeLayout c.shape r.output den dst)
    (bh : Nat) (hb : bh<c.shape.B*c.shape.H) :
    let f:=invokeReal c d profile r t eps den dst h
    f.tensors.Valid c ∧ f.restoredDevice=c.callerDevice ∧
    PublicResult.Agrees c.shape.T c.shape.V
      ((f.tensors.handle c.shape).read f.memory bh)
      (publicSpec c.shape.T c.shape.K (selectedScale c.shape.K c.shape.useScale) eps
        (inputValues realOps c.shape (inputs c) (realObserve h) bh) c.returnBoth c.normalize) := by
  refine ⟨returned_valid c r.output den dst t alloc,rfl,?_⟩
  change PublicResult.Agrees _ _ (((returned c t).handle c.shape).read
    (logicalForward c d profile r eps den dst realOps (realObserve h)).api.memory bh) _
  rw [returned_handle c r.output den dst t alloc]
  exact HostNative.forward_correct (specialization c d profile).compiler c.shape d.shape (inputs c)
    (inputs_valid c) r.trees r.scratch r.output r.ks r.hs (r.layout d.shape)
    (realObserve h) eps den dst c.returnBoth c.normalize fresh bh hb

specification real_frame (c : Call) (d : Domain c) (profile : Syntax)
    (r : Public.Run c.shape (forwardBases (inputs c)) 128 forwardKinds)
    (t : ForwardTensors) (eps : ℝ) (den dst : Nat) (h : Nat → ℝ) (address : Nat)
    (outsideKernel : ∀ ev∈r.ks,tensorAddress r.scratch ev≠address)
    (outsideHost : ∀ ev∈r.hs,tensorAddress r.output ev≠address)
    (outsideDen : c.normalize=Bool.true → c.returnBoth=Bool.false → ∀ i,i<c.shape.B*c.shape.H*c.shape.T → den+i≠address)
    (outsideOutput : c.normalize=Bool.true → c.returnBoth=Bool.false → ∀ i,i<c.shape.B*c.shape.H*c.shape.T*c.shape.V → dst+i≠address) :
    (invokeReal c d profile r t eps den dst h).memory address=h (4*address) :=
  HostNative.forward_frame realOps (specialization c d profile).compiler c.shape (inputs c)
    r.trees r.scratch r.output r.ks r.hs (realObserve h) eps den dst c.returnBoth c.normalize
    address outsideKernel outsideHost outsideDen outsideOutput

#axiomsClean real_frame
#axiomsClean correct_real
#axiomsClean forward_correct
#axiomsClean both_vjp
#axiomsClean normalized_vjp
#axiomsClean raw_vjp
#guard_msgs (drop info) in
#auditModuleAxioms

end
end VeriTile.Bench.Repairs.ParallelAttention.Correct
