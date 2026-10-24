import bench.repairs.tritonbench_g.parallel_attention.Proofs.HostPropagation
import bench.repairs.tritonbench_g.parallel_attention.Proofs.NativeExecution
import bench.repairs.tritonbench_g.parallel_attention.Proofs.Software

namespace VeriTile.Bench.Repairs.ParallelAttention.HostNative
noncomputable section
open scoped BigOperators
open HostAPI

/-- The actual kernel path, including the wrapper's explicit empty dispatch. -/
def dispatch (a : Ops α) (c : CompilerLayout) (s : Shape) (b : Bases)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat)
    (ks hs : List Event) (m : Memory α) : Memory α :=
  if HostAPI.Empty s then m else compiledRun a c s b trees scratch output ks hs m

theorem dispatch_eq (a : Ops α) (c : CompilerLayout) (s : Shape) (b : Bases)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat)
    (kinds : List OutputKind) (ks hs : List Event)
    (layout : RunLayout s b trees scratch output kinds ks hs) (m : Memory α) :
    dispatch a c s b trees scratch output ks hs m=compiledRun a c s b trees scratch output ks hs m := by
  unfold dispatch
  split_ifs with empty
  · have hk : ks=[] := List.perm_nil.mp (layout.kernelCoverage.trans
      (List.Perm.of_eq (events_empty s empty true kinds)))
    have hh : hs=[] := List.perm_nil.mp (layout.hostCoverage.trans
      (List.Perm.of_eq (events_empty s empty false kinds)))
    simp [hk,hh,compiledRun,compiledScratch,combinedHost,runTagged]
  · rfl

/-- z[...,None]+eps allocates the actual denominator retained by DivBackward.
No floating-point re-addition of z and eps is substituted in backward. -/
def denominatorRun (a : Ops α) (eps : α) (s : Shape) (z den : Nat) (m : Memory α) : Memory α :=
  runStores (fun i => den+i)
    (fun current i => if i<s.B*s.H*s.T then a.add (current (z+i)) eps else a.zero)
    (List.range (s.B*s.H*s.T)) m

def divisionRun (a : Ops α) (s : Shape) (o den dst : Nat) (m : Memory α) : Memory α :=
  runStores (fun i => dst+i)
    (fun current i => if i<s.B*s.H*s.T*s.V then
      a.div (current (o+i)) (current (den+i/s.V)) else a.zero)
    (List.range (s.B*s.H*s.T*s.V)) m

theorem denominator_output (a : Ops α) (eps : α) (s : Shape) (z den : Nat) (m : Memory α)
    (fresh : ∀ i,i<s.B*s.H*s.T → den+i∉Window.Cells ⟨z,s.B*s.H*s.T⟩)
    (i : Nat) (hi : i<s.B*s.H*s.T) :
    denominatorRun a eps s z den m (den+i)=a.add (m (z+i)) eps := by
  have h := actual_store_value (Window.Cells ⟨z,s.B*s.H*s.T⟩) (fun i => den+i)
    (fun current i => if i<s.B*s.H*s.T then a.add (current (z+i)) eps else a.zero)
    (by
      intro m m' same j
      dsimp only
      split_ifs with hj
      · rw [same ⟨Nat.le_add_right _ _,Nat.add_lt_add_left hj _⟩]
      · rfl)
    (List.range (s.B*s.H*s.T)) m (by intro j hj;exact fresh j (List.mem_range.mp hj))
    i (List.mem_range.mpr hi) (by intro j k eq;dsimp at eq;omega)
  simpa [denominatorRun,hi] using h

theorem denominator_frame (a : Ops α) (eps : α) (s : Shape) (z den : Nat) (m : Memory α)
    (address : Nat) (outside : ∀ i,i<s.B*s.H*s.T → den+i≠address) :
    denominatorRun a eps s z den m address=m address :=
  runStores_frame _ _ _ _ _ (by intro i hi;exact outside i (List.mem_range.mp hi))

theorem division_output (a : Ops α) (s : Shape) (o den dst : Nat) (m : Memory α)
    (fresh : ∀ i,i<s.B*s.H*s.T*s.V → dst+i∉normalizeReads (s.B*s.H) s.T s.V o den)
    (i : Nat) (hi : i<s.B*s.H*s.T*s.V) :
    divisionRun a s o den dst m (dst+i)=a.div (m (o+i)) (m (den+i/s.V)) := by
  have h := actual_store_value (normalizeReads (s.B*s.H) s.T s.V o den) (fun i => dst+i)
    (fun current i => if i<s.B*s.H*s.T*s.V then a.div (current (o+i)) (current (den+i/s.V)) else a.zero)
    (by
      intro m m' same j
      dsimp only
      split_ifs with hj
      · have positive : 0<s.V := by
          by_contra h
          have hz : s.V=0 := by omega
          simp [hz] at hj
        have hd := (Nat.div_lt_iff_lt_mul positive).mpr hj
        rw [same (Or.inl ⟨Nat.le_add_right _ _,Nat.add_lt_add_left hj _⟩),
          same (Or.inr ⟨Nat.le_add_right _ _,Nat.add_lt_add_left hd _⟩)]
      · rfl)
    (List.range (s.B*s.H*s.T*s.V)) m (by intro j hj;exact fresh j (List.mem_range.mp hj))
    i (List.mem_range.mpr hi) (by intro j k eq;dsimp at eq;omega)
  simpa [divisionRun,hi] using h

theorem division_frame (a : Ops α) (s : Shape) (o den dst : Nat) (m : Memory α)
    (address : Nat) (outside : ∀ i,i<s.B*s.H*s.T*s.V → dst+i≠address) :
    divisionRun a s o den dst m address=m address :=
  runStores_frame _ _ _ _ _ (by intro i hi;exact outside i (List.mem_range.mp hi))

/-- Fresh ordinary denominator/output allocations preserve the raw O/Z and
retained denominator. Only bounded physical read sets occur here. -/
structure NormalizeLayout (s : Shape) (output : OutputKind → Nat) (den dst : Nat) : Prop where
  denominatorFresh : ∀ i,i<s.B*s.H*s.T → den+i∉normalizeReads (s.B*s.H) s.T s.V (output .forward) (output .normalizer)
  outputFresh : ∀ i,i<s.B*s.H*s.T*s.V → dst+i∉
    normalizeReads (s.B*s.H) s.T s.V (output .forward) den ∪ Window.Cells ⟨output .normalizer,s.B*s.H*s.T⟩

structure ForwardResult (α : Type) where
  api : HostAPI.ForwardResult α
  denominator : Option Tensor

def forwardMemory (a : Ops α) (c : CompilerLayout) (s : Shape) (x : HostAPI.Inputs)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat)
    (ks hs : List Event) (m : Memory α) (eps : α) (den dst : Nat) (returnBoth normalize : Bool) : Memory α :=
  let raw := dispatch a c s (forwardBases x) trees scratch output ks hs m
  if returnBoth then raw else if normalize then
    divisionRun a s (output .forward) den dst (denominatorRun a eps s (output .normalizer) den raw)
  else raw

def forward (a : Ops α) (c : CompilerLayout) (s : Shape) (x : HostAPI.Inputs)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat)
    (ks hs : List Event) (m : Memory α) (eps : α) (den dst : Nat) (returnBoth normalize : Bool) : ForwardResult α :=
  ⟨⟨forwardMemory a c s x trees scratch output ks hs m eps den dst returnBoth normalize,
    save a s x,returned s output dst returnBoth normalize,
    tensor s .forward (output .forward),tensor s .normalizer (output .normalizer)⟩,
    if returnBoth then none else if normalize then some (tensor s .normalizer den) else none⟩

def backward (a : Ops α) (c : CompilerLayout) (ctx : Saved α) (do_ dz : Tensor)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat)
    (ks hs : List Event) (ready : Memory α) : HostAPI.BackwardResult α :=
  ⟨dispatch (savedOps a ctx) c ctx.shape (backwardBases ctx do_ dz) trees scratch output ks hs ready,
    gradients ctx.shape output⟩

def propagated (a : Ops α) (s : Shape) (normalize : Bool) (tree : Nat → Nat → SumPlan)
    (o den : Tensor) (dy : View) (m : Memory α) (bh : Nat) : Cotangent α :=
  if normalize then
    ⟨fun t e => if bh<s.B*s.H ∧ t<s.T ∧ e<s.V then
      a.div (viewRead a s s.V dy m bh t e) (den.read m bh t 0) else a.zero,
     fun t => if bh<s.B*s.H ∧ t<s.T then (tree bh t).eval a (fun e => if e<s.V then
      a.mul (a.neg (viewRead a s s.V dy m bh t e))
        (a.div (a.div (o.read m bh t e) (den.read m bh t 0)) (den.read m bh t 0))
      else a.zero) else a.zero⟩
  else ⟨viewRead a s s.V dy m bh,fun _ => a.zero⟩

theorem propagated_local (a : Ops α) (s : Shape) (normalize : Bool) (tree : Nat → Nat → SumPlan)
    (o den : Tensor) (dy : View) (m m' : Memory α)
    (same : Set.EqOn m m' (propagationReads s o den dy)) (bh : Nat) :
    propagated a s normalize tree o den dy m bh=propagated a s normalize tree o den dy m' bh := by
  have hdy : viewRead a s s.V dy m bh=viewRead a s s.V dy m' bh := by
    funext t e
    unfold viewRead
    split_ifs with active
    · exact same (Or.inr ⟨bh,t,e,active.1,active.2.1,active.2.2,rfl⟩)
    · rfl
  have ho (t e : Nat) (hb : bh<s.B*s.H) (ht : t<s.T) (he : e<s.V) : o.read m bh t e=o.read m' bh t e :=
    same (Or.inl (Or.inl ⟨bh,t,e,hb,ht,he,rfl⟩))
  have hd (t : Nat) (hb : bh<s.B*s.H) (ht : t<s.T) : den.read m bh t 0=den.read m' bh t 0 :=
    same (Or.inl (Or.inr ⟨bh,t,hb,ht,rfl⟩))
  cases normalize with
  | false => simp only [propagated,Bool.false_eq_true,if_false,hdy]
  | true =>
    unfold propagated
    simp only [if_true,hdy]
    congr 1
    · funext t e
      split_ifs with active
      · rw [hd t active.1 active.2.1]
      · rfl
    · funext t
      split_ifs with active
      · rw [hd t active.1 active.2]
        congr 1
        funext e
        split_ifs with he
        · rw [ho t e active.1 active.2 he]
        · rfl
      · rfl

def cotangentValue (a : Ops α) (s : Shape) (normalize : Bool) (tree : Nat → Nat → SumPlan)
    (o den : Tensor) (dy : View) (m : Memory α) (ev : Event) : α :=
  let D := ev.1.width s
  let bh := ev.2/D/s.T%(s.B*s.H)
  let t := ev.2/D%s.T
  let e := ev.2%D
  match ev.1 with
  | .forward => (propagated a s normalize tree o den dy m bh).output t e
  | .normalizer => (propagated a s normalize tree o den dy m bh).normalizer t
  | _ => a.zero

def propagateRun (a : Ops α) (s : Shape) (normalize : Bool) (tree : Nat → Nat → SumPlan)
    (o den : Tensor) (dy : View) (m : Memory α) (dest : OutputKind → Nat) : Memory α :=
  runTagged (tensorAddress dest) (fun current ev => cotangentValue a s normalize tree o den dy current ev)
    (events s false forwardKinds) m

theorem propagate_output (a : Ops α) (s : Shape) (normalize : Bool) (tree : Nat → Nat → SumPlan)
    (o den : Tensor) (dy : View) (m : Memory α) (dest : OutputKind → Nat)
    (layout : PropagationLayout s dest) (fresh : PropagationFresh s o den dy dest)
    (bh t e : Nat) (hb : bh<s.B*s.H) (ht : t<s.T) (he : e<s.V) :
    propagateRun a s normalize tree o den dy m dest (dest .forward+dense s.T s.V bh t e)=
      (propagated a s normalize tree o den dy m bh).output t e := by
  have h := tagged_output (propagationReads s o den dy) (tensorAddress dest)
    (fun current ev => cotangentValue a s normalize tree o den dy current ev)
    (by intro m m' same ev;unfold cotangentValue;simp_rw [propagated_local a s normalize tree o den dy m m' same])
    (events s false forwardKinds) m fresh layout (.forward,dense s.T s.V bh t e)
    ((events_member _ _ _ _ _).mpr ⟨by simp [forwardKinds],dense_bound _ _ _ _ _ _ hb ht he⟩)
  have hd := scratch_decode (s.B*s.H) s.T s.V 0 bh t e hb ht he
  simp only [zero_mul,zero_add] at hd
  simpa [propagateRun,tensorAddress,cotangentValue,OutputKind.width,hd.1,hd.2.1,hd.2.2.1] using h

theorem propagate_normalizer (a : Ops α) (s : Shape) (normalize : Bool) (tree : Nat → Nat → SumPlan)
    (o den : Tensor) (dy : View) (m : Memory α) (dest : OutputKind → Nat)
    (layout : PropagationLayout s dest) (fresh : PropagationFresh s o den dy dest)
    (bh t : Nat) (hb : bh<s.B*s.H) (ht : t<s.T) :
    propagateRun a s normalize tree o den dy m dest (dest .normalizer+dense s.T 1 bh t 0)=
      (propagated a s normalize tree o den dy m bh).normalizer t := by
  have h := tagged_output (propagationReads s o den dy) (tensorAddress dest)
    (fun current ev => cotangentValue a s normalize tree o den dy current ev)
    (by intro m m' same ev;unfold cotangentValue;simp_rw [propagated_local a s normalize tree o den dy m m' same])
    (events s false forwardKinds) m fresh layout (.normalizer,dense s.T 1 bh t 0)
    ((events_member _ _ _ _ _).mpr ⟨by simp [forwardKinds],dense_bound _ _ _ _ _ _ hb ht (by omega)⟩)
  have hd := scratch_decode (s.B*s.H) s.T 1 0 bh t 0 hb ht (by omega)
  simp only [zero_mul,zero_add] at hd
  simpa only [propagateRun,tensorAddress,cotangentValue,OutputKind.width,hd.1,hd.2.1,hd.2.2.1] using h

theorem propagate_frame (a : Ops α) (s : Shape) (normalize : Bool) (tree : Nat → Nat → SumPlan)
    (o den : Tensor) (dy : View) (m : Memory α) (dest : OutputKind → Nat) (address : Nat)
    (outside : ∀ ev∈events s false forwardKinds,tensorAddress dest ev≠address) :
    propagateRun a s normalize tree o den dy m dest address=m address := tagged_frame _ _ _ _ _ outside

/-- These FP endpoints follow the same selected compiled syntax and actual
returned handles. Only fixed-arity scalar agreement is numerical. -/
theorem fp32_forward (native : Ops FP32Word) (primitives : ScalarAgreement native softwareOps)
    (c : CompilerLayout) (s : Shape) (x : HostAPI.Inputs)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat) (ks hs : List Event)
    (entry : Memory FP32Word) (eps : FP32Word) (den dst : Nat) (returnBoth normalize : Bool) :
    forward native c s x trees scratch output ks hs entry eps den dst returnBoth normalize=
      forward softwareOps c s x trees scratch output ks hs entry eps den dst returnBoth normalize := by
  rw [primitives.eq]

theorem fp32_backward (native : Ops FP32Word) (primitives : ScalarAgreement native softwareOps)
    (c : CompilerLayout) (ctx : Saved FP32Word) (do_ dz : Tensor)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat) (ks hs : List Event)
    (ready : Memory FP32Word) :
    backward native c ctx do_ dz trees scratch output ks hs ready=
      backward softwareOps c ctx do_ dz trees scratch output ks hs ready := by
  rw [primitives.eq]

theorem fp32_propagation (native : Ops FP32Word) (primitives : ScalarAgreement native softwareOps)
    (s : Shape) (normalize : Bool) (tree : Nat → Nat → SumPlan)
    (o den : Tensor) (dy : View) (later : Memory FP32Word) (dest : OutputKind → Nat) :
    propagateRun native s normalize tree o den dy later dest=
      propagateRun softwareOps s normalize tree o den dy later dest := by
  rw [primitives.eq]

def singleBackward (a : Ops α) (c : CompilerLayout) (f : ForwardResult α)
    (normalize : Bool) (tree : Nat → Nat → SumPlan) (dy : View) (later : Memory α)
    (dest : OutputKind → Nat) (trees : OutputKind → SumPlan)
    (scratch output : OutputKind → Nat) (ks hs : List Event) : HostAPI.BackwardResult α :=
  let s := f.api.saved.shape
  let den := f.denominator.getD f.api.rawNormalizer
  let ready := propagateRun a s normalize tree f.api.rawOutput den dy later dest
  backward a c f.api.saved (tensor s .forward (dest .forward)) (tensor s .normalizer (dest .normalizer))
    trees scratch output ks hs ready

theorem fp32_single_backward (native : Ops FP32Word) (primitives : ScalarAgreement native softwareOps)
    (c : CompilerLayout) (f : ForwardResult FP32Word) (normalize : Bool)
    (tree : Nat → Nat → SumPlan) (dy : View) (later : Memory FP32Word) (dest : OutputKind → Nat)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat) (ks hs : List Event) :
    singleBackward native c f normalize tree dy later dest trees scratch output ks hs=
      singleBackward softwareOps c f normalize tree dy later dest trees scratch output ks hs := by
  rw [primitives.eq]

theorem forward_frame (a : Ops α) (c : CompilerLayout) (s : Shape) (x : HostAPI.Inputs)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat) (ks hs : List Event)
    (entry : Memory α) (eps : α) (den dst : Nat) (returnBoth normalize : Bool) (address : Nat)
    (outsideKernel : ∀ ev∈ks,tensorAddress scratch ev≠address)
    (outsideHost : ∀ ev∈hs,tensorAddress output ev≠address)
    (outsideDen : normalize=true → returnBoth=false → ∀ i,i<s.B*s.H*s.T → den+i≠address)
    (outsideOutput : normalize=true → returnBoth=false → ∀ i,i<s.B*s.H*s.T*s.V → dst+i≠address) :
    (forward a c s x trees scratch output ks hs entry eps den dst returnBoth normalize).api.memory address=entry address := by
  have raw : dispatch a c s (forwardBases x) trees scratch output ks hs entry address=entry address := by
    unfold dispatch
    split_ifs
    · rfl
    · exact compiled_frame _ _ _ _ _ _ _ _ _ _ _ outsideKernel outsideHost
  cases returnBoth <;> cases normalize <;>
    simp only [forward,forwardMemory,Bool.false_eq_true,if_false,if_true]
  · exact raw
  · rw [division_frame _ _ _ _ _ _ _ (outsideOutput rfl rfl),
      denominator_frame _ _ _ _ _ _ _ (outsideDen rfl rfl)]
    exact raw
  · exact raw
  · exact raw

theorem backward_frame (a : Ops α) (c : CompilerLayout) (ctx : Saved α) (do_ dz : Tensor)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat) (ks hs : List Event)
    (ready : Memory α) (address : Nat)
    (outsideKernel : ∀ ev∈ks,tensorAddress scratch ev≠address)
    (outsideHost : ∀ ev∈hs,tensorAddress output ev≠address) :
    (backward a c ctx do_ dz trees scratch output ks hs ready).memory address=ready address := by
  dsimp only [backward]
  unfold dispatch
  split_ifs
  · rfl
  · exact compiled_frame _ _ _ _ _ _ _ _ _ _ _ outsideKernel outsideHost

theorem empty_forward (a : Ops α) (c : CompilerLayout) (s : Shape) (x : HostAPI.Inputs)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat) (ks hs : List Event)
    (entry : Memory α) (eps : α) (den dst : Nat) (returnBoth normalize : Bool) (empty : HostAPI.Empty s) :
    (forward a c s x trees scratch output ks hs entry eps den dst returnBoth normalize).api.memory=entry := by
  have noDispatch : dispatch a c s (forwardBases x) trees scratch output ks hs entry=entry := if_pos empty
  cases returnBoth <;> cases normalize <;>
    simp only [forward,forwardMemory,Bool.false_eq_true,if_false,if_true,noDispatch,
      denominatorRun,divisionRun,empty,zero_mul,List.range_zero,runStores]

theorem empty_backward (a : Ops α) (c : CompilerLayout) (ctx : Saved α) (do_ dz : Tensor)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat) (ks hs : List Event)
    (ready : Memory α) (empty : HostAPI.Empty ctx.shape) :
    (backward a c ctx do_ dz trees scratch output ks hs ready).memory=ready ∧
    (backward a c ctx do_ dz trees scratch output ks hs ready).returned.scale=none := by
  exact ⟨if_pos empty,rfl⟩

theorem forward_preserves_saved (a : Ops α) (c : CompilerLayout) (s : Shape) (x : HostAPI.Inputs)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat) (ks hs : List Event)
    (entry : Memory α) (eps : α) (den dst : Nat) (returnBoth normalize : Bool)
    (kernelFresh : ∀ ev∈ks,tensorAddress scratch ev∉inputCellsOnly s x)
    (hostFresh : ∀ ev∈hs,tensorAddress output ev∉inputCellsOnly s x)
    (denFresh : normalize=true → returnBoth=false → ∀ i,i<s.B*s.H*s.T → den+i∉inputCellsOnly s x)
    (dstFresh : normalize=true → returnBoth=false → ∀ i,i<s.B*s.H*s.T*s.V → dst+i∉inputCellsOnly s x) :
    SavedValid (save a s x) entry
      (forward a c s x trees scratch output ks hs entry eps den dst returnBoth normalize).api.memory := by
  intro address ha
  apply forward_frame
  · intro ev he eq
    exact kernelFresh ev he (eq ▸ ha)
  · intro ev he eq
    exact hostFresh ev he (eq ▸ ha)
  · intro hn hb i hi eq
    exact denFresh hn hb i hi (eq ▸ ha)
  · intro hn hb i hi eq
    exact dstFresh hn hb i hi (eq ▸ ha)

theorem single_frame (a : Ops α) (c : CompilerLayout) (f : ForwardResult α)
    (normalize : Bool) (tree : Nat → Nat → SumPlan) (dy : View) (later : Memory α)
    (dest : OutputKind → Nat) (trees : OutputKind → SumPlan)
    (scratch output : OutputKind → Nat) (ks hs : List Event) (address : Nat)
    (outsidePropagation : ∀ ev∈events f.api.saved.shape false forwardKinds,tensorAddress dest ev≠address)
    (outsideKernel : ∀ ev∈ks,tensorAddress scratch ev≠address)
    (outsideHost : ∀ ev∈hs,tensorAddress output ev≠address) :
    (singleBackward a c f normalize tree dy later dest trees scratch output ks hs).memory address=later address := by
  unfold singleBackward
  rw [backward_frame _ _ _ _ _ _ _ _ _ _ _ _ outsideKernel outsideHost]
  exact propagate_frame _ _ _ _ _ _ _ _ _ _ outsidePropagation

theorem empty_single_backward (a : Ops α) (c : CompilerLayout) (f : ForwardResult α)
    (normalize : Bool) (tree : Nat → Nat → SumPlan) (dy : View) (later : Memory α)
    (dest : OutputKind → Nat) (trees : OutputKind → SumPlan)
    (scratch output : OutputKind → Nat) (ks hs : List Event) (empty : HostAPI.Empty f.api.saved.shape) :
    (singleBackward a c f normalize tree dy later dest trees scratch output ks hs).memory=later ∧
    (singleBackward a c f normalize tree dy later dest trees scratch output ks hs).returned.scale=none := by
  constructor
  · unfold singleBackward
    rw [(empty_backward _ _ _ _ _ _ _ _ _ _ _ empty).1]
    simp only [propagateRun,events_empty _ empty,runTagged]
  · rfl

theorem fp32_returned_context (native : Ops FP32Word) (primitives : ScalarAgreement native softwareOps)
    (c : CompilerLayout) (s : Shape) (x : HostAPI.Inputs)
    (fTrees : OutputKind → SumPlan) (fScratch fOutput : OutputKind → Nat) (fks fhs : List Event)
    (entry : Memory FP32Word) (eps : FP32Word) (den dst : Nat) (normalize : Bool)
    (tree : Nat → Nat → SumPlan) (dy : View) (later : Memory FP32Word) (dest : OutputKind → Nat)
    (bTrees : OutputKind → SumPlan) (bScratch bOutput : OutputKind → Nat) (bks bhs : List Event) :
    let nativeForward := forward native c s x fTrees fScratch fOutput fks fhs entry eps den dst false normalize
    let referenceForward := forward softwareOps c s x fTrees fScratch fOutput fks fhs entry eps den dst false normalize
    singleBackward native c nativeForward normalize tree dy later dest bTrees bScratch bOutput bks bhs=
      singleBackward softwareOps c referenceForward normalize tree dy later dest bTrees bScratch bOutput bks bhs := by
  rw [primitives.eq]


end
end VeriTile.Bench.Repairs.ParallelAttention.HostNative
