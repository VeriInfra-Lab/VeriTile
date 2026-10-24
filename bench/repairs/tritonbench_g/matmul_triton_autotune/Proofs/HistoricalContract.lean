import VeriTile.Torch.Tensor
import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.Wrapper

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Checked
open VeriTile.Triton FP.Structural Whole
noncomputable section

structure Metadata where
  a : VeriTile.Torch.TensorMetadata
  b : VeriTile.Torch.TensorMetadata
  request : Request

def metadataChecks (p : Metadata) : Bool :=
  decide (p.a.layout=.strided ∧ p.b.layout=.strided ∧
    p.a.shape=[p.request.rows,p.request.inner] ∧ p.b.shape=[p.request.innerB,p.request.columns] ∧
    p.a.dtype=.floating .fp32 ∧ p.b.dtype=.floating .fp32 ∧
    p.a.device.kind="cuda" ∧ p.b.device.kind="cuda" ∧ p.a.device=p.b.device ∧
    p.request.inner=p.request.innerB ∧ p.request.aContiguous=Bool.true ∧ p.a.isContiguous=Bool.true ∧
    p.a.isNeg=Bool.false ∧ p.b.isNeg=Bool.false ∧
    (p.a.numel=0 ∨ p.a.dataPtr.toNat%4=0) ∧ (p.b.numel=0 ∨ p.b.dataPtr.toNat%4=0))

inductive Error where | dynamic | metadata deriving DecidableEq, Repr

def checkInputs (p : Metadata) : Except Error Unit :=
  if metadataChecks p then .ok () else .error .metadata
abbrev InputContract (p : Metadata) := metadataChecks p=Bool.true

theorem checkInputs_iff (p : Metadata) : checkInputs p=.ok () ↔ InputContract p := by
  simp [checkInputs,InputContract]

/-- Dynamic classification checks Tensor/layout/rank/dtype/CUDA, raw/conjugate
views, nonnegative strides and builtin-string activation before metadata use. -/
inductive Arguments where | unsupported | tensors (p : Metadata)
def checkArguments : Arguments → Except Error Unit
  | .unsupported => .error .dynamic
  | .tensors p => checkInputs p
def AdmittedArguments : Arguments → Prop
  | .unsupported => False
  | .tensors p => InputContract p
theorem checkArguments_iff (args : Arguments) : checkArguments args=.ok () ↔ AdmittedArguments args := by
  cases args <;> simp [checkArguments,AdmittedArguments,checkInputs_iff]

/-- Ordinary observed layout/storage/fresh-allocation facts. Input alignment,
shape compatibility and A contiguity are derived from the executed checks. -/
structure OrdinaryTensor {α : Type} (p : Metadata) (s : State α) where
  extentA : Nat
  extentB : Nat
  baseA : Nat
  baseB : Nat
  baseC : Nat
  metadataA : TensorMetadata
  metadataB : TensorMetadata
  metadataC : TensorMetadata
  shapeA : metadataA.shape=[p.request.rows,p.request.inner]
  shapeB : metadataB.shape=[p.request.innerB,p.request.columns]
  shapeC : metadataC.shape=[p.request.rows,p.request.columns]
  stridesA : metadataA.strides=[p.request.strideAM,p.request.strideAK]
  stridesB : metadataB.strides=[p.request.strideBK,p.request.strideBN]
  stridesC : metadataC.strides=[max p.request.columns 1,1]
  dtypeA : metadataA.dtype=.fp32
  dtypeB : metadataB.dtype=.fp32
  dtypeC : metadataC.dtype=.fp16
  deviceB : metadataB.device=metadataA.device
  deviceC : metadataC.device=metadataA.device
  selected : p.request.config ∈ configurations p.request.target
  freshA : p.request.C≠p.request.A
  freshB : p.request.C≠p.request.B
  physicalFreshA : byteSpansDisjoint baseC (2*(p.request.rows*p.request.columns)) baseA (4*extentA)
  physicalFreshB : byteSpansDisjoint baseC (2*(p.request.rows*p.request.columns)) baseB (4*extentB)
  alignmentC : baseC%2=0
  spanA : baseA+4*extentA≤2^64
  spanB : baseB+4*extentB≤2^64
  spanC : baseC+2*(p.request.rows*p.request.columns)≤2^64
  allocationA : 4*extentA<2^63
  allocationB : 4*extentB<2^63
  allocationC : 2*(p.request.rows*p.request.columns)<2^63
  denseA : p.request.rows*p.request.inner≤extentA
  strideAM : p.request.strideAM<2^63
  strideAK : p.request.strideAK<2^63
  strideBK : p.request.strideBK<2^63
  strideBN : p.request.strideBN<2^63
  inputA : ∀ row<p.request.rows, ∀ k<p.request.inner, row*p.request.strideAM+k*p.request.strideAK<extentA
  inputB : ∀ k<p.request.inner, ∀ col<p.request.columns, k*p.request.strideBK+col*p.request.strideBN<extentB
  tagA : ∀ row<p.request.rows, ∀ k<p.request.inner, ∃ x, s.mem p.request.A (row*p.request.strideAM+k*p.request.strideAK)=Cell.mk .fp32 x
  tagB : ∀ k<p.request.inner, ∀ col<p.request.columns, ∃ x, s.mem p.request.B (k*p.request.strideBK+col*p.request.strideBN)=Cell.mk .fp32 x
  sameViewOrigin : p.request.A=p.request.B → baseA=baseB
  aliasConsistency : ∀ row<p.request.rows, ∀ k<p.request.inner, ∀ k'<p.request.inner, ∀ col<p.request.columns,
    baseA+4*(row*p.request.strideAM+k*p.request.strideAK)=baseB+4*(k'*p.request.strideBK+col*p.request.strideBN) →
    s.mem p.request.A (row*p.request.strideAM+k*p.request.strideAK)=s.mem p.request.B (k'*p.request.strideBK+col*p.request.strideBN)
  pointerA : baseA=p.a.dataPtr.toNat
  pointerB : baseB=p.b.dataPtr.toNat
  emptyA : p.a.numel=0 → p.a.dataPtr.toNat=0
  emptyB : p.b.numel=0 → p.b.dataPtr.toNat=0

def OrdinaryTensor.admitted {α : Type} {p : Metadata} {s : State α}
    (ordinary : OrdinaryTensor p s) (admitted : InputContract p) : TensorAPI p.request s := by
  simp only [InputContract,metadataChecks,decide_eq_true_eq] at admitted
  obtain ⟨_,_,_,_,_,_,_,_,_,dimensions,contiguous,_,_,_,alignA,alignB⟩ := admitted
  have ha : p.a.dataPtr.toNat%4=0 := by
    rcases alignA with h|h
    · simp [ordinary.emptyA h]
    · exact h
  have hb : p.b.dataPtr.toNat%4=0 := by
    rcases alignB with h|h
    · simp [ordinary.emptyB h]
    · exact h
  exact {
    extentA := ordinary.extentA
    extentB := ordinary.extentB
    baseA := ordinary.baseA
    baseB := ordinary.baseB
    baseC := ordinary.baseC
    metadataA := ordinary.metadataA
    metadataB := ordinary.metadataB
    metadataC := ordinary.metadataC
    shapeA := ordinary.shapeA
    shapeB := ordinary.shapeB
    shapeC := ordinary.shapeC
    stridesA := ordinary.stridesA
    stridesB := ordinary.stridesB
    stridesC := ordinary.stridesC
    dtypeA := ordinary.dtypeA
    dtypeB := ordinary.dtypeB
    dtypeC := ordinary.dtypeC
    deviceB := ordinary.deviceB
    deviceC := ordinary.deviceC
    selected := ordinary.selected
    freshA := ordinary.freshA
    freshB := ordinary.freshB
    physicalFreshA := ordinary.physicalFreshA
    physicalFreshB := ordinary.physicalFreshB
    alignmentC := ordinary.alignmentC
    spanA := ordinary.spanA
    spanB := ordinary.spanB
    spanC := ordinary.spanC
    allocationA := ordinary.allocationA
    allocationB := ordinary.allocationB
    allocationC := ordinary.allocationC
    denseA := ordinary.denseA
    strideAM := ordinary.strideAM
    strideAK := ordinary.strideAK
    strideBK := ordinary.strideBK
    strideBN := ordinary.strideBN
    inputA := ordinary.inputA
    inputB := ordinary.inputB
    tagA := ordinary.tagA
    tagB := ordinary.tagB
    sameViewOrigin := ordinary.sameViewOrigin
    aliasConsistency := ordinary.aliasConsistency
    dimensions := dimensions
    contiguousA := contiguous
    alignmentA := by rw [ordinary.pointerA]; exact ha
    alignmentB := by rw [ordinary.pointerB]; exact hb }

/-- Original wrapper preparation uses only allocation on this raw domain. -/
structure API (α : Type) where
  allocateC : State α → Except String (RegionName × State α)
  compileLaunch : State α → Except BackendError (List Config)

def API.legacy {α : Type} (api : API α) : WrapperAPI α :=
  ⟨Except.ok,Except.ok,api.allocateC,api.compileLaunch⟩

theorem prepare_no_copy {α : Type} (api : API α) (r : Request) (before : State α) :
    prepare api.legacy r before=
      if r.inner≠r.innerB then .error .incompatibleDimensions
      else if r.aContiguous=Bool.false then .error .noncontiguousA
      else (api.allocateC before).mapError WrapperError.allocation := by
  unfold prepare API.legacy
  split_ifs <;> simp_all
  cases api.allocateC before <;> rfl

/-- Allocation preservation and caller logical observations produce the old
resolution relation with both copy flags false and the original input regions. -/
def noCopyResolve {α : Type} [Inhabited α] {p : Metadata}
    (admitted : InputContract p) (before prepared : State α) (ordinary : OrdinaryTensor p prepared)
    (A : Fin p.request.rows → Fin p.request.inner → α)
    (B : Fin p.request.inner → Fin p.request.columns → α)
    (hA : ∀ i k,inputA p.request before i.val k.val=A i k)
    (hB : ∀ k j,inputB p.request before k.val j.val=B k j)
    (frame : ∀ region offset, ¬Footprint p.request region offset →
      prepared.mem region offset=before.mem region offset) :
    ResolveNegContract p.request before prepared (ordinary.admitted admitted) A B := by
  let t := ordinary.admitted admitted
  exact {
    copyA := Bool.false
    copyB := Bool.false
    originalA := p.request.A
    originalB := p.request.B
    originalBaseA := t.baseA
    originalBaseB := t.baseB
    originalExtentA := t.extentA
    originalExtentB := t.extentB
    outputByteFreshA := t.physicalFreshA
    outputByteFreshB := t.physicalFreshB
    copyAByteFreshA := by simp
    copyAByteFreshB := by simp
    copyBByteFreshA := by simp
    copyBByteFreshB := by simp
    outputFreshA := t.freshA
    outputFreshB := t.freshB
    copyAFreshA := by simp
    copyAFreshB := by simp
    copyBFreshA := by simp
    copyBFreshB := by simp
    logicalA := by
      intro i k
      rw [inputA,frame _ _ (fun h => t.freshA h.1.symm)]
      exact hA i k
    logicalB := by
      intro k j
      rw [inputB,frame _ _ (fun h => t.freshB h.1.symm)]
      exact hB k j
    frame := fun region offset outside _ => frame region offset outside }

/-- Successful allocator and autotuner API facts only. No assumption asserts
kernel success, values, coverage, index safety or the final caller frame. -/
structure Library {α : Type} [Inhabited α] (api : API α) (p : Metadata) (before : State α) where
  allocation : ∀ c prepared, api.allocateC before=.ok (c,prepared) →
    c=p.request.C ∧ ∃ ordinary : OrdinaryTensor p prepared,
      ∀ region offset, ¬Footprint p.request region offset →
        prepared.mem region offset=before.mem region offset
  trialsLegal : ∀ prepared trials, api.compileLaunch prepared=.ok trials →
    ∀ config∈trials, config∈configurations p.request.target
  trialABI : ∀ prepared trials, api.compileLaunch prepared=.ok trials →
    ∀ config∈trials, programs (withConfig p.request config)<2^31
  launchABI : ∀ prepared trials, api.compileLaunch prepared=.ok trials → programs p.request<2^31

def Library.legacy {α : Type} [Inhabited α] {api : API α} {p : Metadata} {before : State α}
    (library : Library api p before) (admitted : InputContract p)
    (A : Fin p.request.rows → Fin p.request.inner → α)
    (B : Fin p.request.inner → Fin p.request.columns → α)
    (hA : ∀ i k,inputA p.request before i.val k.val=A i k)
    (hB : ∀ k j,inputB p.request before k.val j.val=B k j) :
    WrapperAPIContract api.legacy p.request before A B := by
  refine ⟨?_,library.trialsLegal,library.trialABI,library.launchABI⟩
  intro c prepared h
  obtain ⟨a,b,ha,hb,halloc⟩ := h.stages
  change Except.ok before=Except.ok a at ha
  have ea := Except.ok.inj ha
  subst a
  change Except.ok before=Except.ok b at hb
  have eb := Except.ok.inj hb
  subst b
  obtain ⟨identity,ordinary,frame⟩ := library.allocation c prepared halloc
  exact ⟨identity,ordinary.admitted admitted,⟨noCopyResolve admitted before prepared ordinary A B hA hB frame⟩⟩

/-- Full caller frame for the no-copy checked wrapper, with no copy-footprint exclusion. -/
theorem full_frame {α : Type} [Inhabited α]
    {M : Algebra α} {D : FPValue.DotAcc α} {Q : Numerics α}
    {api : API α} {p : Metadata} {before : State α}
    (library : Library api p before)
    {A : Fin p.request.rows → Fin p.request.inner → α}
    {B : Fin p.request.inner → Fin p.request.columns → α}
    {c : RegionName} {final : State α}
    (success : WrapperSuccess M D Q api.legacy p.request before A B c final) :
    ∀ region offset, ¬Footprint p.request region offset → final.mem region offset=before.mem region offset := by
  obtain ⟨_,prepared,_,_,done,_,frame,_⟩ := success
  obtain ⟨a,b,ha,hb,halloc⟩ := done.stages
  change Except.ok before=Except.ok a at ha
  have ea := Except.ok.inj ha
  subst a
  change Except.ok before=Except.ok b at hb
  have eb := Except.ok.inj hb
  subst b
  obtain ⟨_,_,allocationFrame⟩ := library.allocation c prepared halloc
  intro region offset outside
  exact (frame region offset outside).trans (allocationFrame region offset outside)

def run {α : Type} [Inhabited α] (M : Algebra α) (D : FPValue.DotAcc α)
    (api : API α) (p : Metadata) (before : State α) :
    Except (Sum Error WrapperError) (RegionName × State α) :=
  match checkInputs p with
  | .error e => .error (.inl e)
  | .ok () => (wrapper M D api.legacy p.request before).mapError Sum.inr

#axiomsClean checkArguments_iff
#axiomsClean prepare_no_copy
#axiomsClean noCopyResolve
#axiomsClean Library.legacy
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Checked
