import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelContract
/-! Executable metadata admission. All quantification over roles is implemented
by finite lists. No source execution, arithmetic result, or host success occurs
in this checker. Scale type/conversion admission is handled by KernelCompiler. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelCheck
open VeriTile.Torch KernelProgram KernelContract
set_option maxHeartbeats 12000000
set_option maxRecDepth 20000

def allRegions : List Region := [.q,.k,.v,.beta,.h0,.o,.ht,.do_,.dht,.dq,.dk,.dv,.dbeta,.dh0,.retq,.retk,.retv,.retbeta]
theorem member_all (r : Region) : r∈allRegions := by cases r <;> simp [allRegions]
def domainCheck (s : Shape) (f : Flags) : Bool :=
  decide (0<s.B ∧ 0<s.H ∧ 0<s.T ∧ 0<s.K ∧ 0<s.V ∧ s.BH≤65535 ∧
    s.FNV<2^31 ∧ s.BNV<2^31 ∧ s.BK*s.BBV≤1048576) &&
  allRegions.all (fun r => decide (span s f r<2^31)) &&
  decide (f.initialGrad=Bool.true → f.initial=Bool.true) &&
  decide (f.terminalGrad=Bool.true → f.final=Bool.true) &&
  decide (f.defaultBeta=Bool.true → f.vectorBeta=Bool.false)
def tensorCheck (s : Shape) (f : Flags) (r : Region) (device : Device) (o : Observation) : Bool :=
  decide (o.metadata.layout=.strided ∧ o.metadata.dtype=.floating .fp32 ∧ o.metadata.device.kind="cuda" ∧
    o.metadata.isContiguous=Bool.true ∧ o.metadata.isNeg=Bool.false ∧
    (0<o.metadata.numel → o.metadata.dataPtr.toNat%4=0) ∧
    o.metadata.shape=dimensions s f r ∧ o.strides=denseStrides (dimensions s f r) ∧
    o.isConj=Bool.false ∧ o.metadata.device=device ∧
    4*max (max s.K s.V) 1≤o.metadata.dataPtr.toNat ∧ o.metadata.dataPtr.toNat+4*span s f r+4≤2^64)
def separateCheck (kind : Kind) (f : Flags) (o : Region → Observation) : Bool :=
  (roles kind f).all fun r => (roles kind f).all fun q => decide (r=q) ||
    decide ((o r).metadata.dataPtr.toNat+4*(o r).metadata.numel≤(o q).metadata.dataPtr.toNat ∨
      (o q).metadata.dataPtr.toNat+4*(o q).metadata.numel≤(o r).metadata.dataPtr.toNat)
def checkInput (kind : Kind) (s : Shape) (f : Flags) (o : Region → Observation) : Bool :=
  domainCheck s f && (roles kind f).all (fun r => tensorCheck s f r (o .q).metadata.device (o r)) &&
    separateCheck kind f o

theorem domainCheck_iff (s : Shape) (f : Flags) : domainCheck s f=Bool.true ↔ Domain s f := by
  have fields : Domain s f ↔
      (0<s.B ∧ 0<s.H ∧ 0<s.T ∧ 0<s.K ∧ 0<s.V ∧ s.BH≤65535 ∧
        s.FNV<2^31 ∧ s.BNV<2^31 ∧ s.BK*s.BBV≤1048576) ∧
      (∀ r,span s f r<2^31) ∧
      (f.initialGrad=Bool.true → f.initial=Bool.true) ∧
      (f.terminalGrad=Bool.true → f.final=Bool.true) ∧
      (f.defaultBeta=Bool.true → f.vectorBeta=Bool.false) := by
    constructor
    · intro h
      exact ⟨⟨h.positive.1,h.positive.2.1,h.positive.2.2.1,h.positive.2.2.2.1,h.positive.2.2.2.2,
        h.gridZ,h.gridForward,h.gridBackward,h.block⟩,h.span,h.initialGradient,h.terminalGradient,h.defaultBeta⟩
    · rintro ⟨⟨b,h,t,k,v,z,gf,gb,block⟩,bounds,ig,tg,db⟩
      exact ⟨⟨⟨b,h,t,k,v⟩,z,gf,gb,bounds⟩,block,ig,tg,db⟩
  rw [fields]
  simp only [domainCheck,Bool.and_eq_true,decide_eq_true_eq,List.all_eq_true,
    member_all,true_implies,and_assoc]
theorem tensorCheck_iff (s : Shape) (f : Flags) (r : Region) (device : Device) (o : Observation) :
    tensorCheck s f r device o=Bool.true ↔ TensorCheck s f r device o := by
  simp only [tensorCheck,decide_eq_true_eq]
  constructor
  · rintro ⟨layout,dtype,cuda,dense,neg,aligned,shape,strides,conj,dev,lower,window⟩
    exact ⟨⟨layout,dtype,cuda,fun _ => dense,neg,aligned⟩,shape,strides,conj,dev,lower,window⟩
  · intro h
    exact ⟨h.layout,h.dtype,h.cuda,h.dense rfl,h.noNeg,h.alignment,h.shape,h.strides,h.noConj,h.device,h.lower,h.window⟩
theorem checkInput_iff (kind : Kind) (s : Shape) (f : Flags) (o : Region → Observation) :
    checkInput kind s f o=Bool.true ↔ InputContract kind s f o := by
  simp only [checkInput,Bool.and_eq_true,domainCheck_iff,List.all_eq_true,tensorCheck_iff,
    separateCheck,Bool.or_eq_true,decide_eq_true_eq]
  constructor
  · rintro ⟨⟨domain,tensors⟩,apart⟩
    exact ⟨domain,tensors,fun r hr q hq different => (apart r hr q hq).resolve_left different⟩
  · intro h
    refine ⟨⟨h.domain,h.tensors⟩,?_⟩
    intro r hr q hq
    by_cases same : r=q
    · exact Or.inl same
    · exact Or.inr (h.separated r hr q hq same)

#axiomsClean domainCheck_iff
#axiomsClean tensorCheck_iff
#axiomsClean checkInput_iff
#specNonCircular checkInput avoiding [VeriTile.Triton.FP.Structural.exec,VeriTile.Triton.FP.Structural.run]
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelCheck
