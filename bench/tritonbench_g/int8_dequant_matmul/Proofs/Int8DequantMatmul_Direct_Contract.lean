import Mathlib
import bench.tritonbench_g.int8_dequant_matmul.Proofs.Contract
import bench.tritonbench_g.int8_dequant_matmul.Proofs.Source
namespace VeriTile.Bench.Repairs.Int8DequantMatmul.Direct
set_option maxHeartbeats 1500000
set_option maxRecDepth 10000

theorem bias_iff (a b t : TensorInfo) : BiasContract a b t ↔
    (t.metadata.layout=.strided ∧ t.metadata.dtype=.floating .fp32 ∧ t.metadata.shape=[cols b] ∧ t.metadata.device=a.metadata.device) ∧
    (t.metadata.isContiguous=Bool.true ∧ t.metadata.isNeg=Bool.false) ∧
    (t.storage≠a.storage ∧ t.storage≠b.storage) ∧ Aligned t := by
  constructor
  · intro h;exact ⟨h.descriptor,h.direct,h.separated,h.aligned⟩
  · rintro ⟨a,b,c,d⟩;exact ⟨a,b,c,d⟩

theorem checkBias_iff (a b t : TensorInfo) : checkBias a b t=.ok () ↔ BiasContract a b t := by
  rw [bias_iff]
  simp [checkBias,Aligned,Except.bind,Except.pure,bind,pure,ite_eq_iff,and_assoc]

theorem launch_iff (a b x w : TensorInfo) (bias : Option TensorInfo) : LaunchContract a b x w bias ↔
    (rows a<2^61 ∧ cols b<2^61 ∧ cols a<2^61) ∧
    ((rows a+15)/16)*((cols b+31)/32)≤2^31-1 ∧
    a.offset+rows a*cols a+cols a+4095<2^63 ∧ b.offset+(cols a+4095)*cols b+cols b<2^63 ∧
    2*((rows a+255)*cols b+cols b+255)<2^63 ∧ 4*(x.offset+rows a)<2^63 ∧
    4*(w.offset+cols b)<2^63 ∧ 4*(biasOffset bias+cols b+255)<2^63 := by
  constructor
  · intro h;exact ⟨h.dimensions,h.grid,h.aFormed,h.bFormed,h.cFormed,h.xFormed,h.wFormed,h.biasFormed⟩
  · rintro ⟨a,b,c,d,e,f,g,h⟩;exact ⟨a,b,c,d,e,f,g,h⟩

theorem checkLaunch_iff (a b x w : TensorInfo) (bias : Option TensorInfo) :
    checkLaunch a b x w bias=.ok () ↔ LaunchContract a b x w bias := by
  rw [launch_iff]
  simp only [checkLaunch]
  simp [Except.bind,Except.pure,bind,pure,throw,ite_eq_iff,and_assoc]
  omega

theorem contract_iff (a b x w : TensorInfo) (bias : Option TensorInfo) : InputContract a b x w bias ↔
    (a.metadata.shape.length=2 ∧ b.metadata.shape.length=2 ∧ x.metadata.shape.length=1 ∧ w.metadata.shape.length=1) ∧
    (a.metadata.layout=.strided ∧ b.metadata.layout=.strided ∧ x.metadata.layout=.strided ∧ w.metadata.layout=.strided) ∧
    (a.metadata.dtype=.other "int8" ∧ b.metadata.dtype=.other "int8" ∧ x.metadata.dtype=.floating .fp32 ∧ w.metadata.dtype=.floating .fp32) ∧
    (a.metadata.device.kind="cuda" ∧ b.metadata.device=a.metadata.device ∧ x.metadata.device=a.metadata.device ∧ w.metadata.device=a.metadata.device) ∧
    (a.metadata.isContiguous=Bool.true ∧ b.metadata.isContiguous=Bool.true ∧ x.metadata.isContiguous=Bool.true ∧ w.metadata.isContiguous=Bool.true) ∧
    (a.metadata.isNeg=Bool.false ∧ b.metadata.isNeg=Bool.false ∧ x.metadata.isNeg=Bool.false ∧ w.metadata.isNeg=Bool.false) ∧
    (rows b=cols a ∧ x.metadata.shape=[rows a] ∧ w.metadata.shape=[cols b]) ∧
    (∀ t,bias=some t → BiasContract a b t) ∧
    (x.storage≠a.storage ∧ x.storage≠b.storage ∧ w.storage≠a.storage ∧ w.storage≠b.storage) ∧
    (Aligned x ∧ Aligned w) ∧ (rows a=0 ∨ cols b=0 ∨ LaunchContract a b x w bias) := by
  constructor
  · intro h;exact ⟨h.rank,h.layout,h.dtype,h.device,h.contiguous,h.noNeg,h.shape,h.biasContract,h.inputAlias,h.alignment,h.launch⟩
  · rintro ⟨a,b,c,d,e,f,g,h,i,j,k⟩;exact ⟨a,b,c,d,e,f,g,h,i,j,k⟩

theorem checkTensors_iff (a b x w : TensorInfo) (bias : Option TensorInfo) :
    checkTensors a b x w bias=.ok () ↔ InputContract a b x w bias := by
  rw [contract_iff]
  cases bias with
  | none=>
    simp [checkTensors,checkLaunch_iff,Aligned,VeriTile.Torch.TensorMetadata.isCuda,
      Except.bind,Except.pure,bind,pure,ite_eq_iff,and_assoc]
    tauto
  | some t=>
    cases hb : checkBias a b t with
    | error e=>
      have bad : ¬BiasContract a b t := by
        intro h
        have := (checkBias_iff a b t).mpr h
        rw [hb] at this
        cases this
      simp [checkTensors,hb,bad,Except.bind,Except.pure,bind,pure,ite_eq_iff]
    | ok v=>
      have good : BiasContract a b t := (checkBias_iff a b t).mp (by cases v;exact hb)
      simp [checkTensors,hb,good,checkLaunch_iff,Aligned,VeriTile.Torch.TensorMetadata.isCuda,
        Except.bind,Except.pure,bind,pure,ite_eq_iff,and_assoc]
      tauto

theorem checkInputs_iff (a b x w : Argument) (bias : BiasArgument) :
    checkInputs a b x w bias=.ok () ↔ AdmittedArguments a b x w bias := by
  cases a <;> cases b <;> cases x <;> cases w <;> cases bias <;> simp [checkInputs,AdmittedArguments,checkTensors_iff]

def shape (a b : TensorInfo) : Shape := ⟨rows a,cols b,cols a⟩
def canonical (r c : Nat) : Dense2 r c := ⟨c,1,fun _ _=>Or.inr rfl,fun _ _=>Or.inr rfl⟩
def buffers (a b x w : TensorInfo) (bias : Option TensorInfo) (out : Nat) : Buffers :=
  ⟨a.storage,b.storage,x.storage,w.storage,(bias.map TensorInfo.storage).getD x.storage,out,
    a.offset,b.offset,x.offset,w.offset,biasOffset bias,0⟩

theorem raw_bounds : ∀ raw∈Source.configurations,
    16≤raw.BM ∧ raw.BM≤256 ∧ 32≤raw.BN ∧ raw.BN≤256 ∧ raw.BK*raw.S≤2048 ∧ raw.S≤16 := by decide +kernel

theorem config_bounds (c : Config) (member : Source.Admitted c) :
    16≤c.BM ∧ c.BM≤256 ∧ 32≤c.BN ∧ c.BN≤256 ∧ c.BK*c.S≤2048 ∧ c.S≤16 := by
  obtain ⟨raw,mem,eqM,eqN,eqK,eqS⟩:=member
  simpa [eqM,eqN,eqK,eqS] using raw_bounds raw mem

theorem ceil_mono (n d e : Nat) (positive : 0<d) (le : d≤e) :
    (n+e-1)/e ≤ (n+d-1)/d := by
  have pe : 0<e := by omega
  have top : n ≤ ((n+d-1)/d)*d := by
    have := Nat.mod_lt (n+d-1) positive
    have eq := Nat.mod_add_div (n+d-1) d
    rw [Nat.mul_comm d] at eq
    omega
  have upper := top.trans (Nat.mul_le_mul_left ((n+d-1)/d) le)
  apply Nat.le_of_lt_succ
  apply (Nat.div_lt_iff_lt_mul pe).mpr
  rw [Nat.succ_mul]
  omega

theorem native_admitted (a b x w : TensorInfo) (bias : Option TensorInfo) (out : Nat)
    (h : InputContract a b x w bias) (hm : 0<rows a) (hn : 0<cols b)
    (c : Config) (member : Source.Admitted c) :
    IndexABI (shape a b) c (buffers a b x w bias out)
      (canonical (rows a) (cols a)) (canonical (cols a) (cols b)) (canonical (rows a) (cols b)) ∧
    GridABI (shape a b) c := by
  have hc : LaunchContract a b x w bias := by rcases h.launch with z|z|hc <;> omega
  obtain ⟨loM,hiM,loN,hiN,hiKS,hiS⟩:=config_bounds c member
  have pm : padM (shape a b) c ≤ rows a+255 := by
    have := Nat.div_mul_le_self (rows a+c.BM-1) c.BM
    change (rows a+c.BM-1)/c.BM*c.BM≤rows a+255
    omega
  have pn : padN (shape a b) c ≤ cols b+255 := by
    have := Nat.div_mul_le_self (cols b+c.BN-1) c.BN
    change (cols b+c.BN-1)/c.BN*c.BN≤cols b+255
    omega
  have fk : formedK (shape a b) c ≤ cols a+4095 := by
    have div:=Nat.div_mul_le_self (cols a+c.BK*c.S-1) (c.BK*c.S)
    change (((cols a+c.BK*c.S-1)/(c.BK*c.S))+1)*(c.BK*c.S)≤cols a+4095
    rw [Nat.add_mul,Nat.one_mul]
    omega
  have dims:=hc.dimensions
  have aa:=hc.aFormed
  have bb:=hc.bFormed
  have cc:=hc.cFormed
  have xx:=hc.xFormed
  have ww:=hc.wFormed
  have zz:=hc.biasFormed
  have pmb:=Nat.mul_le_mul_right (cols b) pm
  have fkb:=Nat.mul_le_mul_right (cols b) fk
  have cm:=ceil_mono (rows a) 16 c.BM (by decide) loM
  have cn:=ceil_mono (cols b) 32 c.BN (by decide) loN
  have cgrid:=hc.grid
  have product:=(Nat.mul_le_mul cm cn).trans cgrid
  have cdiv : (cols b+c.BN-1)/c.BN ≤ cols b+255 := by
    have := Nat.div_le_self (cols b+c.BN-1) c.BN
    omega
  constructor
  · refine ⟨?_,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
    · change rows a+c.BM<2^63 ∧ cols b+c.BN<2^63 ∧ cols a+c.BK*c.S<2^63;omega
    · change cols a<2^63 ∧ 1<2^63 ∧ cols b<2^63 ∧ 1<2^63 ∧ cols b<2^63 ∧ 1<2^63;omega
    · exact ⟨by omega,by omega,by omega⟩
    · change 8*((cols b+c.BN-1)/c.BN)<2^63;omega
    · change a.offset+rows a*cols a+formedK (shape a b) c*1<2^63;omega
    · change b.offset+formedK (shape a b) c*cols b+cols b*1<2^63;omega
    · change 2*(0+padM (shape a b) c*cols b+padN (shape a b) c*1)<2^63;omega
    · exact xx
    · exact ww
    · change 4*(biasOffset bias+padN (shape a b) c)<2^63;omega
  · refine ⟨?_,by omega⟩
    simpa only [Nat.reduceAdd,Nat.reduceSub,shape] using product

#axiomsClean checkInputs_iff
#axiomsClean native_admitted
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.Int8DequantMatmul.Direct
