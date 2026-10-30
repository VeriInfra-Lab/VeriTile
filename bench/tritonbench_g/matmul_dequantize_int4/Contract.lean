import bench.tritonbench_g.matmul_dequantize_int4.Whole
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.Checked
open VeriTile.Triton

/-- Direct observations of the actual Tensor and its untyped allocation. The
view origin may have a nonzero storage offset. No payload is materialized. -/
structure Observation where
  object : Nat
  metadata : VeriTile.Torch.TensorMetadata
  storageObject : Nat
  storageBase : Nat
  storageBytes : Nat
  storageOffset : Nat
  isConj : Bool
  deriving DecidableEq, Repr

def viewChecks (v : View) : Prop := 0<v.sr ∧ 0<v.sc ∧
  (v.cols*v.sc≤v.sr ∨ v.rows*v.sr≤v.sc) ∧
  (v.rows=0 ∨ v.cols=0 ∨ (v.rows-1)*v.sr+(v.cols-1)*v.sc<v.capacity)
instance (v : View) : Decidable (viewChecks v) := inferInstanceAs (Decidable (_ ∧ _))

def tensorChecks (t : Observation) (v : View) (dtype : VeriTile.Torch.DType) (width : Nat) : Prop :=
  t.metadata.dtype=dtype ∧ t.metadata.layout=.strided ∧ t.metadata.shape=[v.rows,v.cols] ∧
  t.metadata.isNeg=Bool.false ∧ t.isConj=Bool.false ∧
  t.metadata.dataPtr.toNat=t.storageBase+width*t.storageOffset ∧
  t.metadata.dataPtr.toNat%width=0 ∧ t.storageBase+t.storageBytes<2^64 ∧
  t.storageBytes%width=0 ∧ width*(t.storageOffset+v.capacity)≤t.storageBytes ∧
  t.metadata.dataPtr.toNat+width*2^31<2^64
instance (t : Observation) (v : View) (dtype : VeriTile.Torch.DType) (width : Nat) :
    Decidable (tensorChecks t v dtype width) := inferInstanceAs (Decidable (_ ∧ _))

def apart (a b : Observation) :=
  a.storageBase+a.storageBytes≤b.storageBase ∨ b.storageBase+b.storageBytes≤a.storageBase
instance (a b : Observation) : Decidable (apart a b) := inferInstanceAs (Decidable (_ ∨ _))

structure Metadata where
  call : Call
  (a q s z output : Observation)

def observations (p : Metadata) := [p.a,p.q,p.s,p.z,p.output]
def metadataChecks (p : Metadata) : Bool := decide (
  tensorChecks p.a p.call.A (.other "float16") 2 ∧
  tensorChecks p.q p.call.Q (.other "int32") 4 ∧
  tensorChecks p.s p.call.S (.other "float16") 2 ∧
  tensorChecks p.z p.call.Z (.other "int32") 4 ∧
  tensorChecks p.output p.call.C (.other "float16") 2 ∧
  p.a.metadata.isContiguous=Bool.true ∧
  (∀ t∈observations p,t.metadata.device=p.a.metadata.device ∧ t.metadata.device.kind="cuda") ∧
  (observations p).Pairwise apart)

def configurationChecks (c : Call) : Bool := decide (
  0<c.M ∧ 0<c.N ∧ 0<c.K ∧ 0<c.group ∧ 256∣c.N ∧ 128∣c.K ∧ 128∣c.group ∧ c.group∣c.K ∧
  c.A.rows=c.M ∧ c.A.cols=c.K ∧ c.Q.rows=c.K/8 ∧ c.Q.cols=c.N ∧
  c.S.rows=c.K/c.group ∧ c.S.cols=c.N ∧ c.Z.rows=c.K/c.group ∧ c.Z.cols=c.N/8 ∧
  c.C.rows=c.M ∧ c.C.cols=c.N ∧ c.A.sr=c.K ∧ c.A.sc=1 ∧
  viewChecks c.A ∧ viewChecks c.Q ∧ viewChecks c.S ∧ viewChecks c.Z ∧ viewChecks c.C ∧
  (c.supplied=Bool.false → c.C.sr=c.N ∧ c.C.sc=1))

def indexChecks (c : Call) : Bool := decide (
  c.M+128<2^31 ∧ 8*c.N+256<2^31 ∧ c.K+128<2^31 ∧ c.group<2^31 ∧
  (∀ v∈[c.A,c.Q,c.S,c.Z,c.C],v.sr<2^31 ∧ v.sc<2^31) ∧
  (c.M+128)*c.A.sr+(c.K+128)*c.A.sc<2^31 ∧
  (c.K/8+16)*c.Q.sr+c.N*c.Q.sc<2^31 ∧
  (c.K/c.group)*c.S.sr+c.N*c.S.sc<2^31 ∧
  (c.K/c.group)*c.Z.sr+(c.N/8)*c.Z.sc<2^31 ∧
  (c.M+128)*c.C.sr+c.N*c.C.sc<2^31)

inductive Error where | dynamic | metadata | shape | indices | nonfinite deriving DecidableEq, Repr

def checkInputs (p : Metadata) : Except Error Unit :=
  if !metadataChecks p then .error .metadata
  else if !configurationChecks p.call then .error .shape
  else if !indexChecks p.call then .error .indices else .ok ()
def InputContract (p : Metadata) := metadataChecks p=Bool.true ∧
  configurationChecks p.call=Bool.true ∧ indexChecks p.call=Bool.true

theorem checkInputs_iff (p : Metadata) : checkInputs p=.ok () ↔ InputContract p := by
  simp [checkInputs,InputContract,ite_eq_iff]

theorem view_valid (v : View) (h : viewChecks v) : v.Valid := ⟨h.1,h.2.1,h.2.2.1,h.2.2.2⟩
theorem admitted (p : Metadata) (h : InputContract p) : Public.Admitted p.call := by
  have hc := h.2.1
  have hi := h.2.2
  simp only [configurationChecks,decide_eq_true_eq] at hc
  simp only [indexChecks,decide_eq_true_eq] at hi
  obtain ⟨m,n,k,g,n256,k128,g128,groups,am,ak,qk,qn,sg,sn,zg,zn,cm,cn,asr,asc,av,qv,sv,zv,cv,fresh⟩ := hc
  obtain ⟨mi,ni,ki,gi,strides,af,qf,sf,zf,cf⟩ := hi
  exact ⟨⟨m,n,k,g,n256,k128,g128,groups,⟨am,ak⟩,⟨qk,qn⟩,⟨sg,sn⟩,⟨zg,zn⟩,⟨cm,cn⟩,
    ⟨asr,asc⟩,view_valid _ av,view_valid _ qv,view_valid _ sv,view_valid _ zv,view_valid _ cv,fresh⟩,
    ⟨⟨mi,ni,ki,gi⟩,strides,af,qf,sf,zf,cf⟩⟩

/-- Python type/rank/integer classification precedes all metadata queries. -/
inductive Arguments where | unsupported | tensors (metadata : Metadata)
def checkArguments : Arguments → Except Error Unit
  | .unsupported => .error .dynamic | .tensors p => checkInputs p
def AdmittedArguments : Arguments → Prop
  | .unsupported => False | .tensors p => InputContract p
theorem checkArguments_iff (p : Arguments) : checkArguments p=.ok () ↔ AdmittedArguments p := by
  cases p <;> simp [checkArguments,AdmittedArguments,checkInputs_iff]

#axiomsClean admitted
#axiomsClean checkInputs_iff
#axiomsClean checkArguments_iff
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.Checked
