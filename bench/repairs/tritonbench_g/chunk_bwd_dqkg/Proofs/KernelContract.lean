import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Domain
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.KernelStorage
import VeriTile.Triton.Float.Structural

/-! Kernel-entry admission: all ten arguments already exist. There is no
allocator, host wrapper, autotuner lifecycle or launch-success condition. -/
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.KernelContract
open VeriTile.Triton VeriTile.Torch Whole
inductive Role where
  | q | k | v | h | g | dout | dh | dq | dk | dg
  deriving DecidableEq
def roles : List Role := [.q,.k,.v,.h,.g,.dout,.dh,.dq,.dk,.dg]
theorem listed (r : Role) : r∈roles := by cases r <;> simp [roles]
def region : Role → RegionName
  | .q=>"q" | .k=>"k" | .v=>"v" | .h=>"h" | .g=>"g"
  | .dout=>"do" | .dh=>"dh" | .dq=>"dq" | .dk=>"dk" | .dg=>"dg"
def dimensions (c : Config) : Role → List Nat
  | .q | .k | .dq | .dk=>[c.B,c.H,c.T,c.K]
  | .v | .dout=>[c.B,c.H,c.T,c.V]
  | .h | .dh=>[c.B,c.H,c.NT*c.K,c.V]
  | .g=>[c.B,c.H,c.T]
  | .dg=>[c.NK,c.B,c.H,c.T]
def denseStrides : List Nat → List Nat
  | []=>[]
  | _::rest=>(rest.map (fun n=>max n 1)).prod::denseStrides rest
def size (c : Config) (r : Role) : Nat := (dimensions c r).prod
structure Observation where
  metadata : TensorMetadata
  strides : List Nat
  isConj : Bool
def nonempty (c : Config) : Prop := 0<c.B ∧ 0<c.H ∧ 0<c.T
instance (c : Config) : Decidable (nonempty c) := inferInstanceAs (Decidable (0<c.B ∧ 0<c.H ∧ 0<c.T))
def domain (c : Config) : Prop :=
  0<c.K ∧ 0<c.V ∧ (nonempty c → envelope c<2^31 ∧ c.NK<2^31 ∧ c.NT≤65535 ∧ c.heads≤65535)
instance (c : Config) : Decidable (domain c) := inferInstanceAs (Decidable
  (0<c.K ∧ 0<c.V ∧ (nonempty c → envelope c<2^31 ∧ c.NK<2^31 ∧ c.NT≤65535 ∧ c.heads≤65535)))
def tensor (c : Config) (device : Device) (r : Role) (o : Observation) : Prop :=
  o.metadata.layout=.strided ∧ o.metadata.dtype=.floating .fp32 ∧ o.metadata.device.kind="cuda" ∧
  o.metadata.device=device ∧ o.metadata.isContiguous=Bool.true ∧ o.metadata.isNeg=Bool.false ∧
  o.isConj=Bool.false ∧ o.metadata.shape=dimensions c r ∧ o.strides=denseStrides (dimensions c r) ∧
  (0<o.metadata.numel → o.metadata.dataPtr.toNat%4=0) ∧
  (nonempty c → o.metadata.dataPtr.toNat+4*envelope c<2^64)
instance (c : Config) (d : Device) (r : Role) (o : Observation) : Decidable (tensor c d r o) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _))
def separate (c : Config) (o : Role → Observation) (r s : Role) : Prop :=
  nonempty c → r≠s →
    (o r).metadata.dataPtr.toNat+4*size c r≤(o s).metadata.dataPtr.toNat ∨
    (o s).metadata.dataPtr.toNat+4*size c s≤(o r).metadata.dataPtr.toNat
instance (c : Config) (o : Role → Observation) (r s : Role) : Decidable (separate c o r s) :=
  inferInstanceAs (Decidable (nonempty c → r≠s → _ ∨ _))
structure Admitted (c : Config) (o : Role → Observation) : Prop where
  shape : domain c
  tensors : ∀ r,tensor c (o .q).metadata.device r (o r)
  separated : ∀ r s,separate c o r s
def check (c : Config) (o : Role → Observation) : Bool :=
  decide (domain c) && roles.all (fun r=>decide (tensor c (o .q).metadata.device r (o r))) &&
    roles.all (fun r=>roles.all (fun s=>decide (separate c o r s)))
theorem check_iff (c : Config) (o : Role → Observation) : check c o=Bool.true ↔ Admitted c o := by
  simp only [check,Bool.and_eq_true,decide_eq_true_eq,List.all_eq_true,listed,true_implies]
  constructor
  · rintro ⟨⟨hd,ht⟩,hs⟩; exact ⟨hd,ht,hs⟩
  · intro h; exact ⟨⟨h.shape,h.tensors⟩,h.separated⟩
theorem tensor_checked (c : Config) (d : Device) (r : Role) (o : Observation) (h : tensor c d r o) :
    VeriTile.Bench.Repairs.FastLayernorm.Contract.TensorContract o.metadata Bool.true :=
  ⟨h.1,h.2.1,h.2.2.1,fun _=>h.2.2.2.2.1,h.2.2.2.2.2.1,h.2.2.2.2.2.2.2.2.2.1⟩
theorem guard (c : Config) (o : Role → Observation) (h : Admitted c o) (hn : nonempty c) : Guard c :=
  ⟨h.shape.1,h.shape.2.1,(h.shape.2.2 hn).1,(h.shape.2.2 hn).2.1,
    (h.shape.2.2 hn).2.2.1,(h.shape.2.2 hn).2.2.2⟩

/-- Conversion of a Python numeric scale to these FP32 bits is a local ABI
boundary; finite decoding, the Real projection, and -0 payload are explicit. -/
structure Scale where
  word : FP.Value .fp32
  rational : ℚ
  finite : word.toRat?=some rational
def Scale.real (s : Scale) : ℝ := s.rational

#axiomsClean check_iff
#axiomsClean tensor_checked
#axiomsClean guard
#specNonCircular check avoiding [Whole.runGrid,VeriTile.Triton.exec]
end VeriTile.Bench.Repairs.ChunkBwdDqkg.KernelContract
