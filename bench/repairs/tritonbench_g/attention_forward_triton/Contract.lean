import bench.repairs.tritonbench_g.attention_forward_triton.Proofs.Binding

namespace VeriTile.Bench.Repairs.AttentionForwardTriton.Checked
open Source Whole

def Live (p : Parameters) : Prop := 0<p.batches ∧ 0<p.heads ∧ 0<p.length

instance (p : Parameters) : Decidable (Live p) :=
  inferInstanceAs (Decidable (0<p.batches ∧ 0<p.heads ∧ 0<p.length))

/-- Actual raw descriptors. Region and stride extraction are ordinary metadata
bindings; supported shape/dtype/device and resolved-view facts are checked. -/
structure Arguments (p : Parameters) where
  (query key value queryScale keyScale : Descriptor)
  (cuda strided tensors : Bool)
  strides : query.strides=[p.q.batch,p.q.head,p.q.row,p.q.dim] ∧
    key.strides=[p.k.batch,p.k.head,p.k.row,p.k.dim] ∧
    value.strides=[p.v.batch,p.v.head,p.v.row,p.v.dim]
  regions : query.region=p.Q ∧ key.region=p.K ∧ value.region=p.V ∧
    queryScale.region=p.QS ∧ keyScale.region=p.KS

def inputChecks (p : Parameters) (a : Arguments p) : Bool := decide (
  a.cuda=Bool.true ∧ a.strided=Bool.true ∧ a.tensors=Bool.true ∧
  (a.query.shape=[p.batches,p.heads,p.length,96] ∧ a.key.shape=a.query.shape ∧
    a.value.shape=a.query.shape ∧ a.queryScale.shape=[p.batches,p.heads,(p.length+127)/128] ∧
    a.keyScale.shape=[p.batches,p.heads,(p.length+63)/64]) ∧
  (Live p → a.queryScale.strides=[p.heads*((p.length+127)/128),(p.length+127)/128,1] ∧
    a.keyScale.strides=[p.heads*((p.length+63)/64),(p.length+63)/64,1]) ∧
  (∀ t∈[a.query,a.key,a.value,a.queryScale,a.keyScale], t.dtype=.fp32) ∧
  (∀ t∈[a.query,a.key,a.value,a.queryScale,a.keyScale], t.device=a.query.device) ∧
  (∀ t∈[a.query,a.key,a.value,a.queryScale,a.keyScale], t.negative=Bool.false ∧ t.conjugate=Bool.false) ∧
  (Live p → [p.Q,p.K,p.V,p.QS,p.KS].Nodup))

def admittedABI (p : Parameters) (a : Arguments p) (h : inputChecks p a=Bool.true) : InputABI p := by
  simp only [inputChecks,decide_eq_true_eq] at h
  exact ⟨a.query,a.key,a.value,a.queryScale,a.keyScale,h.2.2.2.1,a.strides,
    fun hb hh hn=>h.2.2.2.2.1 ⟨hb,hh,hn⟩,a.regions,h.2.2.2.2.2.1,
    h.2.2.2.2.2.2.1,h.2.2.2.2.2.2.2.1,
    fun hb hh hn=>h.2.2.2.2.2.2.2.2 ⟨hb,hh,hn⟩⟩

def strideChecks (p : Parameters) : Bool := decide (Live p →
  ∀ s∈[p.q,p.k,p.v],0<s.batch ∧ 0<s.head ∧ 0<s.row ∧ 0<s.dim)
def launchChecks (p : Parameters) : Bool := decide (envelope p<2^31 ∧ p.batches*p.heads≤65535)

theorem admittedGuard (p : Parameters) (strides : strideChecks p=Bool.true)
    (launch : launchChecks p=Bool.true) : Guard p := by
  simp only [strideChecks,decide_eq_true_eq] at strides
  simp only [launchChecks,decide_eq_true_eq] at launch
  exact ⟨launch.1,launch.2,fun hb hh hn=>strides ⟨hb,hh,hn⟩⟩

/-- Ordinary live allocation geometry; no padded-address or alignment premise. -/
structure OrdinaryPlacement (p : Parameters) (storage : TensorStorage p) where
  arena : Nat
  base : Buffer → Nat
  arena64 : arena≤2^64
  bounded : ∀ r,base r+width r*extent storage r≤arena
  apart : ∀ r,r≠.output → ∀ a<width r*extent storage r,∀ b<2*storage.sizeOut,
    base r+a≠base .output+b

def buffers : List Buffer := [.query,.key,.value,.queryScale,.keyScale,.output]
def pointerChecks (p : Parameters) (base : Buffer → Nat) : Bool :=
  buffers.all fun r=>decide (base r%width r=0 ∧ base r+4*envelope p<2^64)

theorem all_buffers (r : Buffer) : r∈buffers := by cases r <;> simp [buffers]

def admittedPlacement (p : Parameters) (storage : TensorStorage p)
    (ordinary : OrdinaryPlacement p storage) (h : pointerChecks p ordinary.base=Bool.true) : Placement p storage where
  arena:=ordinary.arena
  base:=ordinary.base
  arena64:=ordinary.arena64
  bounded:=ordinary.bounded
  apart:=ordinary.apart
  formed64:=by
    simp only [pointerChecks,List.all_eq_true,decide_eq_true_eq] at h
    exact fun r=>(h r (all_buffers r)).2

theorem aligned (p : Parameters) (base : Buffer → Nat) (h : pointerChecks p base=Bool.true) :
    ∀ r,base r%width r=0 := by
  simp only [pointerChecks,List.all_eq_true,decide_eq_true_eq] at h
  exact fun r=>(h r (all_buffers r)).1

def profileCheck (p : Parameters) (base : Buffer → Nat) (profile : Profile) : Bool := decide (
  p.batches=0 ∨ p.heads=0 ∨ p.length=0 ∨
    (buffers.map (fun r=>if base r%16=0 then 16 else 0)=pointerClasses profile ∧
      (dynamicValues p).map argumentClass=integerClasses profile))

theorem admittedCompiler (p : Parameters) (storage : TensorStorage p)
    (ordinary : OrdinaryPlacement p storage) (pointers : pointerChecks p ordinary.base=Bool.true)
    (profile : Profile) (h : profileCheck p ordinary.base profile=Bool.true) :
    CompilerBinding p storage (admittedPlacement p storage ordinary pointers) profile := by
  simpa only [profileCheck,decide_eq_true_eq,CompilerBinding,admittedPlacement,buffers] using h

#axiomsClean admittedABI
#axiomsClean admittedGuard
#axiomsClean admittedPlacement
#axiomsClean admittedCompiler
end VeriTile.Bench.Repairs.AttentionForwardTriton.Checked
