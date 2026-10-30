import bench.tritonbench_g.attention_fwd_triton2.Proofs.Machine

namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.Checked
open Whole001 Whole001.Source Whole001.Whole Whole001.Machine

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
  (∀ t∈[a.query,a.key,a.value,a.queryScale,a.keyScale], t.negative=Bool.false ∧ t.conjugate=Bool.false))

def admittedABI (p : Parameters) (a : Arguments p) (h : inputChecks p a=Bool.true) : InputABI p := by
  simp only [inputChecks,decide_eq_true_eq] at h
  exact ⟨a.query,a.key,a.value,a.queryScale,a.keyScale,h.2.2.2.1,a.strides,
    fun hb hh hn=>h.2.2.2.2.1 ⟨hb,hh,hn⟩,a.regions,h.2.2.2.2.2.1,
    h.2.2.2.2.2.2.1,h.2.2.2.2.2.2.2⟩

def denseChecks (p : Parameters) : Bool :=
  decide (Live p → p.q=dense p ∧ p.k=dense p ∧ p.v=dense p)

/-- Canonical output stride is the ordinary fresh empty_like allocation fact. -/
def launchChecks (p : Parameters) : Bool := decide (envelope p<2^31 ∧ p.batches*p.heads≤65535)
def pointerChecks (p : Parameters) (b : Bases) : Bool :=
  (baseWidths b).all fun x=>decide (x.1%x.2=0 ∧ x.1+x.2*envelope p<2^64)
def setupChecks (p : Parameters) (b : Bases) : Bool :=
  if Live p then launchChecks p && pointerChecks p b else Bool.true

theorem admittedDomain (p : Parameters) (b : Bases)
    (denseOK : denseChecks p=Bool.true) (output : Live p → p.o=dense p)
    (setup : setupChecks p b=Bool.true) : Domain p b := by
  have hd : Live p → p.q=dense p ∧ p.k=dense p ∧ p.v=dense p := by
    simpa only [denseChecks,decide_eq_true_eq] using denseOK
  refine ⟨fun live=>⟨(hd live).1,(hd live).2.1,(hd live).2.2,output live⟩,?_,?_⟩
  · intro live
    have h:=setup
    simp only [setupChecks,live,↓reduceIte,Bool.and_eq_true,launchChecks,decide_eq_true_eq] at h
    refine ⟨h.1.1,h.1.2,?_⟩
    intro s hs
    have pos : 0<(dense p).batch ∧ 0<(dense p).head ∧ 0<(dense p).row ∧ 0<(dense p).dim := by
      simp only [dense]
      exact ⟨Nat.mul_pos (Nat.mul_pos live.2.1 live.2.2) (by decide),
        Nat.mul_pos live.2.2 (by decide),by decide,by decide⟩
    simp only [List.mem_cons,List.not_mem_nil,or_false] at hs
    rcases hs with rfl|rfl|rfl
    · simpa only [(hd live).1] using pos
    · simpa only [(hd live).2.1] using pos
    · simpa only [(hd live).2.2] using pos
  · intro live
    have h : launchChecks p=Bool.true ∧ pointerChecks p b=Bool.true := by
      simpa only [setupChecks,live,↓reduceIte,Bool.and_eq_true] using setup
    simpa only [pointerChecks,List.all_eq_true,decide_eq_true_eq] using h.2

#axiomsClean admittedABI
#axiomsClean admittedDomain
end VeriTile.Bench.Repairs.AttentionFwdTriton2.Checked
