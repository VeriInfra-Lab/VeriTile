import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.Physical
namespace VeriTile.Bench.Repairs.AttnFwdCausal.Contract
open VeriTile Triton Source Whole
noncomputable section
structure Observations (p : Parameters) where
  ordinaryCUDAFP32 : Bool
  canonicalResolved : Bool
  pointer : Buffer → Nat
  sourceEnvelope : Nat
  /-- Ordinary binding of the observed Python stride arithmetic. Empty input
  strides remain arbitrary; live canonical strides determine the model envelope. -/
  liveEnvelope : 0<p.batches → 0<p.heads → 0<p.length → sourceEnvelope=envelope p

def InputContract (p : Parameters) (o : Observations p) : Prop :=
  o.ordinaryCUDAFP32=Bool.true ∧ o.canonicalResolved=Bool.true ∧
  o.sourceEnvelope<2147483648 ∧ p.batches*p.heads≤65535 ∧
  (0<p.batches → 0<p.heads → 0<p.length → ∀ r,
    o.pointer r%width r=0 ∧ o.pointer r+4*o.sourceEnvelope<2^64)

def check (p : Parameters) (o : Observations p) : Bool := by
  classical
  exact decide (InputContract p o)

theorem check_iff (p : Parameters) (o : Observations p) :
    check p o=Bool.true ↔ InputContract p o := by
  classical
  simp [check]

def guard (p : Parameters) (o : Observations p) (h : InputContract p o) : Guard p where
  signed := by
    intro hb hh hn
    rw [←o.liveEnvelope hb hh hn]
    exact h.2.2.1
  launchY := h.2.2.2.1
  inputStrides := by
    intro hb hh hn s hs
    simp only [List.mem_cons,List.not_mem_nil,or_false] at hs
    rcases hs with rfl|rfl|rfl <;> simp [Parameters.q,Parameters.k,Parameters.v]

/-- Current finite Tensor storage and fresh allocator metadata. Separate
output bytes come from the ordinary successful fresh allocation. -/
structure StorageAPI (p : Parameters) (storage : TensorStorage p) (o : Observations p) where
  arena : Nat
  arena64 : arena≤2^64
  bounded : ∀ r,o.pointer r+width r*extent storage r≤arena
  fresh : ∀ r,r≠.output → ∀ a<width r*extent storage r,∀ b<2*storage.sizeOut,
    o.pointer r+a≠o.pointer .output+b

def placement (p : Parameters) (storage : TensorStorage p) (o : Observations p)
    (api : StorageAPI p storage o) (h : InputContract p o) : Placement p storage where
  arena := api.arena
  base := o.pointer
  arena64 := api.arena64
  bounded := api.bounded
  formed64 := by
    intro hb hh hn r
    rw [←o.liveEnvelope hb hh hn]
    exact (h.2.2.2.2 hb hh hn r).2
  apart := api.fresh

theorem aligned (p : Parameters) (o : Observations p) (h : InputContract p o)
    (hb : 0<p.batches) (hh : 0<p.heads) (hn : 0<p.length) (r : Buffer) (off : Nat) :
    (o.pointer r+width r*off)%width r=0 := by
  have hp:=(h.2.2.2.2 hb hh hn r).1
  cases r <;> dsimp [width] at * <;> omega

#specNonCircular InputContract avoiding [FP.Structural.exec,Whole.invoke]
#axiomsClean check_iff
#axiomsClean aligned
end
end VeriTile.Bench.Repairs.AttnFwdCausal.Contract
