import bench.repairs.tritonbench_g.destindex_copy_kv1.Proofs.Kernel
import bench.repairs.tritonbench_g.destindex_copy_kv1.Proofs.TensorBinding

namespace VeriTile.Bench.Repairs.DestindexCopyKv1
noncomputable section
set_option maxHeartbeats 3000000

/-- All native source facts are conclusions of the checked contract and ordinary
initial Tensor binding. No output, coverage, or numerical equality is a premise. -/
structure SourceConformance (m : Inputs) (w : Nat → Word) (before : Memory) : Prop where
  grid : (config m).count<2^31
  masks : ∀ a b, a<block (config m).heads → b<block (config m).dims →
    Native.mask (config m) a b=decide (a<(config m).heads ∧ b<(config m).dims)
  metadata : ∀ p, p<(config m).count →
    (before (addr (base m .destinations) p)).toInt=(destination w p:Int)
  metadataPointer : ∀ p, p<(config m).count →
    addr (base m .destinations) p+4≤base m .destinations+4*(m .destinations).numel ∧
    addr (base m .destinations) p+4≤2^64
  indexes : ∀ p a b, p<(config m).count → a<(config m).heads → b<(config m).dims →
    (Native.index (config m) p a b).toInt=(flat (config m) p a b:Int) ∧
    (Native.index (config m) (destination w p) a b).toInt=(flat (config m) (destination w p) a b:Int)
  sourcePointer : ∀ p a b, p<(config m).count → a<(config m).heads → b<(config m).dims →
    addr (base m .source) (flat (config m) p a b)+4≤base m .source+4*(m .source).numel ∧
    addr (base m .source) (flat (config m) p a b)+4≤2^64
  outputPointer : ∀ p a b, p<(config m).count → a<(config m).heads → b<(config m).dims →
    addr (base m .output) (flat (config m) (destination w p) a b)+4≤base m .output+4*(m .output).numel ∧
    addr (base m .output) (flat (config m) (destination w p) a b)+4≤2^64
  readsSeparate : ∀ x r y, x<(m .output).numel → r≠.output → y<(m r).numel →
    addr (base m .output) x+4≤addr (base m r) y ∨ addr (base m r) y+4≤addr (base m .output) x
  writesUnique : ∀ p q a b a' b', p<(config m).count → q<(config m).count →
    a<(config m).heads → a'<(config m).heads → b<(config m).dims → b'<(config m).dims →
    flat (config m) (destination w p) a b=flat (config m) (destination w q) a' b' → p=q ∧ a=a' ∧ b=b'

theorem source_conformance (m : Inputs) (w : Nat → Word) (h : InputContract m w)
    (before : Memory) (binding : TensorBinding m w before) : SourceConformance m w before := by
  refine ⟨h.extent .destinations,Native.mask_exact m w h,Native.metadata m w h before binding,?_,
    Native.active_index m w h,?_,?_,?_,?_⟩
  · intro p hp
    exact (Native.pointer m w h before binding .destinations p hp).2
  · intro p a b hp ha hb
    exact (Native.pointer m w h before binding .source _ (source_bound m w h p a b hp ha hb)).2
  · intro p a b hp ha hb
    exact (Native.pointer m w h before binding .output _ (output_bound m w h p a b hp ha hb)).2
  · intro x r y hx hr hy
    have separated := h.separate r hr
    unfold addr
    rcases separated with left|right <;> omega
  · intro p q a b a' b' hp hq ha ha' hb hb' same
    have parts := Grid.flat_injective (config m) _ _ _ _ _ _ ha ha' hb hb' same
    exact ⟨h.unique p q hp hq parts.1,parts.2⟩

/-- Each original integer term is nonnegative and cannot wrap before it is
added to the pointer. Thus the original left-associated pointer chain and the
single offset in typed source semantics agree on every active lane. -/
theorem pointer_terms (c : Config) (p a b : Nat) (bound : flat c p a b<2^31) :
    (BitVec.ofNat 32 p * BitVec.ofNat 32 (c.heads*c.dims)).toInt=(p*(c.heads*c.dims):Int) ∧
    (BitVec.ofNat 32 c.dims * BitVec.ofNat 32 a).toInt=(c.dims*a:Int) ∧
    (BitVec.ofNat 32 b).toInt=(b:Int) := by
  have h1 : p*(c.heads*c.dims)<2^31 := by unfold flat at bound; omega
  have h2 : c.dims*a<2^31 := by rw [Nat.mul_comm]; unfold flat at bound; omega
  have h3 : b<2^31 := by unfold flat at bound; omega
  rw [BitVec.ofNat_mul_ofNat,BitVec.ofNat_mul_ofNat,Native.signed32 _ h1,Native.signed32 _ h2,
    Native.signed32 _ h3]
  simp only [Nat.cast_mul,and_self]

#axiomsClean source_conformance
#axiomsClean pointer_terms
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DestindexCopyKv1
