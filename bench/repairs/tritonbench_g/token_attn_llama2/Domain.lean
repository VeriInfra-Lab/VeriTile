import bench.repairs.tritonbench_g.token_attn_llama2.Proofs.Storage
namespace VeriTile.Bench.Repairs.TokenAttnLlama2.Whole
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000

/-- Exact dense strides and storage extents in the selected ordinary FP32 ABI. -/
structure Dense (c : Call) : Prop where
  q : c.qBatch=c.heads*c.dim ∧ c.qHead=c.dim ∧ c.qDim=1
  k : c.kBatch=c.kvHeads*c.dim ∧ c.kHead=c.dim ∧ c.kDim=1
  loc : c.locBatch=c.maxLength ∧ c.locSeq=1
  out : c.outHead=c.outputTokens ∧ c.outSeq=1
  sizes : c.qSize=c.batches*c.heads*c.dim ∧ c.kSize=c.cacheRows*c.kvHeads*c.dim ∧
    c.locSize=c.batches*c.maxLength ∧ c.outSize=c.heads*c.outputTokens

/-- Metadata input facts. No output value, run result, or numerical fact appears. -/
structure Data (c : Call) (m : Metadata c) : Prop where
  lengths : ∀ b : Fin c.batches, 0≤m.lengths b ∧ m.lengths b≤(c.maxLength : Int)
  starts : ∀ b : Fin c.batches, 0≤m.starts b ∧ m.begin b.val+m.len b.val≤c.outputTokens
  pages : ∀ b : Fin c.batches, ∀ t, t<m.len b.val →
    0≤finiteRead m.locations (c.locBatch*b.val+c.locSeq*(c.maxLength-m.len b.val+t)) ∧ m.page b.val t<c.cacheRows
  segments : ∀ b<c.batches, ∀ b'<c.batches, ∀ t<m.len b, ∀ t'<m.len b',
    m.begin b+t=m.begin b'+t' → b=b' ∧ t=t'

def envelope (c : Call) : Nat :=
  c.batches+c.heads+c.kvHeads+c.dim+c.maxLength+32+c.cacheRows+c.outputTokens+
  c.qSize+c.kSize+c.locSize+c.outSize+c.batches*c.qBatch+c.heads*c.qHead+c.dim*c.qDim+
  c.cacheRows*c.kBatch+c.kvHeads*c.kHead+c.dim*c.kDim+
  c.locBatch*c.batches+c.locSeq*(c.maxLength+32)+c.heads*c.outHead+(c.outputTokens+32)*c.outSeq

structure Guard (c : Call) : Prop where
  dimension : c.dim=16 ∨ c.dim=32 ∨ c.dim=64 ∨ c.dim=128
  kvPositive : 0<c.kvHeads
  groupsPositive : 0<c.heads/c.kvHeads
  groupsExact : c.heads=c.kvHeads*(c.heads/c.kvHeads)
  signed : envelope c<2147483648

theorem row_bound (a n i j : Nat) (hi : i<a) (hj : j<n) : i*n+j<a*n := by
  have step := Nat.mul_le_mul_right n (show i+1≤a by omega)
  rw [Nat.add_mul,Nat.one_mul] at step
  omega

theorem pair_injective (n a b c d : Nat) (hb : b<n) (hd : d<n)
    (same : a*n+b=c*n+d) : a=c ∧ b=d := by
  have positive : 0<n := by omega
  have q := congrArg (fun k => k/n) same
  have rem := congrArg (fun k => k%n) same
  have da : (a*n+b)/n=a := by rw [Nat.add_comm,Nat.add_mul_div_right _ _ positive,Nat.div_eq_of_lt hb]; omega
  have dc : (c*n+d)/n=c := by rw [Nat.add_comm,Nat.add_mul_div_right _ _ positive,Nat.div_eq_of_lt hd]; omega
  exact ⟨by simpa [da,dc] using q,by simpa [Nat.add_mod,Nat.mod_eq_of_lt hb,Nat.mod_eq_of_lt hd] using rem⟩

/-- Bounds and write injectivity are derived from the selected layout. -/
theorem domain_valid (c : Call) (m : Metadata c) (dense : Dense c) (g : Guard c) (data : Data c m) : Valid c m := by
  refine ⟨g.dimension,g.kvPositive,g.groupsPositive,g.groupsExact,data.lengths,data.starts,data.pages,?_,?_,?_,?_,?_,data.segments,g.signed⟩
  · intro b hb h hh d hd
    have outer := row_bound c.batches c.heads b h hb hh
    have inner := row_bound (c.batches*c.heads) c.dim (b*c.heads+h) d outer hd
    simpa only [dense.q.1,dense.q.2.1,dense.q.2.2,dense.sizes.1,Nat.mul_one,Nat.add_mul,Nat.mul_assoc] using inner
  · intro r hr h hh d hd
    have outer := row_bound c.cacheRows c.kvHeads r h hr hh
    have inner := row_bound (c.cacheRows*c.kvHeads) c.dim (r*c.kvHeads+h) d outer hd
    simpa only [dense.k.1,dense.k.2.1,dense.k.2.2,dense.sizes.2.1,Nat.mul_one,Nat.add_mul,Nat.mul_assoc] using inner
  · intro b hb w hw
    simpa only [dense.loc.1,dense.loc.2,dense.sizes.2.2.1,Nat.one_mul,Nat.mul_comm,Nat.mul_one] using row_bound c.batches c.maxLength b w hb hw
  · intro h hh t ht
    simpa only [dense.out.1,dense.out.2,dense.sizes.2.2.2,Nat.mul_one] using row_bound c.heads c.outputTokens h t hh ht
  · intro h _ h' _ t ht t' ht' eq
    rw [dense.out.1,dense.out.2,Nat.mul_one,Nat.mul_one] at eq
    exact pair_injective c.outputTokens h t h' t' ht ht' eq

/-- Tensor identity and runtime flags; base addresses belong to Placement. -/
inductive DType where | fp32 | int32 deriving DecidableEq
structure Descriptor where
  (object device : Nat)
  dtype : DType
  (shape strides : List Nat)
  negative : Bool
structure ABI (c : Call) where
  tensor : Buffer → Descriptor
  shapes : (tensor .query).shape=[c.batches,c.heads,c.dim] ∧
    (tensor .key).shape=[c.cacheRows,c.kvHeads,c.dim] ∧
    (tensor .output).shape=[c.heads,c.outputTokens] ∧
    (tensor .locations).shape=[c.batches,c.maxLength] ∧
    (tensor .starts).shape=[c.batches] ∧ (tensor .lengths).shape=[c.batches]
  strides : (tensor .query).strides=[c.qBatch,c.qHead,c.qDim] ∧
    (tensor .key).strides=[c.kBatch,c.kHead,c.kDim] ∧
    (tensor .output).strides=[c.outHead,c.outSeq] ∧
    (tensor .locations).strides=[c.locBatch,c.locSeq] ∧
    (tensor .starts).strides=[1] ∧ (tensor .lengths).strides=[1]
  types : ∀ r, (tensor r).dtype=match r with | .query | .key | .output => .fp32 | _ => .int32
  device : ∀ r, (tensor r).device=(tensor .query).device
  ordinary : ∀ r, (tensor r).negative=Bool.false

#axiomsClean domain_valid
#specNonCircular Data avoiding [FP.Structural.exec,host]
end
end VeriTile.Bench.Repairs.TokenAttnLlama2.Whole
