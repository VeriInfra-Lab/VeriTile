import bench.repairs.tritonbench_g.token_attn_llama2.Proofs.Host

namespace VeriTile.Bench.Repairs.TokenAttnLlama2.Whole
open VeriTile.Triton FP.Structural
open FPValue
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

/-- Signed 32-bit wrap interpretation. The admitted nonnegative range makes
this encoding transparent; negative metadata is never silently clamped. -/
def signed32 (z : Int) : Int := (z+2147483648)%4294967296-2147483648

theorem signed32_exact (z : Int) (h0 : 0≤z) (h1 : z<2147483648) : signed32 z = z := by
  unfold signed32
  rw [Int.emod_eq_of_lt (by omega) (by omega)]
  omega

theorem grouped_head (c : Call) (m : Metadata c) (v : Valid c m)
    (h : Nat) (hh : h<c.heads) : h/(c.heads/c.kvHeads)<c.kvHeads := by
  apply (Nat.div_lt_iff_lt_mul v.groupsPositive).2
  simpa only [← v.groupsExact] using hh

variable {α : Type} [Inhabited α]

def lane (c : Call) (scale : ℝ) (t : Nat) : Index (c.parameters scale) :=
  (⟨t%32,Nat.mod_lt _ (by decide)⟩,PUnit.unit)

/-- Actual Q, Loc, K and Out addresses of every live lane lie in supplied
finite storage. Starts and lengths use the contiguous offsets b. -/
theorem access_bounds (c : Call) (scale : ℝ) (m : Metadata c) (x : Buffers α c)
    (s : State α) (v : Valid c m) (a : Program) (ha : a ∈ programs c.grid)
    (i : Index (c.parameters scale))
    (hi : active (c.parameters scale) (start c.grid a (initial c m x s)) i)
    (d : Fin c.dim) :
    a.1 < c.batches ∧
    queryOffset (c.parameters scale) (start c.grid a (initial c m x s)) d < c.qSize ∧
    c.locBatch*a.1+c.locSeq*(c.maxLength-m.len a.1+(a.2.2*32+i.1.val)) < c.locSize ∧
    keyOffset (c.parameters scale) (start c.grid a (initial c m x s)) i d < c.kSize ∧
    outputOffset (c.parameters scale) (start c.grid a (initial c m x s)) i < c.outSize := by
  obtain ⟨hb,hh,hn⟩ := (mem_programs c.grid a.1 a.2.1 a.2.2).mp ha
  have hl := length_bound c m v a.1 hb
  have hact := (initial_active c m x s scale a i).mp hi
  have ht : a.2.2*32+i.1.val<m.len a.1 := by omega
  have hstart := (v.starts ⟨a.1,hb⟩).2
  dsimp at hstart
  have hpage := (v.pages ⟨a.1,hb⟩ _ ht).2
  refine ⟨hb,?_,v.locationBounds _ hb _ hact,?_,?_⟩
  · simpa [queryOffset,Call.parameters,start,ids] using
      v.queryBounds _ hb _ hh _ d.isLt
  · have hbound := v.keyBounds _ hpage _ (grouped_head c m v _ hh) _ d.isLt
    simpa [keyOffset,keyLocation,hi,Call.parameters,start,ids,initial,Region.cast,
      Metadata.page,windowIndex,length,token,active,hact] using hbound
  · rw [initial_offset]
    exact v.outputBounds _ hh _ (by omega)

/-- Signed lengths, starts and live page IDs reconstruct exactly from the Nat
index model. They are source data obligations, not numerical assumptions. -/
theorem metadata_signed_exact (c : Call) (m : Metadata c) (v : Valid c m)
    (b : Fin c.batches) (t : Nat) (ht : t<m.len b.val) :
    (m.len b.val : Int) = m.lengths b ∧
    (m.begin b.val : Int) = m.starts b ∧
    (m.page b.val t : Int) =
      finiteRead m.locations (c.locBatch*b.val+c.locSeq*(c.maxLength-m.len b.val+t)) := by
  refine ⟨(signed_length c m v b).1,?_,?_⟩
  · simp [Metadata.begin,Int.toNat_of_nonneg (v.starts b).1]
  · exact Int.toNat_of_nonneg (v.pages b t ht).1

/-- All active addresses fit signed int32; the public envelope also includes
unmasked address expressions of inactive lanes. -/
theorem active_addresses_signed32 (c : Call) (scale : ℝ) (m : Metadata c)
    (x : Buffers α c) (s : State α) (v : Valid c m) (a : Program)
    (ha : a ∈ programs c.grid) (i : Index (c.parameters scale))
    (hi : active (c.parameters scale) (start c.grid a (initial c m x s)) i)
    (d : Fin c.dim) :
    signed32 (queryOffset (c.parameters scale) (start c.grid a (initial c m x s)) d) =
      queryOffset (c.parameters scale) (start c.grid a (initial c m x s)) d ∧
    signed32 (keyOffset (c.parameters scale) (start c.grid a (initial c m x s)) i d) =
      keyOffset (c.parameters scale) (start c.grid a (initial c m x s)) i d ∧
    signed32 (outputOffset (c.parameters scale) (start c.grid a (initial c m x s)) i) =
      outputOffset (c.parameters scale) (start c.grid a (initial c m x s)) i := by
  obtain ⟨_,hq,_,hk,ho⟩ := access_bounds c scale m x s v a ha i hi d
  have env := v.signedEnvelope
  refine ⟨signed32_exact _ (by omega) (by omega),
    signed32_exact _ (by omega) (by omega),signed32_exact _ (by omega) (by omega)⟩

def Unwritten (c : Call) (m : Metadata c) (r : RegionName) (o : Nat) : Prop :=
  r ≠ "Out" ∨ ∀ b<c.batches, ∀ h<c.heads, ∀ t<m.len b,
    o ≠ h*c.outHead+(m.begin b+t)*c.outSeq

def Result (c : Call) (m : Metadata c) (value : Nat → Nat → Nat → α)
    (s t : State α) : Prop :=
  (∀ b<c.batches, ∀ h<c.heads, ∀ n<m.len b,
    t.mem "Out" (h*c.outHead+(m.begin b+n)*c.outSeq) = .mk .fp32 (value b h n)) ∧
  (∀ r o, Unwritten c m r o → t.mem r o = s.mem r o)

def query {α : Type} [Inhabited α] (c : Call) (x : Buffers α c) (b h : Nat)
    (d : Fin c.dim) : α := finiteRead x.q (b*c.qBatch+h*c.qHead+d.val*c.qDim)

def key {α : Type} [Inhabited α] (c : Call) (m : Metadata c) (x : Buffers α c)
    (b h t : Nat) (d : Fin c.dim) : α :=
  finiteRead x.k (m.page b t*c.kBatch+(h/(c.heads/c.kvHeads))*c.kHead+d.val*c.kDim)

/-- Logical row readback and the frame follow from the complete launched grid. -/
theorem host_result (M : Algebra α) (c : Call) (scale : ℝ) (m : Metadata c)
    (x : Buffers α c) (s : State α) (v : Valid c m) :
    ∃ final, host M c scale (initial c m x s) = some (none,final) ∧
      Result c m (fun b h t => score M (c.parameters scale)
        (start c.grid (b,h,t/32) (initial c m x s)) (lane c scale t))
        (initial c m x s) final := by
  obtain ⟨final,he,hv,hf⟩ := host_correct M c scale m x s v
  refine ⟨final,he,?_,?_⟩
  · intro b hb h hh t ht
    obtain ⟨hmem,hindex⟩ := grid_covers c m v b h t hb hh ht
    have hl := length_bound c m v b hb
    have ha : active (c.parameters scale) (start c.grid (b,h,t/32) (initial c m x s))
        (lane c scale t) := by
      rw [initial_active]
      change c.maxLength-m.len b+(t/32*32+t%32)<c.maxLength
      omega
    have h := hv (b,h,t/32) hmem (lane c scale t) ha
    rw [initial_offset] at h
    simpa only [lane,Prod.fst,Prod.snd,hindex] using h
  · intro r o hu
    apply hf
    rcases hu with hu | hu
    · exact Or.inl hu
    · right
      intro a ha i hi
      obtain ⟨hb,hh,_⟩ := (mem_programs c.grid a.1 a.2.1 a.2.2).mp ha
      have hl := length_bound c m v a.1 hb
      rw [initial_active] at hi
      rw [initial_offset]
      exact hu a.1 hb a.2.1 hh _ (by omega)

#axiomsClean access_bounds
#axiomsClean metadata_signed_exact
#axiomsClean active_addresses_signed32
#axiomsClean host_result
end
end VeriTile.Bench.Repairs.TokenAttnLlama2.Whole
