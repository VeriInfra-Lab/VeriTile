import bench.repairs.tritonbench_g.token_attn_llama2.Proofs.Grid

/-! Host shape/stride/metadata adapter for the original score-only wrapper.
All buffers are caller-owned. There is no allocation, softmax, V, or backward
stage. Finite storage and nonnegative signed metadata are explicit. -/
namespace VeriTile.Bench.Repairs.TokenAttnLlama2.Whole
open VeriTile.Triton FP.Structural
open FPValue
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

structure Call where
  (batches heads kvHeads dim maxLength cacheRows outputTokens : Nat)
  (qBatch qHead qDim kBatch kHead kDim locBatch locSeq outHead outSeq : Nat)
  (qSize kSize locSize outSize : Nat)

/-- Shapes actually consumed by the wrapper; metadata vectors are contiguous. -/
def Call.shapes (c : Call) : List (List Nat) :=
  [[c.batches,c.heads,c.dim], [c.cacheRows,c.kvHeads,c.dim],
   [c.heads,c.outputTokens], [c.batches,c.maxLength], [c.batches], [c.batches]]

def Call.grid (c : Call) : Grid := ⟨c.batches,c.heads,(c.maxLength+31)/32⟩
def Call.warps (_ : Call) : Nat := 2
def Call.stages (_ : Call) : Nat := 1

def Call.parameters (c : Call) (scale : ℝ) : Parameters where
  q := "Q"
  k := "K"
  locations := ⟨"Loc"⟩
  starts := ⟨"Start"⟩
  lengths := ⟨"Length"⟩
  output := "Out"
  maxLength := c.maxLength
  locationBatchStride := c.locBatch
  locationSequenceStride := c.locSeq
  queryBatchStride := c.qBatch
  queryHeadStride := c.qHead
  queryDimStride := c.qDim
  keyBatchStride := c.kBatch
  keyHeadStride := c.kHead
  keyDimStride := c.kDim
  outputHeadStride := c.outHead
  outputSequenceStride := c.outSeq
  groups := c.heads / c.kvHeads
  dimensions := c.dim
  tokens := 32
  scale := scale

/-- Finite signed-int32 metadata payload before the Nat index adapter. -/
structure Metadata (c : Call) where
  lengths : Fin c.batches → Int
  starts : Fin c.batches → Int
  locations : Fin c.locSize → Int

def finiteRead {α : Type} [Inhabited α] {n : Nat} (xs : Fin n → α) (i : Nat) : α :=
  if h : i < n then xs ⟨i,h⟩ else default

@[simp] theorem finiteRead_lt {α : Type} [Inhabited α] {n : Nat}
    (xs : Fin n → α) (i : Fin n) : finiteRead xs i.val = xs i := by
  simp [finiteRead,i.isLt]

def Metadata.len {c : Call} (m : Metadata c) (b : Nat) : Nat :=
  (finiteRead m.lengths b).toNat

def Metadata.begin {c : Call} (m : Metadata c) (b : Nat) : Nat :=
  (finiteRead m.starts b).toNat

def Metadata.page {c : Call} (m : Metadata c) (b t : Nat) : Nat :=
  (finiteRead m.locations (c.locBatch*b+c.locSeq*(c.maxLength-m.len b+t))).toNat

/-- The floating payloads retain all FP32 words, including exceptional values. -/
structure Buffers (α : Type) (c : Call) where
  q : Fin c.qSize → α
  k : Fin c.kSize → α
  out : Fin c.outSize → α

def initial {α : Type} [Inhabited α] (c : Call) (m : Metadata c) (x : Buffers α c)
    (rest : State α) : State α :=
  {rest with mem := fun r o =>
    if r = "Q" then .mk .fp32 (finiteRead x.q o)
    else if r = "K" then .mk .fp32 (finiteRead x.k o)
    else if r = "Out" then .mk .fp32 (finiteRead x.out o)
    else if r = "Length" then .mk .nat (m.len o)
    else if r = "Start" then .mk .nat (m.begin o)
    else if r = "Loc" then .mk .nat ((finiteRead m.locations o).toNat)
    else rest.mem r o}

/-- Necessary admitted domain. Output addresses are unique across the whole
batch/head/token grid; only the active suffix of each Loc row is read. The
integer envelope also covers inactive lane address arithmetic before masking. -/
structure Valid (c : Call) (m : Metadata c) : Prop where
  dimension : c.dim = 16 ∨ c.dim = 32 ∨ c.dim = 64 ∨ c.dim = 128
  kvPositive : 0 < c.kvHeads
  groupsPositive : 0 < c.heads / c.kvHeads
  groupsExact : c.heads = c.kvHeads * (c.heads / c.kvHeads)
  lengths : ∀ b : Fin c.batches, 0 ≤ m.lengths b ∧ m.lengths b ≤ (c.maxLength : Int)
  starts : ∀ b : Fin c.batches, 0 ≤ m.starts b ∧ m.begin b.val + m.len b.val ≤ c.outputTokens
  pages : ∀ b : Fin c.batches, ∀ t, t < m.len b.val →
    0 ≤ finiteRead m.locations (c.locBatch*b.val+c.locSeq*(c.maxLength-m.len b.val+t)) ∧
    m.page b.val t < c.cacheRows
  queryBounds : ∀ b < c.batches, ∀ h < c.heads, ∀ d < c.dim,
    b*c.qBatch+h*c.qHead+d*c.qDim < c.qSize
  keyBounds : ∀ row < c.cacheRows, ∀ h < c.kvHeads, ∀ d < c.dim,
    row*c.kBatch+h*c.kHead+d*c.kDim < c.kSize
  locationBounds : ∀ b < c.batches, ∀ w < c.maxLength,
    c.locBatch*b+c.locSeq*w < c.locSize
  outputBounds : ∀ h < c.heads, ∀ t < c.outputTokens,
    h*c.outHead+t*c.outSeq < c.outSize
  outputInjective : ∀ h < c.heads, ∀ h' < c.heads,
    ∀ t < c.outputTokens, ∀ t' < c.outputTokens,
    h*c.outHead+t*c.outSeq = h'*c.outHead+t'*c.outSeq → h = h' ∧ t = t'
  segmentsInjective : ∀ b < c.batches, ∀ b' < c.batches,
    ∀ t < m.len b, ∀ t' < m.len b',
    m.begin b+t = m.begin b'+t' → b = b' ∧ t = t'
  signedEnvelope : c.batches + c.heads + c.kvHeads + c.dim + c.maxLength + 32 +
    c.cacheRows + c.outputTokens + c.qSize + c.kSize + c.locSize + c.outSize +
    c.batches*c.qBatch+c.heads*c.qHead+c.dim*c.qDim +
    c.cacheRows*c.kBatch+c.kvHeads*c.kHead+c.dim*c.kDim +
    c.locBatch*c.batches+c.locSeq*(c.maxLength+32) +
    c.heads*c.outHead+(c.outputTokens+32)*c.outSeq < 2147483648

/-- Read-region separation in the region model. Physical view separation is
stated separately below and is not discharged by distinct string names. -/
theorem parameters_separated (c : Call) (scale : ℝ) : Separated (c.parameters scale) := by
  constructor <;> simp [Call.parameters,Region.cast]

variable {α : Type} [Inhabited α]

@[simp] theorem initial_length (c : Call) (m : Metadata c) (x : Buffers α c)
    (s : State α) (scale : ℝ) (a : Program) :
    length (c.parameters scale) (start c.grid a (initial c m x s)) = m.len a.1 := by
  simp [length,Call.parameters,start,ids,initial,Region.cast]

@[simp] theorem initial_active (c : Call) (m : Metadata c) (x : Buffers α c)
    (s : State α) (scale : ℝ) (a : Program) (i : Index (c.parameters scale)) :
    active (c.parameters scale) (start c.grid a (initial c m x s)) i ↔
      c.maxLength-m.len a.1+(a.2.2*32+i.1.val) < c.maxLength := by
  simp [active,windowIndex,token,start,ids,Call.parameters,initial,length,Region.cast]

@[simp] theorem initial_offset (c : Call) (m : Metadata c) (x : Buffers α c)
    (s : State α) (scale : ℝ) (a : Program) (i : Index (c.parameters scale)) :
    outputOffset (c.parameters scale) (start c.grid a (initial c m x s)) i =
      a.2.1*c.outHead+(m.begin a.1+(a.2.2*32+i.1.val))*c.outSeq := by
  simp [outputOffset,token,start,ids,Call.parameters,initial,Region.cast]

theorem length_bound (c : Call) (m : Metadata c) (v : Valid c m)
    (b : Nat) (hb : b < c.batches) : m.len b ≤ c.maxLength := by
  have h := (v.lengths ⟨b,hb⟩).2
  unfold Metadata.len
  rw [show finiteRead m.lengths b = m.lengths ⟨b,hb⟩ by simp [finiteRead,hb]]
  exact Int.toNat_le.mpr h

/-- On this domain the signed source subtraction is exactly the Nat adapter. -/
theorem signed_length (c : Call) (m : Metadata c) (v : Valid c m)
    (b : Fin c.batches) :
    (m.len b.val : Int) = m.lengths b ∧
    ((c.maxLength-m.len b.val : Nat) : Int) = (c.maxLength : Int)-m.lengths b := by
  have hn := (v.lengths b).1
  have hl := length_bound c m v b.val b.isLt
  constructor
  · simp [Metadata.len,Int.toNat_of_nonneg hn]
  · rw [Int.natCast_sub hl]
    simp [Metadata.len,Int.toNat_of_nonneg hn]

theorem grid_unique (c : Call) (m : Metadata c) (x : Buffers α c)
    (s : State α) (scale : ℝ) (v : Valid c m) :
    Unique (c.parameters scale) c.grid (programs c.grid) (initial c m x s) := by
  intro a haMem b hbMem i j hi hj he
  rcases a with ⟨ba,ha,na⟩
  rcases b with ⟨bb,hb,nb⟩
  have haGrid := (mem_programs c.grid ba ha na).mp haMem
  have hbGrid := (mem_programs c.grid bb hb nb).mp hbMem
  have hla := length_bound c m v ba haGrid.1
  have hlb := length_bound c m v bb hbGrid.1
  rw [initial_active] at hi hj
  simp only [initial_offset] at he
  have hta : na*32+i.1.val < m.len ba := by dsimp at hi; omega
  have htb : nb*32+j.1.val < m.len bb := by dsimp at hj; omega
  have hsa := (v.starts ⟨ba,haGrid.1⟩).2
  have hsb := (v.starts ⟨bb,hbGrid.1⟩).2
  dsimp at hsa hsb
  obtain ⟨heHead,heToken⟩ := v.outputInjective ha haGrid.2.1 hb hbGrid.2.1
    _ (by omega) _ (by omega) he
  obtain ⟨heBatch,heLane⟩ := v.segmentsInjective ba haGrid.1 bb hbGrid.1
    _ hta _ htb heToken
  have hiBound : i.1.val < 32 := i.1.isLt
  have hjBound : j.1.val < 32 := j.1.isLt
  have heBlock : na = nb := by omega
  have heIndex : i = j := by
    apply Prod.ext
    · apply Fin.ext; omega
    · exact Subsingleton.elim _ _
  exact ⟨by simp [heBatch,heHead,heBlock],heIndex⟩

/-- Every logical token has exactly the block and lane launched by cdiv(M,32). -/
theorem grid_covers (c : Call) (m : Metadata c) (v : Valid c m)
    (b h t : Nat) (hb : b<c.batches) (hh : h<c.heads) (ht : t<m.len b) :
    (b,h,t/32) ∈ programs c.grid ∧ t/32*32+t%32 = t := by
  have hl := length_bound c m v b hb
  constructor
  · rw [mem_programs]
    refine ⟨hb,hh,?_⟩
    dsimp [Call.grid]
    omega
  · omega

/-- Source wrapper return: successful mutation produces Python None. -/
def host (M : Algebra α) (c : Call) (scale : ℝ) (s : State α) :
    Option (Option Unit × State α) :=
  (runPrograms M (c.parameters scale) c.grid (programs c.grid) s).map fun t => (none,t)

theorem host_correct (M : Algebra α) (c : Call) (scale : ℝ) (m : Metadata c)
    (x : Buffers α c) (s : State α) (v : Valid c m) :
    ∃ t, host M c scale (initial c m x s) = some (none,t) ∧
      Post M (c.parameters scale) c.grid (programs c.grid) (initial c m x s) t := by
  obtain ⟨t,he,hp⟩ := run_correct M (c.parameters scale) (parameters_separated c scale)
    c.grid (programs c.grid) (initial c m x s) (grid_unique c m x s scale v)
  exact ⟨t,by simp [host,he],hp⟩

#axiomsClean signed_length
#axiomsClean grid_unique
#axiomsClean grid_covers
#axiomsClean host_correct
end
end VeriTile.Bench.Repairs.TokenAttnLlama2.Whole
