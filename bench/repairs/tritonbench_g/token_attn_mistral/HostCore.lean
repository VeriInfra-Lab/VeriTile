import bench.repairs.tritonbench_g.token_attn_mistral.Stores
namespace VeriTile.Bench.Repairs.TokenAttnMistral.Host
open VeriTile.Triton FP.Structural Grid
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

inductive DType where | fp32 | int32
  deriving DecidableEq
structure Tensor where
  (object device allocationBase storageOffset capacity : Nat)
  region : RegionName
  dtype : DType
structure Vector extends Tensor where
  (size stride : Nat)
structure Matrix extends Tensor where
  (rows cols rowStride colStride : Nat)
structure Cube extends Tensor where
  (slots heads width slotStride headStride dimStride : Nat)
structure Call where
  probability : Matrix
  (value output : Cube)
  pages : Matrix
  (request unusedStart sequence start length : Vector)
  window : Nat

def batches (c : Call) := c.request.size
def heads (c : Call) := c.probability.rows
def group (c : Call) := heads c / c.value.heads

def params (c : Call) : Parameters where
  prob := c.probability.region
  value := c.value.region
  output := c.output.region
  tokens := Region.cast c.pages.region
  request := Region.cast c.request.region
  start := Region.cast c.start.region
  length := Region.cast c.sequence.region
  attentionLength := Region.cast c.length.region
  unusedStart := c.unusedStart.region
  window := c.window
  requestStride := c.request.stride
  attentionStartStride := c.start.stride
  attentionLengthStride := c.length.stride
  sequenceStride := c.sequence.stride
  requestBatchStride := c.pages.rowStride
  requestTokenStride := c.pages.colStride
  probHeadStride := c.probability.rowStride
  probTokenStride := c.probability.colStride
  valueBatchStride := c.value.slotStride
  valueHeadStride := c.value.headStride
  valueDimStride := c.value.dimStride
  outputBatchStride := c.output.slotStride
  outputHeadStride := c.output.headStride
  outputDimStride := c.output.dimStride
  kvGroup := group c
  width := c.value.width
  block := 128

/-- Ordinary FP32 host domain, including actual probability/metadata strides.
Pages, values and output use their contiguous layouts. -/
structure Layout (c : Call) : Prop where
  outputShape : c.output.slots=batches c ∧ c.output.heads=heads c ∧ c.output.width=c.value.width
  metadataShape : batches c≤c.start.size ∧ batches c≤c.length.size
  sequenceShape : batches c≤c.sequence.size
  unusedShape : batches c≤c.unusedStart.size
  valueDense : c.value.slotStride=c.value.heads*c.value.width ∧ c.value.headStride=c.value.width ∧ c.value.dimStride=1
  outputDense : c.output.slotStride=heads c*c.value.width ∧ c.output.headStride=c.value.width ∧ c.output.dimStride=1
  pagesDense : c.pages.rowStride=c.pages.cols ∧ c.pages.colStride=1
  groupPositive : 0<group c
  groupShape : heads c=c.value.heads*group c
  widthPowerTwo : ∃ k, c.value.width=2^k
  numericalDTypes : c.probability.dtype=.fp32 ∧ c.value.dtype=.fp32 ∧ c.output.dtype=.fp32
  metadataDTypes : c.pages.dtype=.int32 ∧ c.request.dtype=.int32 ∧ c.start.dtype=.int32 ∧ c.length.dtype=.int32
  otherMetadataDTypes : c.sequence.dtype=.int32 ∧ c.unusedStart.dtype=.int32
  sameDevice : c.probability.device=c.output.device ∧ c.value.device=c.output.device ∧
    c.pages.device=c.output.device ∧ c.request.device=c.output.device ∧
    c.start.device=c.output.device ∧ c.length.device=c.output.device ∧
    c.sequence.device=c.output.device ∧ c.unusedStart.device=c.output.device
  separated : Separate (params c)

def seq {α : Type} [Inhabited α] (c : Call) (m : Memory α) (b : Nat) : Nat :=
  (m c.length.region (b*c.length.stride)).read .nat

def fullLength {α : Type} [Inhabited α] (c : Call) (m : Memory α) (b : Nat) : Nat :=
  (m c.sequence.region (b*c.sequence.stride)).read .nat

def windowAt {α : Type} [Inhabited α] (c : Call) (m : Memory α) (b : Nat) : Nat :=
  fullLength c m b-c.window

def startAt {α : Type} [Inhabited α] (c : Call) (m : Memory α) (b : Nat) : Nat :=
  (m c.start.region (b*c.start.stride)).read .nat

def requestAt {α : Type} [Inhabited α] (c : Call) (m : Memory α) (b : Nat) : Nat :=
  (m c.request.region (b*c.request.stride)).read .nat

def pageAt {α : Type} [Inhabited α] (c : Call) (m : Memory α) (b n : Nat) : Nat :=
  (m c.pages.region (requestAt c m b*c.pages.rowStride+(windowAt c m b+n)*c.pages.colStride)).read .nat

/-- Nonnegative, bounded int32 metadata is ordinary input interpretation.
No normalization, positivity or finiteness is assumed of supplied P or V. -/
structure Data {α : Type} [Inhabited α] (c : Call) (m : Memory α) : Prop where
  lengths : ∀ b<batches c, fullLength c m b≤c.pages.cols
  probabilityWindow : ∀ b<batches c, startAt c m b+seq c m b≤c.probability.cols
  requests : ∀ b<batches c, requestAt c m b<c.pages.rows
  pages : ∀ b<batches c, ∀ n, n+windowAt c m b<fullLength c m b → pageAt c m b n<c.value.slots
  sequenceInt32 : ∀ b<batches c, fullLength c m b<2^31
  windowInt32 : c.window<2^31
  lengthInt32 : ∀ b<batches c, seq c m b<2^31
  startInt32 : ∀ b<batches c, startAt c m b<2^31
  requestInt32 : ∀ b<batches c, requestAt c m b<2^31
  pageInt32 : ∀ b<batches c, ∀ n, n+windowAt c m b<fullLength c m b → pageAt c m b n<2^31
  metadataCells : ∀ b<batches c,
    m c.sequence.region (b*c.sequence.stride)=.mk .nat (fullLength c m b) ∧
    m c.length.region (b*c.length.stride)=.mk .nat (seq c m b) ∧
    m c.start.region (b*c.start.stride)=.mk .nat (startAt c m b) ∧
    m c.request.region (b*c.request.stride)=.mk .nat (requestAt c m b)
  pageCells : ∀ b<batches c, ∀ n, n+windowAt c m b<fullLength c m b →
    m c.pages.region (requestAt c m b*c.pages.rowStride+(windowAt c m b+n)*c.pages.colStride)=.mk .nat (pageAt c m b n)
  probCells : ∀ h<heads c, ∀ n<c.probability.cols,
    ∃ x, m c.probability.region (h*c.probability.rowStride+n*c.probability.colStride)=.mk .fp32 x
  valueCells : ∀ v<c.value.slots, ∀ h<c.value.heads, ∀ d<c.value.width,
    ∃ x, m c.value.region (v*c.value.slotStride+h*c.value.headStride+d*c.value.dimStride)=.mk .fp32 x

theorem outputOffset_dense (c : Call) (layout : Layout c) (pid : PID)
    (i : TileIndex [(params c).width]) :
    offset (params c) pid i=(pid.1*heads c+pid.2)*c.value.width+i.1.val := by
  simp only [offset,params,layout.outputDense.1,layout.outputDense.2.1,layout.outputDense.2.2,Nat.mul_one]
  ring

theorem pair_injective (n : Nat) (a b c d : Nat) (hb : b<n) (hd : d<n)
    (same : a*n+b=c*n+d) : a=c ∧ b=d := by
  have positive : 0<n := by omega
  have q := congrArg (fun k => k/n) same
  have rem := congrArg (fun k => k%n) same
  have divA : (a*n+b)/n=a := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ positive,Nat.div_eq_of_lt hb]; omega
  have divC : (c*n+d)/n=c := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ positive,Nat.div_eq_of_lt hd]; omega
  have modA : (a*n+b)%n=b := by simp [Nat.add_mod,Nat.mod_eq_of_lt hb]
  have modC : (c*n+d)%n=d := by simp [Nat.add_mod,Nat.mod_eq_of_lt hd]
  exact ⟨by simpa [divA,divC] using q,by simpa [modA,modC] using rem⟩

theorem unique (c : Call) (layout : Layout c) : Unique (params c) (fullGrid (batches c) (heads c)) := by
  intro e he f hf same
  have eh := (mem_fullGrid _ _ _).mp ((mem_events _ _ _).mp he)
  have fh := (mem_fullGrid _ _ _).mp ((mem_events _ _ _).mp hf)
  rw [outputOffset_dense c layout,outputOffset_dense c layout] at same
  have inner := pair_injective c.value.width _ _ _ _ e.2.1.isLt f.2.1.isLt same
  have outer := pair_injective (heads c) _ _ _ _ eh.2 fh.2 inner.1
  apply Prod.ext
  · exact Prod.ext outer.1 outer.2
  · apply Prod.ext
    · exact Fin.ext inner.2
    · exact Subsingleton.elim _ _

#axiomsClean unique
end
end VeriTile.Bench.Repairs.TokenAttnMistral.Host
