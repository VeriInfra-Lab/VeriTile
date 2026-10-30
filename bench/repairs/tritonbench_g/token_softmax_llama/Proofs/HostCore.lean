import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.Stores
import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.Real
import bench.repairs.tritonbench_g.square_matrix.Proofs.Geometry
namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided.Host
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
structure Call where
  (input output : Matrix)
  (start length : Vector)
  maximum : Nat

def batches (c : Call) := c.start.size
def heads (c : Call) := c.input.rows
def width (c : Call) := VeriTile.Bench.Repairs.SquareMatrix.block c.maximum
def warps (c : Call) := if 4096 ≤ width c then 16 else if 2048 ≤ width c then 8 else 4

theorem width_cover (c : Call) : c.maximum ≤ width c :=
  VeriTile.Bench.Repairs.SquareMatrix.block_covers c.maximum

theorem width_positive (c : Call) (h : c.maximum≠0) : 0 < width c := by
  have := width_cover c
  omega


def params (c : Call) : Parameters where
  input := c.input.region
  output := c.output.region
  start := Region.cast c.start.region
  length := Region.cast c.length.region
  startStride := c.start.stride
  lengthStride := c.length.stride
  inputHeadStride := c.input.rowStride
  inputTokenStride := c.input.colStride
  outputHeadStride := c.output.rowStride
  outputTokenStride := c.output.colStride
  width := width c

structure Layout (c : Call) : Prop where
  outputShape : c.output.rows=heads c ∧ c.output.cols=c.input.cols
  lengthShape : batches c ≤ c.length.size
  outputInjective : ∀ h < heads c, ∀ t < c.input.cols, ∀ k < heads c, ∀ u < c.input.cols,
    h*c.output.rowStride+t*c.output.colStride=k*c.output.rowStride+u*c.output.colStride → h=k ∧ t=u
  numericalDTypes : c.input.dtype=.fp32 ∧ c.output.dtype=.fp32
  metadataDTypes : c.start.dtype=.int32 ∧ c.length.dtype=.int32
  sameDevice : c.input.device=c.output.device ∧ c.start.device=c.output.device ∧ c.length.device=c.output.device
  separated : Separate (params c)

def seq {α : Type} [Inhabited α] (c : Call) (m : Memory α) (b : Nat) :=
  (m c.length.region (b*c.length.stride)).read .nat
def startAt {α : Type} [Inhabited α] (c : Call) (m : Memory α) (b : Nat) :=
  (m c.start.region (b*c.start.stride)).read .nat

/-- Ordinary finite tensor and nonnegative INT32 metadata contract. Segments
may be empty, unordered and separated by holes; active segments do not overlap. -/
structure Data {α : Type} [Inhabited α] (c : Call) (m : Memory α) : Prop where
  sufficientMaximum : ∀ b < batches c, seq c m b ≤ c.maximum
  window : ∀ b < batches c, startAt c m b+seq c m b ≤ c.input.cols
  disjoint : ∀ b < batches c, ∀ d < batches c, b≠d → 0 < seq c m b → 0 < seq c m d →
    startAt c m b+seq c m b ≤ startAt c m d ∨ startAt c m d+seq c m d ≤ startAt c m b
  startInt32 : ∀ b < batches c, startAt c m b < 2^31
  lengthInt32 : ∀ b < batches c, seq c m b < 2^31
  metadataCells : ∀ b < batches c,
    m c.start.region (b*c.start.stride)=.mk .nat (startAt c m b) ∧
    m c.length.region (b*c.length.stride)=.mk .nat (seq c m b)
  inputCells : ∀ h < heads c, ∀ t < c.input.cols,
    ∃ x, m c.input.region (h*c.input.rowStride+t*c.input.colStride)=.mk .fp32 x

@[simp] theorem count_seq {α : Type} [Inhabited α] (c : Call) (m : Memory α) (pid : PID) :
    count (params c) (state m pid)=seq c m pid.1 := by simp [count,params,state,seq]
@[simp] theorem start_at {α : Type} [Inhabited α] (c : Call) (m : Memory α) (pid : PID) :
    startIndex (params c) (state m pid)=startAt c m pid.1 := by simp [startIndex,params,state,startAt]

theorem offset_strided {α : Type} [Inhabited α] (c : Call) (m : Memory α)
    (pid : PID) (i : TileIndex [(params c).width]) :
    offset (params c) m pid i=pid.2*c.output.rowStride+(startAt c m pid.1+i.1.val)*c.output.colStride := by
  rfl

theorem pair_injective (n : Nat) (a b c d : Nat) (hb : b < n) (hd : d < n)
    (same : a*n+b=c*n+d) : a=c ∧ b=d := by
  have positive : 0 < n := by omega
  have q := congrArg (fun k => k/n) same
  have rem := congrArg (fun k => k%n) same
  have divA : (a*n+b)/n=a := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ positive,Nat.div_eq_of_lt hb]; omega
  have divC : (c*n+d)/n=c := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ positive,Nat.div_eq_of_lt hd]; omega
  have modA : (a*n+b)%n=b := by simp [Nat.add_mod,Nat.mod_eq_of_lt hb]
  have modC : (c*n+d)%n=d := by simp [Nat.add_mod,Nat.mod_eq_of_lt hd]
  exact ⟨by simpa [divA,divC] using q,by simpa [modA,modC] using rem⟩

theorem disjoint_indices (a n i b k j : Nat) (hi : i < n) (hj : j < k)
    (same : a+i=b+j) (apart : a+n ≤ b ∨ b+k ≤ a) : False := by
  rcases apart with h|h <;> omega

theorem unique {α : Type} [Inhabited α] (c : Call) (m : Memory α) (layout : Layout c)
    (data : Data c m) (work : List PID)
    (legal : (fullGrid (batches c) (heads c)).Perm work) : Unique (params c) m work := by
  intro e he f hf same
  obtain ⟨em,ea⟩ := (mem_events _ _ _ _).mp he
  obtain ⟨fm,fa⟩ := (mem_events _ _ _ _).mp hf
  have eh := (mem_fullGrid _ _ _).mp (legal.mem_iff.mpr em)
  have fh := (mem_fullGrid _ _ _).mp (legal.mem_iff.mpr fm)
  have ei : e.2.1.val < seq c m e.1.1 := by simpa [active] using ea
  have fi : f.2.1.val < seq c m f.1.1 := by simpa [active] using fa
  have eb : startAt c m e.1.1+e.2.1.val < c.input.cols := lt_of_lt_of_le (Nat.add_lt_add_left ei _) (data.window e.1.1 eh.1)
  have fb : startAt c m f.1.1+f.2.1.val < c.input.cols := lt_of_lt_of_le (Nat.add_lt_add_left fi _) (data.window f.1.1 fh.1)
  rw [offset_strided c m,offset_strided c m] at same
  have inner := layout.outputInjective e.1.2 eh.2 _ eb f.1.2 fh.2 _ fb same
  have tokenEq := inner.2
  have batch : e.1.1=f.1.1 := by
    by_contra hn
    exact disjoint_indices _ _ _ _ _ _ ei fi tokenEq (data.disjoint e.1.1 eh.1 f.1.1 fh.1 hn (lt_of_le_of_lt (Nat.zero_le _) ei) (lt_of_le_of_lt (Nat.zero_le _) fi))
  apply Prod.ext
  · exact Prod.ext batch inner.1
  · apply Prod.ext
    · apply Fin.ext
      have := inner.2
      rw [batch] at this
      omega
    · exact Subsingleton.elim _ _

def inputReal (c : Call) (m : Memory Real.Scalar) (b h n : Nat) : ℝ :=
  WithBot.unbotD 0 ((m c.input.region (h*c.input.rowStride+(startAt c m b+n)*c.input.colStride)).read .fp32)
def target (c : Call) (m : Memory Real.Scalar) (b h n : Nat) : ℝ :=
  Real.exp (inputReal c m b h n)/(∑ j ∈ Finset.range (seq c m b),Real.exp (inputReal c m b h j))
def NumericData (c : Call) (m : Memory Real.Scalar) :=
  ∀ b < batches c, ∀ h < heads c, ∀ n < seq c m b,
    ∃ x : ℝ, (m c.input.region (h*c.input.rowStride+(startAt c m b+n)*c.input.colStride)).read .fp32=some x

theorem target_eq (c : Call) (m : Memory Real.Scalar) (b h n : Nat) :
    Real.target (params c) (state m (b,h)) n=target c m b h n := by
  simp [Real.target,Real.input,target,inputReal,params,state,startIndex,count,seq,startAt]

theorem numeric_row (c : Call) (m : Memory Real.Scalar) (numeric : NumericData c m)
    (b h : Nat) (hb : b < batches c) (hh : h < heads c) :
    Real.NumericInput (params c) (state m (b,h)) := by
  intro n hn
  simpa [params,state,startIndex,startAt] using numeric b hb h hh n (by simpa using hn)

#axiomsClean width_cover

#axiomsClean unique
#axiomsClean target_eq
#specNonCircular target avoiding [Grid.run,outputValues,Real.target]
end
end VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided.Host
