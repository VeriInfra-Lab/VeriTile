import bench.tritonbench_g.rotary_emb.Proofs.HostMetadata

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.HostGrid
open VeriTile.Triton FullMath

/-- `Bool.false` selects Q; `Bool.true` selects K. -/
def Plan.tensor (c : Plan) (x : Bool) : Tensor3 := if x then c.request.k else c.request.q
def region (Q K : RegionName) (x : Bool) : RegionName := if x then K else Q

def Tensor3.address (a : Tensor3) (n h d : Nat) : Nat :=
  n*a.batchStride + h*a.headStride + d*a.dimStride

/-- Ordinary finite tensor-layout injection, including dimensions that the
partial-rotation kernel leaves untouched. No contiguity restriction is imposed. -/
def Tensor3.Injective (a : Tensor3) : Prop :=
  ∀ n h d n' h' d', n<a.length → h<a.heads → d<a.width →
    n'<a.length → h'<a.heads → d'<a.width →
    a.address n h d = a.address n' h' d' → n=n' ∧ h=h' ∧ d=d'

def Plan.seq (c : Plan) (b : c.PID) (i : c.Lane) : Nat := b.2.val*16+i.1.val
def Plan.head (c : Plan) (b : c.PID) (i : c.Lane) : Nat := b.1.val*4+i.2.1.val
def Plan.offset (c : Plan) (x : Bool) (b : c.PID) (odd : Bool) (i : c.Lane) : Nat :=
  (c.tensor x).address (c.seq b i) (c.head b i) (FullMath.dimension odd i)
def Plan.active (c : Plan) (x : Bool) (b : c.PID) (i : c.Lane) : Prop :=
  c.seq b i < c.request.q.length ∧ c.head b i < (c.tensor x).heads

def Plan.trig (c : Plan) (b : c.PID) (odd : Bool) (i : c.Lane) : Nat :=
  c.seq b i*c.request.cos.batchStride + FullMath.dimension odd i*c.request.cos.dimStride

@[simp] theorem source_qAddress (c : Plan) (s : BlockState) (b : c.PID) (odd) (i) :
    c.parameters.qAddress (c.at s b) odd i = c.offset Bool.false b odd i := rfl
@[simp] theorem source_kAddress (c : Plan) (s : BlockState) (b : c.PID) (odd) (i) :
    c.parameters.kAddress (c.at s b) odd i = c.offset Bool.true b odd i := rfl
@[simp] theorem source_trigAddress (c : Plan) (s : BlockState) (b : c.PID) (odd) (i) :
    c.parameters.trigAddress (c.at s b) odd i = c.trig b odd i := rfl
@[simp] theorem source_activeQ (c : Plan) (s : BlockState) (b : c.PID) (i) :
    c.parameters.activeQ (c.at s b) i ↔ c.active Bool.false b i := Iff.rfl
@[simp] theorem source_activeK (c : Plan) (s : BlockState) (b : c.PID) (i) :
    c.parameters.activeK (c.at s b) i ↔ c.active Bool.true b i := Iff.rfl

/-- The only layout/domain premises needed below: assertion-matching sequence
shapes, nonnegative converted dimension, full tensor-layout injection, enough
logical data width, and bounds on the addresses actually loaded/stored. Trig
bounds use Cos strides for both input buffers, following the source. -/
structure Domain (c : Plan) (bounds : RegionBounds) (Q K Cos Sin : RegionName) : Prop where
  assertions : c.request.assertions
  headDim_nonneg : 0 ≤ c.signedHeadDim
  separate : RegionsSeparate Q K Cos Sin
  width : ∀ x, c.evenWidth ≤ (c.tensor x).width
  injective : ∀ x, (c.tensor x).Injective
  dataBounds : ∀ x n h d, n<c.request.q.length → h<(c.tensor x).heads →
    h<c.headBlocks*4 → d<c.evenWidth → (c.tensor x).address n h d < bounds (region Q K x)
  trigBounds : ∀ n d, n<c.request.q.length → d<c.evenWidth →
    n*c.request.cos.batchStride+d*c.request.cos.dimStride < bounds Cos ∧
    n*c.request.cos.batchStride+d*c.request.cos.dimStride < bounds Sin

theorem tensor_length (c : Plan) {bounds Q K Cos Sin} (D : Domain c bounds Q K Cos Sin)
    (x : Bool) : (c.tensor x).length = c.request.q.length := by
  cases x
  · rfl
  · exact D.assertions.2.2.1.trans D.assertions.1.symm

theorem dim_bound (c : Plan) (odd : Bool) (i : c.Lane) :
    FullMath.dimension odd i < c.evenWidth := dimension_lt_evenWidth odd i

theorem head_bound (c : Plan) (b : c.PID) (i : c.Lane) : c.head b i < c.headBlocks*4 := by
  have hb := b.1.isLt
  have hi : i.2.1.val < 4 := i.2.1.isLt
  unfold Plan.head
  omega

/-- Equal physical addresses of active output lanes force the same grid block,
parity and lane. This derives all within/cross-block collision conditions from
ordinary logical tensor-layout injectivity. -/
theorem address_injective (c : Plan) {bounds Q K Cos Sin} (D : Domain c bounds Q K Cos Sin)
    (x : Bool) (b b' : c.PID) (o o' : Bool) (i j : c.Lane)
    (hi : c.active x b i) (hj : c.active x b' j)
    (he : c.offset x b o i = c.offset x b' o' j) :
    b=b' ∧ o=o' ∧ i=j := by
  have di := lt_of_lt_of_le (dim_bound c o i) (D.width x)
  have dj := lt_of_lt_of_le (dim_bound c o' j) (D.width x)
  have hs := (D.injective x) _ _ _ _ _ _
    (by simpa [tensor_length c D x] using hi.1) hi.2 di
    (by simpa [tensor_length c D x] using hj.1) hj.2 dj he
  rcases hs with ⟨hs,hh,hd⟩
  have his : i.1.val < 16 := i.1.isLt
  have hjs : j.1.val < 16 := j.1.isLt
  have hih : i.2.1.val < 4 := i.2.1.isLt
  have hjh : j.2.1.val < 4 := j.2.1.isLt
  dsimp only [Plan.seq, Plan.head] at hs hh
  have hb0 : b.1 = b'.1 := Fin.ext (by omega)
  have hb1 : b.2 = b'.2 := Fin.ext (by omega)
  have hs' : i.1 = j.1 := Fin.ext (by omega)
  have hh' : i.2.1 = j.2.1 := Fin.ext (by omega)
  have ho : o=o' := by
    cases o <;> cases o' <;> simp_all [FullMath.dimension] <;> omega
  have hd' : i.2.2.1 = j.2.2.1 := by
    apply Fin.ext
    cases o <;> cases o' <;> simp_all [FullMath.dimension] <;> omega
  refine ⟨Prod.ext hb0 hb1,ho,?_⟩
  exact Prod.ext hs' (Prod.ext hh' (Prod.ext hd' (Subsingleton.elim _ _)))

theorem block_geometry (c : Plan) {bounds Q K Cos Sin} (D : Domain c bounds Q K Cos Sin)
    (s : BlockState) (b : c.PID) : c.parameters.Geometry (c.at s b) := by
  constructor
  · intro o o' i j hi hj he
    exact (address_injective c D Bool.false b b o o' i j hi hj he).2
  · intro o o' i j hi hj he
    exact (address_injective c D Bool.true b b o o' i j hi hj he).2

theorem block_bounds (c : Plan) {bounds Q K Cos Sin} (D : Domain c bounds Q K Cos Sin)
    (s : BlockState) (b : c.PID) : c.parameters.Bounds (c.at s b) bounds Q K Cos Sin := by
  constructor
  · intro o i hi
    exact D.dataBounds Bool.false _ _ _ hi.1 hi.2 (head_bound c b i) (dim_bound c o i)
  constructor
  · intro o i hi
    exact D.dataBounds Bool.true _ _ _ hi.1 hi.2 (head_bound c b i) (dim_bound c o i)
  · intro o i hi
    exact D.trigBounds _ _ hi (dimension_lt_evenWidth o i)

def Plan.blockFoot (c : Plan) (Q K : RegionName) (b : c.PID) : WriteFootprint :=
  fun a => ∃ x i o, c.active x b i ∧ a=(region Q K x,c.offset x b o i)

def Plan.foot (c : Plan) (Q K : RegionName) (bs : List c.PID) : WriteFootprint :=
  fun a => ∃ b ∈ bs, c.blockFoot Q K b a

theorem source_foot (c : Plan) (Q K : RegionName) (s : BlockState) (b : c.PID) :
    c.parameters.footprint (c.at s b) Q K = c.blockFoot Q K b := by
  funext a
  apply propext
  constructor
  · rintro (⟨hr,i,o,hi,ha⟩|⟨hr,i,o,hi,ha⟩)
    · exact ⟨Bool.false,i,o,hi,Prod.ext hr ha⟩
    · exact ⟨Bool.true,i,o,hi,Prod.ext hr ha⟩
  · rintro ⟨x,i,o,hi,ha⟩
    cases x
    · exact Or.inl ⟨congrArg Prod.fst ha,i,o,hi,congrArg Prod.snd ha⟩
    · exact Or.inr ⟨congrArg Prod.fst ha,i,o,hi,congrArg Prod.snd ha⟩

theorem block_disjoint (c : Plan) {bounds Q K Cos Sin} (D : Domain c bounds Q K Cos Sin)
    (b b' : c.PID) (hne : b≠b') : WriteFootprint.disjoint (c.blockFoot Q K b) (c.blockFoot Q K b') := by
  rintro a ⟨x,i,o,hi,ha⟩ ⟨y,j,o',hj,hb⟩
  have he := ha.symm.trans hb
  have hr := congrArg Prod.fst he
  have ha' := congrArg Prod.snd he
  cases x <;> cases y
  · exact hne (address_injective c D Bool.false b b' o o' i j hi hj ha').1
  · exact D.separate.1 hr
  · exact D.separate.1 hr.symm
  · exact hne (address_injective c D Bool.true b b' o o' i j hi hj ha').1

theorem trig_unwritten (c : Plan) {bounds Q K Cos Sin} (D : Domain c bounds Q K Cos Sin)
    (b : c.PID) (o : Nat) : ¬c.blockFoot Q K b (Cos,o) ∧ ¬c.blockFoot Q K b (Sin,o) := by
  constructor
  · rintro ⟨x,i,v,hi,he⟩
    have hr := congrArg Prod.fst he
    cases x
    · exact D.separate.2.1 hr.symm
    · exact D.separate.2.2.2.1 hr.symm
  · rintro ⟨x,i,v,hi,he⟩
    have hr := congrArg Prod.fst he
    cases x
    · exact D.separate.2.2.1 hr.symm
    · exact D.separate.2.2.2.2 hr.symm

end VeriTile.Bench.TritonBenchG.RotaryEmb.HostGrid
