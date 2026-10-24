import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.NativeProjection

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Native
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000

/-- Every reachable node has a native interpretation. In particular, no
unsupported callback, unknown literal, or mismatched reduction width reaches
publication. This is a syntactic invariant, not a numerical law. -/
def Supported (c : Call) : Expr → Prop
  | .input .. | .bits .. | .scale | .negInf | .initialCenter => True
  | .literal x => x=0 ∨ x=1 ∨ x=1.4426950408889634
  | .center a b | .add a b | .product a b | .difference a b | .centerDifference a b
  | .divide a b | .alpha a b => Supported c a ∧ Supported c b
  | .exponential a | .logarithm a => Supported c a
  | .fma a b acc => Supported c a ∧ Supported c b ∧ Supported c acc
  | .reduction _ _ n values => n=c.BN ∧ ∀ i,Supported c (values i)
  | .pendingDot _ _ _ a b => (∀ i,Supported c (a i)) ∧ ∀ i,Supported c (b i)
  | .queryDot _ _ _ a b seed =>
      (∀ i,Supported c (a i)) ∧ (∀ i,Supported c (b i)) ∧ Supported c seed
  | .valueDot _ _ _ a b previous alpha =>
      (∀ i,Supported c (a i)) ∧ (∀ i,Supported c (b i)) ∧ Supported c previous ∧ Supported c alpha
  | .unsupported => False

theorem supported_zero (c : Call) : Supported c (literal 0) := by
  norm_num [literal,ScaleBinding.marker,Supported]
theorem supported_one (c : Call) : Supported c (literal 1) := by
  norm_num [literal,ScaleBinding.marker,Supported]
theorem supported_log2e (c : Call) : Supported c (literal 1.4426950408889634) := by
  norm_num [literal,ScaleBinding.marker,Supported]

theorem supported_mul (c : Call) (a b : Expr) (ha : Supported c a) (hb : Supported c b) :
    Supported c (consume .mul a b) := ⟨ha,hb⟩
theorem supported_sub (c : Call) (a b : Expr) (ha : Supported c a) (hb : Supported c b) :
    Supported c (consume .sub a b) := by simp only [consume]; split_ifs <;> exact ⟨ha,hb⟩
theorem supported_max (c : Call) (a b : Expr) (ha : Supported c a) (hb : Supported c b) :
    Supported c (consume .max a b) := ⟨ha,hb⟩
theorem supported_div (c : Call) (a b : Expr) (ha : Supported c a) (hb : Supported c b) :
    Supported c (consume .div a b) := ⟨ha,hb⟩
theorem supported_exp (c : Call) (a : Expr) (ha : Supported c a) :
    Supported c (unary .exp2 a) := by cases a <;> exact ha
theorem supported_log (c : Call) (a : Expr) (ha : Supported c a) :
    Supported c (unary .log2 a) := ha

theorem supported_add_dot (c : Call) (seed : Expr) (row col k : Nat) (a b : Fin k → Expr)
    (hs : Supported c seed) (ha : ∀ i,Supported c (a i)) (hb : ∀ i,Supported c (b i)) :
    Supported c (consume .add seed (.pendingDot row col k a b)) := by
  cases seed <;> simp only [consume,Supported] at hs ⊢ <;> tauto

theorem supported_fma (c : Call) (a b acc : Expr)
    (ha : Supported c a) (hb : Supported c b) (hc : Supported c acc) :
    Supported c (consume .add (.product a b) acc) := by
  cases acc <;> simp only [consume,Supported] at hc ⊢ <;> tauto

theorem supported_add_log (c : Call) (a b : Expr) (ha : Supported c a) (hb : Supported c b) :
    Supported c (consume .add a (.logarithm b)) := by
  cases a <;> simp only [consume,Supported] at ha ⊢ <;> tauto

theorem supported_load (c : Call) (a : Reference.Inputs Expr)
    (good : ∀ rg off,Supported c (a.read rg off)) (ptr : BlockPtr)
    (shape : TileShape) (checked : List Nat) (i : TileIndex shape) :
    Supported c (Reference.loadBlock backend a ptr shape checked i) := by
  unfold Reference.loadBlock
  change Supported c (if _ then _ else literal 0)
  split_ifs
  · exact good _ _
  · exact supported_zero c

theorem supported_reduction (c : Call) (maximum : Bool)
    (values : TileIndex [c.BM,c.BN] → Expr) (good : ∀ i,Supported c (values i))
    (i : TileIndex [c.BM]) :
    Supported c (reduction maximum (shape:=[c.BM,c.BN]) ⟨1,by simp⟩ Bool.false values i) :=
  ⟨rfl,fun j => good _⟩

theorem supported_dot (c : Call) (row col k : Nat) (a b : Fin k → Expr)
    (ha : ∀ i,Supported c (a i)) (hb : ∀ i,Supported c (b i)) :
    Supported c (.pendingDot row col k a b) := ⟨ha,hb⟩

structure SupportedCarry (c : Call) (old : Reference.Carry c Expr) : Prop where
  center : ∀ i,Supported c (old.center i)
  mass : ∀ i,Supported c (old.mass i)
  weighted : ∀ i,Supported c (old.weighted i)

theorem initial_supported (c : Call) (a : Reference.Inputs Expr)
    (good : ∀ rg off,Supported c (a.read rg off)) :
    SupportedCarry c (Reference.initial c backend a) := by
  constructor <;> intro i
  all_goals simp only [Reference.initial,backend,algebra]
  all_goals split_ifs
  · norm_num [literal,ScaleBinding.marker,consume,Supported]
  · exact good _ _
  · norm_num [literal,ScaleBinding.marker,consume,Supported]
  · exact good _ _
  · exact supported_zero c
  · exact good _ _

theorem scale_supported (c : Call) (scale : c.scale=ScaleBinding.marker) :
    Supported c (Reference.scale c backend) := by
  exact supported_mul c _ _ (by simp [backend,algebra,scale,literal,Supported]) (supported_log2e c)

theorem advance_supported (c : Call) (a : Reference.Inputs Expr) (cur : Int)
    (old : Reference.Carry c Expr) (scale : c.scale=ScaleBinding.marker)
    (good : ∀ rg off,Supported c (a.read rg off))
    (carry : SupportedCarry c old) :
    SupportedCarry c (Reference.advance c backend a cur old) := by
  constructor <;> intro i
  all_goals dsimp only [Reference.advance,backend,algebra]
  all_goals repeat' first
    | contradiction
    | exact carry.center _
    | exact carry.mass _
    | exact carry.weighted _
    | exact supported_zero c
    | exact supported_one c
    | exact scale_supported c scale
    | exact True.intro
    | apply supported_max
    | apply supported_mul
    | apply supported_sub
    | apply supported_exp
    | apply supported_fma
    | apply supported_add_dot
    | apply supported_reduction
    | apply supported_dot
    | exact supported_load c a good _ _ _ _
    | change Supported c (Reference.loadBlock backend a _ _ _ _)
    | intro j
    | split_ifs
    | dsimp only [matrix,Reference.query,Reference.scale,ScaleBinding.template,backend,algebra]

#axiomsClean initial_supported
#axiomsClean advance_supported
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Native
