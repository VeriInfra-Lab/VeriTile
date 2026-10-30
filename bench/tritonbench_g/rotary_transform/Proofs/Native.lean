import bench.tritonbench_g.rotary_transform.Proofs.Source

namespace VeriTile.Bench.Repairs.RotaryTransform.Native
abbrev Word := BitVec 32

/-- Local selected PTX primitives over every raw FP32 word, including NaNs,
signed zeros and subnormals. No tensor-valued or output-valued port exists. -/
structure Instructions where
  mul : Word → Word → Word
  neg : Word → Word
  sub0 : Word → Word
  fma : Word → Word → Word → Word

def zero : Word := BitVec.ofNat 32 0
def one : Word := BitVec.ofNat 32 0x3f800000
structure Node where
  value : Word
  product : Option (Word × Word) := none
  deriving DecidableEq

def atom (x : Word) : Node := ⟨x,none⟩
/-- PTX places the coefficient before the input. Preserve that order even
for exceptional words; no FP commutativity is assumed. -/
def multiply (p : Instructions) (a b : Node) : Node :=
  ⟨p.mul b.value a.value,some (b.value,a.value)⟩
def subtract (p : Instructions) (a b : Node) : Node :=
  atom (match a.product with
    | some (x,y) => p.fma x y (p.neg b.value)
    | none => p.fma one a.value (p.neg b.value))
def add (p : Instructions) (a b : Node) : Node :=
  atom (match a.product with
    | some (x,y) => p.fma x y b.value
    | none => p.fma one a.value b.value)
def ops (p : Instructions) : Ops Node :=
  ⟨atom zero,atom one,(fun x => atom (p.sub0 x.value)),multiply p,subtract p,add p⟩

/-- Independent PTX graph. The high split result fuses sine*low; the high
adjacent result fuses cosine*high. Conjugation is sub.f32(+0,sine), whereas
subtraction of the rounded partner product uses neg.f32. -/
def graph (p : Instructions) (interleaved conjugate : Bool) (a b co sine : Word) (second : Bool) : Word :=
  let sn := if conjugate then p.sub0 sine else sine
  if second then
    if interleaved then p.fma co b (p.mul sn a) else p.fma sn a (p.mul co b)
  else p.fma co a (p.neg (p.mul sn b))

theorem source_graph_lowering (p : Instructions) (interleaved conjugate : Bool)
    (a b co sn : Node) (second : Bool) :
    component (sourcePair (ops p) interleaved conjugate a b co sn) second=
      atom (graph p interleaved conjugate a.value b.value co.value sn.value second) := by
  cases interleaved <;> cases conjugate <;> cases second <;> rfl

def liftMemory (m : Memory Word) : Memory Node := fun r a => atom (m r a)
def project (m : Memory Node) : Memory Word := fun r a => (m r a).value

def pairWord (c : Config) (p : Instructions) (w : Work) (m : Memory Word) (second : Bool) : Word :=
  graph p c.shape.interleaved c.shape.conjugate (m .input (c.cell w false)) (m .input (c.cell w true))
    (if c.hasCoeff w then m .cosine (c.coeff w) else one)
    (if c.hasCoeff w then m .sine (c.coeff w) else zero) second

/-- Raw-word stores implement the selected graph directly. The source
provenance interpretation is related by wordStep_project, not assumed. -/
def wordStep (c : Config) (p : Instructions) (w : Work) (m : Memory Word) : Memory Word :=
  let v0 := pairWord c p w m false
  let v1 := pairWord c p w m true
  let even := Stores.store m ((c.shape.output,c.cell w false),v0)
  Stores.store even ((c.shape.output,c.cell w true),v1)

theorem pairWord_source (c : Config) (p : Instructions) (w : Work) (m : Memory Node) (odd : Bool) :
    component (pairValues c (ops p) w m) odd=atom (pairWord c p w (project m) odd) :=
  by
  unfold pairValues pairWord
  rw [source_graph_lowering]
  simp only [project]
  split_ifs <;> rfl

theorem wordStep_project (c : Config) (p : Instructions) (w : Work) (m : Memory Node) :
    project (step c (ops p) w m)=wordStep c p w (project m) := by
  have even := pairWord_source c p w m false
  have odd := pairWord_source c p w m true
  funext r a
  simp only [component,Bool.false_eq_true,↓reduceIte] at even odd
  simp only [step,project,wordStep,Stores.store,Prod.mk.injEq]
  split_ifs <;> simp_all [pairValues,atom]

def wordExecute (c : Config) (p : Instructions) : List Work → Memory Word → Memory Word
  | [],m => m
  | w::ws,m => wordExecute c p ws (wordStep c p w m)

theorem wordExecute_project (c : Config) (p : Instructions) (ws : List Work) (m : Memory Node) :
    project (execute c (ops p) ws m)=wordExecute c p ws (project m) := by
  induction ws generalizing m with
  | nil => rfl
  | cons w ws ih =>
    change project (execute c (ops p) ws (step c (ops p) w m))=_
    rw [ih,wordStep_project]
    rfl

def selected (c : Config) (p : Instructions) (m : Memory Word) : Memory Word :=
  wordExecute c p (schedule c) m

theorem source_selected (c : Config) (valid : c.Valid) (p : Instructions) (m : Memory Word) :
    project (source c (ops p) (liftMemory m))=selected c p m := by
  rw [source_eq_run c valid]
  exact wordExecute_project c p _ _

theorem selected_readback (c : Config) (valid : c.Valid) (p : Instructions)
    (m : Memory Word) (w : Work) (h : c.Active w) (odd : Bool) :
    selected c p m c.shape.output (c.cell w odd)=pairWord c p w m odd := by
  have hr := run_readback c valid (ops p) (liftMemory m) w h odd
  have hv := congrArg Node.value hr
  rw [pairWord_source] at hv
  change project (run c (ops p) (liftMemory m)) _ _=_ at hv
  rw [show project (run c (ops p) (liftMemory m))=selected c p m from wordExecute_project c p _ _] at hv
  exact hv

theorem selected_frame (c : Config) (p : Instructions) (m : Memory Word) (r : Region) (a : Nat)
    (outside : ∀ w,c.Active w → ¬Writes c w r a) : selected c p m r a=m r a := by
  have hr := congrArg Node.value (run_frame c (ops p) (liftMemory m) r a outside)
  change project (run c (ops p) (liftMemory m)) r a=m r a at hr
  rw [show project (run c (ops p) (liftMemory m))=selected c p m from wordExecute_project c p _ _] at hr
  exact hr

/-- Exactly four local numerical obligations. No launch, coverage, source
correspondence, tensor result, frame or output equality is deferred. -/
structure Obligations (actual target : Instructions) : Prop where
  mul : ∀ a b,actual.mul a b=target.mul a b
  neg : ∀ a,actual.neg a=target.neg a
  sub0 : ∀ a,actual.sub0 a=target.sub0 a
  fma : ∀ a b c,actual.fma a b c=target.fma a b c

theorem instructions_eq (actual target : Instructions) (h : Obligations actual target) : actual=target := by
  cases actual;cases target
  simp only [Instructions.mk.injEq]
  exact ⟨funext fun a => funext (h.mul a),funext h.neg,funext h.sub0,funext fun a => funext fun b => funext (h.fma a b)⟩

theorem selected_equiv (actual target : Instructions) (h : Obligations actual target)
    (c : Config) (m : Memory Word) : selected c actual m=selected c target m := by
  rw [instructions_eq actual target h]

#specNonCircular graph avoiding [ops,sourcePair,source,selected]
#axiomsClean source_graph_lowering
#axiomsClean source_selected
#axiomsClean selected_readback
#axiomsClean selected_frame
#axiomsClean selected_equiv
end VeriTile.Bench.Repairs.RotaryTransform.Native
