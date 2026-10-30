import bench.tritonbench_g.rotary_emb.Proofs.Vector

namespace VeriTile.Bench.Repairs.RotaryEmb.Native
abbrev Word := BitVec 32

/-- Local selected PTX primitives over every raw FP32 word, including NaNs,
signed zeros and subnormals. No tensor-valued or output-valued port exists. -/
structure Instructions where
  mul : Word → Word → Word
  neg : Word → Word
  fma : Word → Word → Word → Word

structure Node where
  value : Word
  product : Option (Word × Word) := none
  deriving DecidableEq

def atom (x : Word) : Node := ⟨x,none⟩

def multiply (p : Instructions) (a b : Node) : Node :=
  ⟨p.mul a.value b.value,some (a.value,b.value)⟩

def subtract (p : Instructions) (a b : Node) : Node :=
  atom (match a.product with
    | some (x,y) => p.fma x y (p.neg b.value)
    | none => p.fma (BitVec.ofNat 32 0x3f800000) a.value (p.neg b.value))

def add (p : Instructions) (a b : Node) : Node :=
  atom (match b.product with
    | some (x,y) => p.fma x y a.value
    | none => p.fma (BitVec.ofNat 32 0x3f800000) b.value a.value)

def ops (p : Instructions) : Ops Node := ⟨multiply p,subtract p,add p⟩

/-- Independent selected PTX graph: even fuses x0*c0, while odd fuses x1*c1.
The multiplication and FMA operand orders follow the pinned PTX artifact. -/
def graph (p : Instructions) (x0 x1 c0 s0 c1 s1 : Word) (odd : Bool) : Word :=
  if odd then p.fma x1 c1 (p.mul x0 s1)
  else p.fma x0 c0 (p.neg (p.mul x1 s0))

theorem source_graph_lowering (p : Instructions) (x0 x1 c0 s0 c1 s1 : Node) (odd : Bool) :
    component (sourcePair (ops p) x0 x1 c0 s0 c1 s1) odd=
      atom (graph p x0.value x1.value c0.value s0.value c1.value s1.value odd) := by
  cases odd <;> rfl

def liftMemory (m : Memory Word) : Memory Node := fun r a => atom (m r a)
def project (m : Memory Node) : Memory Word := fun r a => (m r a).value

def pairWord (s : Shape) (p : Instructions) (w : Work) (m : Memory Word) (odd : Bool) : Word :=
  graph p (m (dataRegion w.key) (s.cell w false)) (m (dataRegion w.key) (s.cell w true))
    (m .cos (s.coefficient false w.token (2*w.pair)))
    (m .sin (s.coefficient true w.token (2*w.pair)))
    (m .cos (s.coefficient false w.token (2*w.pair+1)))
    (m .sin (s.coefficient true w.token (2*w.pair+1))) odd

/-- Raw-word stores implement the selected graph directly. The source
provenance interpretation is related by wordStep_project, not assumed. -/
def wordStep (s : Shape) (p : Instructions) (w : Work) (m : Memory Word) : Memory Word :=
  let v0 := pairWord s p w m false
  let v1 := pairWord s p w m true
  let even := Stores.store m ((dataRegion w.key,s.cell w false),v0)
  Stores.store even ((dataRegion w.key,s.cell w true),v1)

theorem pairWord_source (s : Shape) (p : Instructions) (w : Work) (m : Memory Node) (odd : Bool) :
    component (pairValues s (ops p) w m) odd=atom (pairWord s p w (project m) odd) :=
  source_graph_lowering p _ _ _ _ _ _ odd

theorem wordStep_project (s : Shape) (p : Instructions) (w : Work) (m : Memory Node) :
    project (step s (ops p) w m)=wordStep s p w (project m) := by
  have even := pairWord_source s p w m false
  have odd := pairWord_source s p w m true
  funext r a
  simp only [component,Bool.false_eq_true,↓reduceIte] at even odd
  simp only [step,sourceStep,project,wordStep,Stores.store,Prod.mk.injEq]
  split_ifs <;> simp_all [pairValues,atom]

def wordExecute (s : Shape) (p : Instructions) : List Work → Memory Word → Memory Word
  | [],m => m
  | w::ws,m => wordExecute s p ws (wordStep s p w m)

theorem wordExecute_project (s : Shape) (p : Instructions) (ws : List Work) (m : Memory Node) :
    project (execute s (ops p) ws m)=wordExecute s p ws (project m) := by
  induction ws generalizing m with
  | nil => rfl
  | cons w ws ih =>
    change project (execute s (ops p) ws (step s (ops p) w m))=_
    rw [ih,wordStep_project]
    rfl

def selected (s : Shape) (p : Instructions) (m : Memory Word) : Memory Word :=
  wordExecute s p (schedule s) m

theorem source_selected (s : Shape) (valid : s.Valid) (p : Instructions) (m : Memory Word) :
    project (source s (ops p) (liftMemory m))=selected s p m := by
  rw [source_eq_run s valid]
  exact wordExecute_project s p _ _

theorem selected_readback (s : Shape) (valid : s.Valid) (p : Instructions)
    (m : Memory Word) (w : Work) (h : s.Active w) (odd : Bool) :
    selected s p m (dataRegion w.key) (s.cell w odd)=pairWord s p w m odd := by
  have hr := run_readback s valid (ops p) (liftMemory m) w h odd
  have hv := congrArg Node.value hr
  rw [pairWord_source] at hv
  change project (run s (ops p) (liftMemory m)) _ _=_ at hv
  rw [show project (run s (ops p) (liftMemory m))=selected s p m from wordExecute_project s p _ _] at hv
  exact hv

theorem selected_frame (s : Shape) (p : Instructions) (m : Memory Word) (r : Region) (a : Nat)
    (outside : ∀ w,s.Active w → ¬Writes s w r a) : selected s p m r a=m r a := by
  have hr := congrArg Node.value (run_frame s (ops p) (liftMemory m) r a outside)
  change project (run s (ops p) (liftMemory m)) r a=m r a at hr
  rw [show project (run s (ops p) (liftMemory m))=selected s p m from wordExecute_project s p _ _] at hr
  exact hr

/-- Exactly three local numerical obligations. No launch, coverage, source
correspondence, tensor result, frame or output equality is deferred. -/
structure Obligations (actual target : Instructions) : Prop where
  mul : ∀ a b,actual.mul a b=target.mul a b
  neg : ∀ a,actual.neg a=target.neg a
  fma : ∀ a b c,actual.fma a b c=target.fma a b c

theorem instructions_eq (actual target : Instructions) (h : Obligations actual target) : actual=target := by
  cases actual;cases target
  simp only [Instructions.mk.injEq]
  exact ⟨funext fun a => funext (h.mul a),funext h.neg,funext fun a => funext fun b => funext (h.fma a b)⟩

theorem selected_equiv (actual target : Instructions) (h : Obligations actual target)
    (s : Shape) (m : Memory Word) : selected s actual m=selected s target m := by
  rw [instructions_eq actual target h]

#specNonCircular graph avoiding [ops,sourcePair,source,selected]
#axiomsClean source_graph_lowering
#axiomsClean source_selected
#axiomsClean selected_readback
#axiomsClean selected_frame
#axiomsClean selected_equiv
end VeriTile.Bench.Repairs.RotaryEmb.Native
