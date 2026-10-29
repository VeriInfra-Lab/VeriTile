import bench.repairs.tritonbench_g.fused_rotary_embedding.Proofs.Source

namespace VeriTile.Bench.Repairs.FusedRotaryEmbedding.Native
abbrev Word := BitVec 32
structure Instructions where
  mul : Word → Word → Word
  neg : Word → Word
  fma : Word → Word → Word → Word
structure Node where
  value : Word
  product : Option (Word × Word) := none
def atom (x : Word) : Node := ⟨x,none⟩

/-- The selected K lowering exchanges multiplication operands. This records
the actual instruction order, without assuming FP commutativity. -/
def multiply (p : Instructions) (kind : Kind) (a b : Node) : Node :=
  if kind=.k then ⟨p.mul b.value a.value,some (b.value,a.value)⟩
  else ⟨p.mul a.value b.value,some (a.value,b.value)⟩
def subtract (p : Instructions) (a b : Node) : Node :=
  atom (match a.product with
    | some (x,y) => p.fma x y (p.neg b.value)
    | none => p.fma (BitVec.ofNat 32 0x3f800000) a.value (p.neg b.value))
def add (p : Instructions) (kind : Kind) (a b : Node) : Node :=
  if kind=.k then atom (match a.product with
    | some (x,y) => p.fma x y b.value
    | none => p.fma (BitVec.ofNat 32 0x3f800000) a.value b.value)
  else atom (match b.product with
    | some (x,y) => p.fma x y a.value
    | none => p.fma (BitVec.ofNat 32 0x3f800000) b.value a.value)
def ops (p : Instructions) (kind : Kind) : Ops Node := ⟨multiply p kind,subtract p,add p kind⟩

/-- Independent Q and K selected graphs, with exact mul/FMA operand order. -/
def graph (p : Instructions) (kind : Kind) (x0 x1 c s : Word) (second : Bool) : Word :=
  if kind=.k then
    if second then p.fma s x0 (p.mul c x1) else p.fma c x0 (p.neg (p.mul s x1))
  else if second then p.fma x1 c (p.mul x0 s) else p.fma x0 c (p.neg (p.mul x1 s))

theorem source_graph_lowering (p : Instructions) (kind : Kind) (x0 x1 c s : Node) (o : Bool) :
    component (sourcePair (ops p kind) x0 x1 c s) o=atom (graph p kind x0.value x1.value c.value s.value o) := by
  cases kind <;> cases o <;> rfl

def liftMemory (m : Memory Word) : Memory Node := fun r a => atom (m r a)
def project (m : Memory Node) : Memory Word := fun r a => (m r a).value
def pairWord (s : Shape) (p : Instructions) (w : Work) (m : Memory Word) (o : Bool) : Word :=
  if w.kind=.v then m .v (s.input w o)
  else graph p w.kind (m w.kind.input (s.input w false)) (m w.kind.input (s.input w true))
    (m .cos (s.coeff false w.token w.pair)) (m .sin (s.coeff true w.token w.pair)) o

theorem pairWord_source (s : Shape) (p : Instructions) (w : Work) (m : Memory Node) (o : Bool) :
    (component (pairValues s (ops p) w m) o).value=pairWord s p w (project m) o := by
  cases kw : w.kind <;> cases o <;>
    simp [pairValues,pairWord,kw,component,sourcePair,ops,multiply,subtract,add,graph,project,atom,Kind.input]

def wordStep (c : Config) (p : Instructions) (w : Work) (m : Memory Word) : Memory Word :=
  let low := pairWord c.shape p w m false
  let high := pairWord c.shape p w m true
  Stores.store (Stores.store m ((w.kind.output,c.output w false),low))
    ((w.kind.output,c.output w true),high)

theorem wordStep_project (c : Config) (p : Instructions) (w : Work) (m : Memory Node) :
    project (step c (ops p) w m)=wordStep c p w (project m) := by
  have low := pairWord_source c.shape p w m false
  have high := pairWord_source c.shape p w m true
  simp only [component,Bool.false_eq_true,↓reduceIte] at low high
  funext r a
  simp only [step,project,wordStep,Stores.store]
  split_ifs <;> simp_all

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
def selected (c : Config) (p : Instructions) (m : Memory Word) := wordExecute c p (schedule c.shape) m
theorem source_selected (c : Config) (hc : c.Valid) (p : Instructions) (m : Memory Word) :
    project (actualSource c (ops p) (liftMemory m))=selected c p m := by
  rw [actualSource_eq_run c hc]
  exact wordExecute_project c p _ _
theorem selected_readback (c : Config) (hc : c.Valid) (p : Instructions) (m : Memory Word)
    (w : Work) (hw : c.shape.Active w) (o : Bool) :
    selected c p m w.kind.output (c.output w o)=pairWord c.shape p w m o := by
  have h := congrArg Node.value (run_readback c hc (ops p) (liftMemory m) w hw o)
  rw [pairWord_source] at h
  change project (run c (ops p) (liftMemory m)) _ _=_ at h
  rw [show project (run c (ops p) (liftMemory m))=selected c p m from wordExecute_project c p _ _] at h
  exact h
theorem selected_frame (c : Config) (p : Instructions) (m : Memory Word) (r : Region) (a : Nat)
    (outside : ∀ w,c.shape.Active w → ¬Writes c w r a) : selected c p m r a=m r a := by
  have h := congrArg Node.value (run_frame c (ops p) (liftMemory m) r a outside)
  change project (run c (ops p) (liftMemory m)) r a=m r a at h
  rw [show project (run c (ops p) (liftMemory m))=selected c p m from wordExecute_project c p _ _] at h
  exact h

structure Obligations (actual target : Instructions) : Prop where
  mul : ∀ a b,actual.mul a b=target.mul a b
  neg : ∀ a,actual.neg a=target.neg a
  fma : ∀ a b c,actual.fma a b c=target.fma a b c
theorem instructions_eq (actual target : Instructions) (h : Obligations actual target) : actual=target := by
  cases actual;cases target
  simp only [Instructions.mk.injEq]
  exact ⟨funext fun a => funext (h.mul a),funext h.neg,funext fun a => funext fun b => funext (h.fma a b)⟩
theorem selected_equiv (actual target : Instructions) (h : Obligations actual target)
    (c : Config) (m : Memory Word) : selected c actual m=selected c target m := by
  rw [instructions_eq actual target h]

#specNonCircular graph avoiding [ops,sourcePair,actualSource,selected]
#axiomsClean source_graph_lowering
#axiomsClean source_selected
#axiomsClean selected_readback
#axiomsClean selected_frame
#axiomsClean selected_equiv
end VeriTile.Bench.Repairs.FusedRotaryEmbedding.Native
