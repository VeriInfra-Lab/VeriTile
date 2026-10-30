import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.Geometry
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence
noncomputable section
namespace Machine

inductive Expr where
  | lit (value : Int)
  | add (a b : Expr)
  | sub (a b : Expr)
  | mul (a b : Expr)
  | div (a b : Expr)
  | mod (a b : Expr)

def Expr.eval : Expr → Int
  | .lit x => x
  | .add a b => a.eval+b.eval
  | .sub a b => a.eval-b.eval
  | .mul a b => a.eval*b.eval
  | .div a b => a.eval/b.eval
  | .mod a b => a.eval%b.eval

def I32 (x : Int) := -(2^31)≤x ∧ x<2^31

def Expr.Valid : Expr → Prop
  | .lit x => I32 x
  | .add a b => a.Valid ∧ b.Valid ∧ I32 (a.eval+b.eval)
  | .sub a b => a.Valid ∧ b.Valid ∧ I32 (a.eval-b.eval)
  | .mul a b => a.Valid ∧ b.Valid ∧ I32 (a.eval*b.eval)
  | .div a b => a.Valid ∧ b.Valid ∧ 0≤a.eval ∧ 0<b.eval ∧ I32 (a.eval/b.eval)
  | .mod a b => a.Valid ∧ b.Valid ∧ 0≤a.eval ∧ 0<b.eval ∧ I32 (a.eval%b.eval)

/-- Canonical signed 32-bit wrap at every source arithmetic node. On admitted
nonnegative division operands this also agrees with truncation toward zero. -/
def Expr.word : Expr → BitVec 32
  | .lit x => BitVec.ofInt 32 x
  | .add a b => BitVec.ofInt 32 (a.word.toInt+b.word.toInt)
  | .sub a b => BitVec.ofInt 32 (a.word.toInt-b.word.toInt)
  | .mul a b => BitVec.ofInt 32 (a.word.toInt*b.word.toInt)
  | .div a b => BitVec.ofInt 32 (a.word.toInt/b.word.toInt)
  | .mod a b => BitVec.ofInt 32 (a.word.toInt%b.word.toInt)

theorem exact32 (x : Int) (hx : I32 x) : (BitVec.ofInt 32 x).toInt=x :=
  BitVec.toInt_ofInt_eq_self (by decide) hx.1 hx.2

theorem Expr.no_wrap (e : Expr) (h : e.Valid) : e.word.toInt=e.eval := by
  induction e with
  | lit x => exact exact32 _ h
  | add a b ha hb => simp only [word,eval,ha h.1,hb h.2.1]; exact exact32 _ h.2.2
  | sub a b ha hb => simp only [word,eval,ha h.1,hb h.2.1]; exact exact32 _ h.2.2
  | mul a b ha hb => simp only [word,eval,ha h.1,hb h.2.1]; exact exact32 _ h.2.2
  | div a b ha hb => simp only [word,eval,ha h.1,hb h.2.1]; exact exact32 _ h.2.2.2.2
  | mod a b ha hb => simp only [word,eval,ha h.1,hb h.2.1]; exact exact32 _ h.2.2.2.2

structure SeedStrides where
  batch : Nat
  head : Nat
  key : Nat
  value : Nat

def seedOffset (c : Shape) (s : SeedStrides) (h k v : Nat) :=
  (h/c.H)*s.batch+(h%c.H)*s.head+k*s.key+v*s.value

/-- Exact source expression ledger. count ranges through the loop endpoints,
so the N=1 initial S/DI/DG negative offsets and final dead pointers are retained.
No allocation-validity premise is imposed on a dead, undereferenced pointer. -/
def sourceTrees (c : Shape) (s : SeedStrides) (h tk tv e j count : Nat) : List Expr :=
  let n := Expr.lit c.N
  let k := Expr.lit c.K
  let v := Expr.lit c.V
  let head := Expr.lit h
  let plane := .mul k v
  let feature := .add (.add (.add (.mul (.mul (.lit tk) v) (.lit 64)) (.mul (.lit e) v))
    (.mul (.lit tv) (.lit 16))) (.lit j)
  let main := .add (.mul (.mul (.mul head n) k) v) feature
  let step := Expr.lit count
  let tiles := .mul (.div k (.lit 64)) (.div v (.lit 16))
  let dg := .add (.mul (.mul head n) tiles) (.add (.mul (.lit tk) (.div v (.lit 16))) (.lit tv))
  let seed := .add (.add (.add (.mul (.div head (.lit c.H)) (.lit s.batch))
    (.mul (.mod head (.lit c.H)) (.lit s.head)))
    (.mul (.add (.mul (.lit tk) (.lit 64)) (.lit e)) (.lit s.key)))
    (.mul (.add (.mul (.lit tv) (.lit 16)) (.lit j)) (.lit s.value))
  [n,k,v,.lit c.H,.lit c.KT,.lit c.VT,plane,main,seed,
   .add main (.mul step plane),
   .add main (.mul (.add step (.lit 1)) plane),
   .add main (.mul (.sub (.sub n (.lit 2)) step) plane),
   .add main (.mul (.sub (.sub n (.lit 1)) step) plane),
   .add dg (.mul (.sub (.sub n (.lit 2)) step) tiles),
   .add (.mul head n) step,
   .add (.mul head n) (.sub (.sub n (.lit 1)) step),
   .add (.mul (.mul head k) v) feature]

/-- Pointer roots at every reached loop endpoint. This includes one-past
forward pointers and negative dead reverse offsets; allocation membership is
required only for Access, while every formed pointer must remain representable. -/
def pointerExprs (c : Shape) (s : SeedStrides) (seed : Bool) (h tk tv e j count : Nat) (r : Region) : List Expr :=
  let xs := sourceTrees c s h tk tv e j count
  let indices : List Nat := match r with
    | .x => [7,9]
    | .output => [7,10,11]
    | .incoming => [12]
    | .dx => [11]
    | .tileGate => [13]
    | .gate => [14,15]
    | .dseed => [16]
    | .seed => if seed then [8] else []
    | .dgate | .caller => []
  indices.map (fun i => xs.getD i (.lit 0))

structure Allocation where
  storageBase : Int
  byteLength : Nat
  dataPointer : Int

def Allocation.Contains (a : Allocation) (offset : Int) : Prop :=
  a.storageBase≤a.dataPointer+4*offset ∧ a.dataPointer+4*offset+4≤a.storageBase+a.byteLength

def Allocation.Disjoint (a b : Allocation) : Prop :=
  a.storageBase+a.byteLength≤b.storageBase ∨ b.storageBase+b.byteLength≤a.storageBase

/-- Successful ordinary allocation/metadata representation. Inputs may alias;
output allocations are fresh against every caller-live allocation and one another. -/
structure ABI where
  allocation : Region → Allocation
  seedStrides : SeedStrides
  callerLive : List Allocation

def Written (r : Region) := r=.output ∨ r=.dx ∨ r=.tileGate ∨ r=.dseed ∨ r=.dgate

def offset (c : Shape) (abi : ABI) (r : Region) (p t j : Nat) : Int :=
  let h := p/c.tiles
  let tk := p%c.tiles/c.VT
  let tv := p%c.tiles%c.VT
  let k := tk*64+j/16
  let v := tv*16+j%16
  match r with
  | .x | .output | .incoming | .dx => matrixOffset c h t k v
  | .seed => seedOffset c abi.seedStrides h k v
  | .dseed => seedDenseOffset c h k v
  | .tileGate => scratchOffset c h t tk tv
  | .gate | .dgate => p*c.N+t
  | .caller => j

/-- Active complete words only; all unused cells remain frame cells. -/
def Active (c : Shape) (seed : Bool) (r : Region) (p t j : Nat) : Prop :=
  match r with
  | .x | .output | .incoming | .dx => p<c.programs ∧ t<c.N ∧ j<1024
  | .seed => seed ∧ c.N>0 ∧ p<c.programs ∧ t=0 ∧ j<1024
  | .dseed => (c.N>0 ∨ seed) ∧ p<c.programs ∧ t=0 ∧ j<1024
  | .tileGate => p<c.programs ∧ t<c.N ∧ j=0
  | .gate | .dgate => p<c.heads ∧ t<c.N ∧ j=0
  | .caller => False

structure Domain (c : Shape) (seed : Bool) (abi : ABI) : Prop where
  expressions : ∀ h tk tv e j count, h<c.heads → tk<c.KT → tv<c.VT → e<64 → j<16 → count<c.N →
    c.N>0 → ∀ x∈sourceTrees c abi.seedStrides h tk tv e j count, x.Valid
  formedPointers : ∀ h tk tv e j count, h<c.heads → tk<c.KT → tv<c.VT → e<64 → j<16 → count<c.N →
    ∀ r x, x∈pointerExprs c abi.seedStrides seed h tk tv e j count r →
      x.Valid ∧ 0≤(abi.allocation r).dataPointer+4*x.eval ∧
      (abi.allocation r).dataPointer+4*x.eval<2^64
  words : ∀ r p t j, Active c seed r p t j → (abi.allocation r).Contains (offset c abi r p t j)
  pointers : ∀ r p t j, Active c seed r p t j →
    0≤(abi.allocation r).dataPointer+4*offset c abi r p t j ∧
    (abi.allocation r).dataPointer+4*offset c abi r p t j+4≤2^64
  callerFresh : ∀ r, Written r → ∀ a∈abi.callerLive, (abi.allocation r).Disjoint a
  outputFresh : ∀ r q, Written r → Written q → r≠q → (abi.allocation r).Disjoint (abi.allocation q)

/-- Every listed intermediate has exact signed decoding; this includes dead
pointer offsets but does not claim they are live memory accesses. -/
theorem source_exact (c : Shape) (seed : Bool) (abi : ABI) (hd : Domain c seed abi)
    (h tk tv e j count : Nat) (hh : h<c.heads) (hk : tk<c.KT) (hv : tv<c.VT)
    (he : e<64) (hj : j<16) (hc : count<c.N) (hn : c.N>0)
    (x : Expr) (hx : x∈sourceTrees c abi.seedStrides h tk tv e j count) :
    x.word.toInt=x.eval := Expr.no_wrap x (hd.expressions h tk tv e j count hh hk hv he hj hc hn x hx)

theorem formed_pointer_exact (c : Shape) (seed : Bool) (abi : ABI) (hd : Domain c seed abi)
    (h tk tv e j count : Nat) (hh : h<c.heads) (hk : tk<c.KT) (hv : tv<c.VT)
    (he : e<64) (hj : j<16) (hc : count<c.N) (r : Region) (x : Expr)
    (hx : x∈pointerExprs c abi.seedStrides seed h tk tv e j count r) :
    (abi.allocation r).dataPointer+4*x.word.toInt=(abi.allocation r).dataPointer+4*x.eval ∧
    0≤(abi.allocation r).dataPointer+4*x.word.toInt ∧
    (abi.allocation r).dataPointer+4*x.word.toInt<2^64 := by
  have hr := hd.formedPointers h tk tv e j count hh hk hv he hj hc r x hx
  rw [Expr.no_wrap x hr.1]
  exact ⟨rfl,hr.2⟩

#axiomsClean formed_pointer_exact
#axiomsClean Expr.no_wrap
#axiomsClean source_exact
#guard_msgs (drop info) in
#auditModuleAxioms
end Machine
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence
