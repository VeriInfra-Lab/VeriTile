import bench.tritonbench_g.attention_fwd_triton3.Proofs.Return

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3.Raw
open VeriTile.Triton Geometry Host Ownership
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
noncomputable section

abbrev Word := BitVec 32
abbrev Memory := RegionName → Nat → Word

inductive Tree (n : Nat)
  | leaf : Fin n → Tree n
  | node : Tree n → Tree n → Tree n

def Tree.indices : Tree n → List (Fin n)
  | .leaf i => [i]
  | .node a b => a.indices++b.indices

structure Order (n : Nat) where
  tree : Tree n
  complete : tree.indices.Perm (List.finRange n)

def reduce (op : Word → Word → Word) (xs : Fin n → Word) : Tree n → Word
  | .leaf i => xs i
  | .node a b => op (reduce op xs a) (reduce op xs b)

inductive Matrix where
  | mma16n8 | wgmma64n16 | wgmma64n32 | wgmma64n64 | wgmma64n128
  deriving DecidableEq

def Matrix.rows : Matrix → Nat
  | .mma16n8 => 16 | _ => 64

def Matrix.cols : Matrix → Nat
  | .mma16n8 => 8 | .wgmma64n16 => 16 | .wgmma64n32 => 32 | .wgmma64n64 => 64 | .wgmma64n128 => 128

theorem Matrix.rows_pos (kind : Matrix) : 0<kind.rows := by cases kind <;> decide

theorem Matrix.cols_pos (kind : Matrix) : 0<kind.cols := by cases kind <;> decide

/-- Only scalar instructions and one coordinate of a fixed K=8 matrix
instruction. There is no arbitrary-size dot or reduction numerical port. -/
structure Ops where
  add : Word → Word → Word
  sub : Word → Word → Word
  mul : Word → Word → Word
  div : Word → Word → Word
  max : Word → Word → Word
  exp2 : Word → Word
  log2 : Word → Word
  fma : Word → Word → Word → Word
  matrix8 : (kind : Matrix) → Fin kind.rows → Fin kind.cols → (Fin 8 → Word) → (Fin 8 → Word) → Word → Word

structure Schedule (c : Call) where
  qk : Matrix
  pv : Matrix
  fuseScoreSubtract : Bool
  maxOrder : Fin c.BM → Nat → Order c.BN
  sumOrder : Fin c.BM → Nat → Order c.BN
  probabilityOwner : Fin c.BM → Fin c.BN → Nat
  pvProbabilityOwner : Fin c.BM → Fin c.BN → Nat → Nat
  accumulatorAlphaOwner : Fin c.BM → Fin c.D → Nat → Nat
  centerStoreOwner : GridIndex c.grid → Fin c.BM → Fin 4 → Nat
  denominatorStoreOwner : GridIndex c.grid → Fin c.BM → Fin 4 → Nat
  outputStoreOwner : Fin c.BM → Fin c.D → Nat
  normalizationOwner : Fin c.BM → Fin c.D → Nat → Nat

/-- Source tmp!=tmp is NaN classification, with raw exponent/mantissa bits.
This is an integer definition, including both NaN signs and all payloads. -/
def isNaN (w : Word) : Bool := decide
  ((w.toNat/2^23)%256=255 ∧ w.toNat%2^23≠0)

def dot (ops : Ops) (kind : Matrix) (row col k : Nat)
    (a b : Fin k → Word) (seed : Word) : Word :=
  (List.range ((k+7)/8)).foldl (fun acc block=>
    ops.matrix8 kind ⟨row%kind.rows,Nat.mod_lt _ kind.rows_pos⟩ ⟨col%kind.cols,Nat.mod_lt _ kind.cols_pos⟩
      (fun lane=>if h:block*8+lane.val<k then a ⟨block*8+lane.val,h⟩ else 0)
      (fun lane=>if h:block*8+lane.val<k then b ⟨block*8+lane.val,h⟩ else 0) acc) seed

theorem dot_feature_coverage (k d : Nat) (hd : d<k) :
    d/8<(k+7)/8 ∧ d%8<8 ∧ d/8*8+d%8=d := by omega

theorem dot_feature_unique (d block lane : Nat) (hl : lane<8) (he : block*8+lane=d) :
    block=d/8 ∧ lane=d%8 := by omega

structure Carry (c : Call) where
  m : Fin c.BM → Nat → Word
  l : Fin c.BM → Nat → Word
  a : Fin c.BM → Fin c.D → Nat → Word

def initial (c : Call) (memory : Memory) (idx : GridIndex c.grid) : Carry c where
  m := fun r _=>if c.init≠0 then 0xff800000 else memory "M" (c.carryAddress idx r.val)
  l := fun r _=>if c.init≠0 then 0x3f800000 else memory "L" (c.carryAddress idx r.val)
  a := fun r d _=>if c.init≠0 then 0 else memory "O" (c.outAddress idx r.val d.val)

def kept (c : Call) (idx : GridIndex c.grid) (r key : Nat) : Bool :=
  decide (key<c.NK) && RealKeys.keep (c.tile idx) c.BM r c.window c.complement c.offset c.size key

def step (c : Call) (ops : Ops) (schedule : Schedule c) (memory : Memory) (scale : Word)
    (idx : GridIndex c.grid) (previous : Carry c) (start : Nat) : Carry c :=
  let z:=c.hz idx/c.H
  let h:=c.hz idx%c.H
  let hk:=h/(c.H/c.HK)
  let qkScale:=ops.mul scale 0x3fb8aa3b
  let qk:=fun (r : Fin c.BM) (j : Fin c.BN)=>
    let q:=fun d : Fin c.D=>if c.row idx r.val<c.NQ then memory "Q" (c.q.at z h (c.row idx r.val) d.val) else 0
    let k:=fun d : Fin c.D=>if start+j.val<c.NK then memory "K" (c.k.at z hk (start+j.val) d.val) else 0
    dot ops schedule.qk r.val j.val c.D q k 0
  let score:=fun (r : Fin c.BM) (j : Fin c.BN)=>
    let raw:=ops.mul qkScale (qk r j)
    if kept c idx r.val (start+j.val) then raw else 0xff800000
  let m:=fun r owner=>ops.max (previous.m r owner) (reduce ops.max (score r) (schedule.maxOrder r owner).tree)
  let p:=fun r j owner=>
    let delta:=if schedule.fuseScoreSubtract then
      ops.fma qkScale (qk r j) ((m r owner) ^^^ 0x80000000)
      else ops.sub (score r j) (m r owner)
    let weight:=ops.exp2 delta
    if kept c idx r.val (start+j.val) then weight else 0
  let alpha:=fun r owner=>
    let diff:=ops.sub (previous.m r owner) (m r owner)
    if isNaN diff then 0x3f800000 else ops.exp2 diff
  {m:=m,
   l:=fun r owner=>ops.fma (previous.l r owner) (alpha r owner)
     (reduce ops.add (fun j=>p r j (schedule.probabilityOwner r j)) (schedule.sumOrder r owner).tree),
   a:=fun r d owner=>dot ops schedule.pv r.val d.val c.BN
     (fun j=>p r j (schedule.pvProbabilityOwner r j owner))
     (fun j=>if start+j.val<c.NK then memory "V" (c.v.at z hk (start+j.val) d.val) else 0)
     (ops.mul (previous.a r d owner) (alpha r (schedule.accumulatorAlphaOwner r d owner)))}

def starts (c : Call) (idx : GridIndex c.grid) : List Nat :=
  let lo:=RealKeys.lower (c.tile idx) c.BM c.BN c.window c.complement c.offset c.size
  let hi:=RealKeys.upper (c.tile idx) c.BM c.BN c.NK c.window c.complement c.offset
  (List.range (RealKeys.blockCount lo hi c.BN)).map (fun j=>lo+j*c.BN)

def loop (c : Call) (ops : Ops) (schedule : Schedule c) (memory : Memory) (scale : Word)
    (idx : GridIndex c.grid) : Carry c :=
  (starts c idx).foldl (step c ops schedule memory scale idx) (initial c memory idx)

/-- Retain the indicated byte of each concrete candidate word. The selection
contains writer indices only; candidate values are computed by the full graph. -/
def gather (words : Fin 4 → Word) : Word :=
  (words 0 &&& 0x000000ff) ||| (words 1 &&& 0x0000ff00) |||
  (words 2 &&& 0x00ff0000) ||| (words 3 &&& 0xff000000)

theorem gather_same (word : Word) : gather (fun _=>word)=word := by
  unfold gather
  rw [← BitVec.and_or_distrib_left,← BitVec.and_or_distrib_left,← BitVec.and_or_distrib_left]
  exact BitVec.and_allOnes

/-- Native END graph returns all three values. L is read from initial memory
when END is set; no normalized state is silently reused as raw carry. -/
def output (c : Call) (ops : Ops) (schedule : Schedule c) (memory : Memory) (scale : Word)
    (idx : GridIndex c.grid) : Lane c.BM c.D → Word :=
  let state:=loop c ops schedule memory scale idx
  fun i=>match i with
    | .center r=>
      gather (fun byte=>
        let owner:=schedule.centerStoreOwner idx r byte
        if c.finish≠0 then ops.add (state.m r owner) (ops.log2 (state.l r owner)) else state.m r owner)
    | .denominator r=>
      if c.finish≠0 then memory "L" (c.carryAddress idx r.val)
      else gather (fun byte=>state.l r (schedule.denominatorStoreOwner idx r byte))
    | .output r d=>
      let owner:=schedule.outputStoreOwner r d
      if c.finish≠0 then ops.div (state.a r d owner) (state.l r (schedule.normalizationOwner r d owner))
      else state.a r d owner

/-- No reduction reordering is assumed: physical consumers retain their own
binary trees. Routes are structural metadata, never numerical equalities. -/
theorem schedule_covers (c : Call) (s : Schedule c) (r : Fin c.BM) (owner : Nat) :
    (s.maxOrder r owner).tree.indices.Perm (List.finRange c.BN) ∧
    (s.sumOrder r owner).tree.indices.Perm (List.finRange c.BN) :=
  ⟨(s.maxOrder r owner).complete,(s.sumOrder r owner).complete⟩

structure LocalLaws (a b : Ops) : Prop where
  add : ∀ x y,a.add x y=b.add x y
  sub : ∀ x y,a.sub x y=b.sub x y
  mul : ∀ x y,a.mul x y=b.mul x y
  div : ∀ x y,a.div x y=b.div x y
  max : ∀ x y,a.max x y=b.max x y
  exp2 : ∀ x,a.exp2 x=b.exp2 x
  log2 : ∀ x,a.log2 x=b.log2 x
  fma : ∀ x y z,a.fma x y z=b.fma x y z
  matrix8 : ∀ kind row col x y seed,a.matrix8 kind row col x y seed=b.matrix8 kind row col x y seed

theorem operations_eq (a b : Ops) (h : LocalLaws a b) : a=b := by
  cases a
  cases b
  simp only [Ops.mk.injEq]
  exact ⟨funext (fun x=>funext (h.add x)),funext (fun x=>funext (h.sub x)),
    funext (fun x=>funext (h.mul x)),funext (fun x=>funext (h.div x)),
    funext (fun x=>funext (h.max x)),funext h.exp2,funext h.log2,
    funext (fun x=>funext (fun y=>funext (h.fma x y))),
    funext (fun kind=>funext (fun row=>funext (fun col=>funext (fun x=>funext (fun y=>funext (h.matrix8 kind row col x y))))))⟩

theorem complete_equiv (c : Call) (a b : Ops) (h : LocalLaws a b) (schedule : Schedule c)
    (memory : Memory) (scale : Word) : output c a schedule memory scale=output c b schedule memory scale := by
  rw [operations_eq a b h]

#print axioms complete_equiv
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3.Raw
