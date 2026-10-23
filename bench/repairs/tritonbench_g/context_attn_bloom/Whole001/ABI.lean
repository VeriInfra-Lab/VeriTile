import bench.repairs.tritonbench_g.context_attn_bloom.Whole001.FPGrid
import Mathlib.Data.BitVec

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.ABI
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program RealSetup
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000

def Fits32 (x : Int) : Prop := -(2^31)≤x ∧ x<2^31
abbrev word (x : Int) : BitVec 32 := BitVec.ofInt 32 x

theorem word_exact (x : Int) (h : Fits32 x) : (word x).toInt=x :=
  BitVec.toInt_ofInt_eq_self (by decide) h.1 h.2

inductive Expr where
  | value (x : Int)
  | add (a b : Expr)
  | sub (a b : Expr)
  | mul (a b : Expr)
  | min (a b : Expr)

def Expr.int : Expr → Int
  | .value x => x
  | .add a b => a.int+b.int
  | .sub a b => a.int-b.int
  | .mul a b => a.int*b.int
  | .min a b => Min.min a.int b.int

def Expr.bits : Expr → BitVec 32
  | .value x => word x
  | .add a b => a.bits+b.bits
  | .sub a b => a.bits-b.bits
  | .mul a b => a.bits*b.bits
  | .min a b => if a.bits.slt b.bits then a.bits else b.bits

def Expr.Bounded : Expr → Prop
  | .value x => Fits32 x
  | e@(.add a b) | e@(.sub a b) | e@(.mul a b) | e@(.min a b) =>
      Fits32 e.int ∧ a.Bounded ∧ b.Bounded

theorem expression_exact (e : Expr) (h : e.Bounded) : e.bits=word e.int ∧ e.bits.toInt=e.int := by
  induction e with
  | value x => exact ⟨rfl,word_exact x h⟩
  | add a b ia ib =>
    obtain ⟨wa,va⟩ := ia h.2.1
    obtain ⟨wb,vb⟩ := ib h.2.2
    have eq : (Expr.add a b).bits=word (a.int+b.int) := by simp only [Expr.bits,wa,wb,←BitVec.ofInt_add]
    exact ⟨eq,eq ▸ word_exact _ h.1⟩
  | sub a b ia ib =>
    obtain ⟨wa,va⟩ := ia h.2.1
    obtain ⟨wb,vb⟩ := ib h.2.2
    have eq : (Expr.sub a b).bits=word (a.int-b.int) := by
      simp only [Expr.bits,wa,wb,BitVec.sub_eq_add_neg,←BitVec.ofInt_neg,←BitVec.ofInt_add,Int.sub_eq_add_neg]
    exact ⟨eq,eq ▸ word_exact _ h.1⟩
  | mul a b ia ib =>
    obtain ⟨wa,va⟩ := ia h.2.1
    obtain ⟨wb,vb⟩ := ib h.2.2
    have eq : (Expr.mul a b).bits=word (a.int*b.int) := by simp only [Expr.bits,wa,wb,←BitVec.ofInt_mul]
    exact ⟨eq,eq ▸ word_exact _ h.1⟩
  | min a b ia ib =>
    obtain ⟨wa,va⟩ := ia h.2.1
    obtain ⟨wb,vb⟩ := ib h.2.2
    have eq : (Expr.min a b).bits=word (min a.int b.int) := by
      simp only [Expr.bits,BitVec.slt,va,vb]
      split
      next hh => rw [wa,min_eq_left (le_of_lt (of_decide_eq_true hh))]
      next hh => rw [wb,min_eq_right (by simpa using hh)]
    exact ⟨eq,eq ▸ word_exact _ h.1⟩

theorem mask_exact (a b : Expr) (ha : a.Bounded) (hb : b.Bounded) :
    a.bits.slt b.bits=decide (a.int<b.int) := by
  simp [BitVec.slt,(expression_exact a ha).2,(expression_exact b hb).2]

def n (x : Nat) : Expr := .value x

def programRoots (a : Host.Arguments) (u : Inputs) (l : Launch) : List Expr :=
  let p := Host.layout a
  let batch := n l.batch
  let head := n l.head
  let block := n l.block
  let start := Expr.value (u.start l.batch)
  let prompt := Expr.value (u.prompt l.batch)
  let seq := Expr.value (u.sequence l.batch)
  let request := Expr.value (u.request l.batch)
  let length := Expr.sub seq prompt
  let blockStart := Expr.mul (n p.BM) block
  let blockEnd := Expr.min (.add (.mul (.add block (n 1)) (n p.BM)) prompt) (.add length prompt)
  let stop := Expr.mul (n (if blockStart.int<length.int then 1 else 0)) blockEnd
  let kh := n (l.head/p.kvGroup)
  let coordinate := fun row h d sr sh sd => Expr.add (.add (.mul row (n sr)) (.mul h (n sh))) (.mul d (n sd))
  let queryExprs := (List.range p.BM).flatMap fun i =>
    let row := Expr.add (.mul block (n p.BM)) (n i)
    (List.range p.D).flatMap fun d =>
      [row,.add row prompt,
       coordinate (.add start row) head (n d) p.qRow p.qHead p.qDim,
       coordinate (.add start row) head (n d) p.outRow p.outHead p.outDim]
  let keyExprs := (List.range (count p l u)).flatMap fun c =>
    (List.range p.BN).flatMap fun j =>
      let column := Expr.add (.mul (n c) (n p.BN)) (n j)
      let pageOffset := Expr.add (.mul (n p.requestRow) request) (.mul (n p.requestToken) column)
      let page := Expr.value (if column.int<blockEnd.int then u.tokens pageOffset.int.toNat else 0)
      [column,pageOffset,page] ++ (List.range p.D).flatMap fun d =>
        [coordinate page kh (n d) p.kRow p.kHead p.kDim,
         coordinate page kh (n d) p.vRow p.vHead p.vDim]
  [batch,head,block,start,prompt,seq,request,length,blockStart,blockEnd,stop,
   .mul (n (count p l u)) (n p.BN),kh] ++
  ([a.batch,a.q.heads,Bounds.ceilBlocks a.maxInput (Host.block a),a.maxInput,
    p.BM,p.BN,p.D,p.headDim,p.kvGroup,p.qRow,p.qHead,p.qDim,p.kRow,p.kHead,p.kDim,
    p.vRow,p.vHead,p.vDim,p.outRow,p.outHead,p.outDim,p.requestRow,p.requestToken].map n) ++
  queryExprs ++ keyExprs

def Expr.nodes : Expr → List Expr
  | e@(.value _) => [e]
  | e@(.add a b) | e@(.sub a b) | e@(.mul a b) | e@(.min a b) => e::(a.nodes++b.nodes)

def programExpressions (a : Host.Arguments) (u : Inputs) (l : Launch) : List Expr :=
  (programRoots a u l).flatMap Expr.nodes

/-- A concrete formed head-stride product is rejected even if its lane would
be masked later. Source arithmetic overflow is not a generic grid deferral. -/
theorem excludes_overflowing_head_stride : ¬(Expr.mul (n 2) (n (2^30))).Bounded := by
  norm_num [Expr.Bounded,Expr.int,Fits32,n]

/-- These are concrete source bounds on every formed arithmetic expression,
including masked row/channel/time lanes and the final loop increment. This
is stronger than active storage bounds and contains no execution/result law. -/
structure IndexABI (a : Host.Arguments) (r : Regions) (u : Inputs) (base : RegionName → Nat) : Prop where
  formed : ∀ idx : GridIndex (Host.grid a), ∀ e∈programExpressions a u (Host.launchOf a idx), e.Bounded
  groups : a.q.heads<2^31 ∧ (Host.layout a).kvGroup<2^31
  pointerRoom : ∀ rg∈[r.query,r.key,r.value,r.out,r.start,r.sequence,r.prompt,r.request,r.tokens], ∀ x : Int, Fits32 x → 0≤(base rg : Int)+4*x ∧ (base rg : Int)+4*x<2^64

/-- A deliberately conservative base-room condition keeps even inactive
signed pointer expressions representable. Active accesses additionally use
Host.Admissible's actual finite allocation bounds. -/
structure SourceExact (a : Host.Arguments) (r : Regions) (u : Inputs) (base : RegionName → Nat) : Prop where
  operands : ∀ idx : GridIndex (Host.grid a), ∀ e∈programExpressions a u (Host.launchOf a idx),
    e.bits.toInt=e.int
  masks : ∀ idx : GridIndex (Host.grid a), ∀ e∈programExpressions a u (Host.launchOf a idx),
    ∀ f∈programExpressions a u (Host.launchOf a idx), e.bits.slt f.bits=decide (e.int<f.int)
  groupedHead : ∀ h<a.q.heads,
    ((BitVec.ofNat 32 h).sdiv (BitVec.ofNat 32 (Host.layout a).kvGroup)).toInt=
      (h/(Host.layout a).kvGroup : Nat)
  pointer : ∀ rg∈[r.query,r.key,r.value,r.out,r.start,r.sequence,r.prompt,r.request,r.tokens], ∀ x : Int, Fits32 x →
    (BitVec.ofNat 64 (base rg)+BitVec.ofInt 64 4 * BitVec.ofInt 64 x).toNat=
      ((base rg : Int)+4*x).toNat

theorem sdiv_nonnegative (a b : Nat) (ha : a<2^31) (hb : b<2^31) :
    ((BitVec.ofNat 32 a).sdiv (BitVec.ofNat 32 b)).toInt=(a/b : Nat) := by
  have av : (BitVec.ofNat 32 a).toInt=(a : Int) := word_exact (a : Int) (by constructor <;> omega)
  have bv : (BitVec.ofNat 32 b).toInt=(b : Int) := word_exact (b : Int) (by constructor <;> omega)
  have am : (BitVec.ofNat 32 a).msb=Bool.false := by simp [BitVec.msb_eq_toInt,av]
  have bm : (BitVec.ofNat 32 b).msb=Bool.false := by simp [BitVec.msb_eq_toInt,bv]
  have ab : a/b<2^31 := lt_of_le_of_lt (Nat.div_le_self _ _) ha
  have eq : (BitVec.ofNat 32 a).sdiv (BitVec.ofNat 32 b)=BitVec.ofNat 32 (a/b) := by
    apply BitVec.eq_of_toNat_eq
    simp only [BitVec.toNat_sdiv,am,bm,BitVec.toNat_udiv,BitVec.toNat_ofNat]
    change a % 2^32 / (b % 2^32) = (a/b) % 2^32
    norm_num only [Nat.reducePow]
    rw [Nat.mod_eq_of_lt (show a<4294967296 by omega),
      Nat.mod_eq_of_lt (show b<4294967296 by omega),
      Nat.mod_eq_of_lt (show a/b<4294967296 by omega)]
  rw [eq]
  apply word_exact ((a/b : Nat) : Int)
  constructor
  · exact le_trans (by norm_num) (Int.natCast_nonneg _)
  · exact_mod_cast ab

theorem source_exact (a : Host.Arguments) (r : Regions) (u : Inputs) (base : RegionName → Nat)
    (abi : IndexABI a r u base) : SourceExact a r u base := by
  refine ⟨?_,?_,?_,?_⟩
  · intro idx e he; exact (expression_exact e (abi.formed idx e he)).2
  · intro idx e he f hf;exact mask_exact e f (abi.formed idx e he) (abi.formed idx f hf)
  · intro h hh;exact sdiv_nonnegative _ _ (lt_trans hh abi.groups.1) abi.groups.2
  · intro rg hrg x hx
    have room := abi.pointerRoom rg hrg x hx
    change (BitVec.ofInt 64 (base rg)+BitVec.ofInt 64 4*BitVec.ofInt 64 x).toNat=_
    rw [←BitVec.ofInt_mul,←BitVec.ofInt_add,BitVec.toNat_ofInt]
    congr 1
    exact Int.emod_eq_of_lt room.1 room.2

#axiomsClean source_exact
#axiomsClean expression_exact
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.ABI
