import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Host
import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Proofs.RealBase
import Mathlib.Data.BitVec
namespace VeriTile.Bench.Repairs.MatmulDequantizeWhole001.ABI
open VeriTile.Triton
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
def Fits32 (x : Int) : Prop := 0≤x ∧ x<2^31
abbrev word (x : Int) : BitVec 32 := BitVec.ofInt 32 x

theorem word_exact (x : Int) (h : Fits32 x) : (word x).toInt=x :=
  BitVec.toInt_ofInt_eq_self (by decide) (by have hh := h.1; omega) h.2

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

theorem srem_nonnegative (a b : Nat) (ha : a<2^31) (hb : b<2^31) :
    ((BitVec.ofNat 32 a).srem (BitVec.ofNat 32 b)).toInt=(a%b : Nat) := by
  have av : (BitVec.ofNat 32 a).toInt=(a : Int) := word_exact (a : Int) (by constructor <;> omega)
  have bv : (BitVec.ofNat 32 b).toInt=(b : Int) := word_exact (b : Int) (by constructor <;> omega)
  have am : (BitVec.ofNat 32 a).msb=Bool.false := by simp [BitVec.msb_eq_toInt,av]
  have bm : (BitVec.ofNat 32 b).msb=Bool.false := by simp [BitVec.msb_eq_toInt,bv]
  have ab : a%b<2^31 := lt_of_le_of_lt (Nat.mod_le _ _) ha
  have eq : (BitVec.ofNat 32 a).srem (BitVec.ofNat 32 b)=BitVec.ofNat 32 (a%b) := by
    apply BitVec.eq_of_toNat_eq
    simp only [BitVec.toNat_srem,am,bm,BitVec.toNat_umod,BitVec.toNat_ofNat]
    change a % 2^32 % (b % 2^32) = (a%b) % 2^32
    norm_num only [Nat.reducePow]
    rw [Nat.mod_eq_of_lt (show a<4294967296 by omega),
      Nat.mod_eq_of_lt (show b<4294967296 by omega),
      Nat.mod_eq_of_lt (show a%b<4294967296 by omega)]
  rw [eq]
  apply word_exact ((a%b : Nat) : Int)
  constructor
  · exact Int.natCast_nonneg _
  · exact_mod_cast ab

inductive Expr where
  | value (x : Int)
  | add (a b : Expr)
  | sub (a b : Expr)
  | mul (a b : Expr)
  | min (a b : Expr)
  | div (a b : Expr)
  | mod (a b : Expr)

def Expr.int : Expr → Int
  | .value x => x
  | .add a b => a.int+b.int
  | .sub a b => a.int-b.int
  | .mul a b => a.int*b.int
  | .min a b => Min.min a.int b.int
  | .div a b => (a.int.toNat / b.int.toNat : Nat)
  | .mod a b => (a.int.toNat % b.int.toNat : Nat)

def Expr.bits : Expr → BitVec 32
  | .value x => word x
  | .add a b => a.bits+b.bits
  | .sub a b => a.bits-b.bits
  | .mul a b => a.bits*b.bits
  | .min a b => if a.bits.slt b.bits then a.bits else b.bits
  | .div a b => a.bits.sdiv b.bits
  | .mod a b => a.bits.srem b.bits

def Expr.Bounded : Expr → Prop
  | .value x => Fits32 x
  | e@(.add a b) | e@(.sub a b) | e@(.mul a b) | e@(.min a b) | e@(.div a b) | e@(.mod a b) =>
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
  | div a b ia ib =>
    obtain ⟨wa,va⟩ := ia h.2.1
    obtain ⟨wb,vb⟩ := ib h.2.2
    have ha : Fits32 a.int := by cases a <;> first | exact h.2.1.1 | exact h.2.1
    have hb : Fits32 b.int := by cases b <;> first | exact h.2.2.1 | exact h.2.2
    have an : (a.int.toNat : Int)=a.int := Int.toNat_of_nonneg ha.1
    have bn : (b.int.toNat : Int)=b.int := Int.toNat_of_nonneg hb.1
    have eq : (Expr.div a b).bits.toInt=(Expr.div a b).int := by
      simp only [Expr.bits,wa,wb,Expr.int]
      rw [←an,←bn]
      exact sdiv_nonnegative _ _ (by have h := ha.2; omega) (by have h := hb.2; omega)
    refine ⟨?_,eq⟩
    apply BitVec.eq_of_toInt_eq
    rw [eq,word_exact _ h.1]
  | mod a b ia ib =>
    obtain ⟨wa,va⟩ := ia h.2.1
    obtain ⟨wb,vb⟩ := ib h.2.2
    have ha : Fits32 a.int := by cases a <;> first | exact h.2.1.1 | exact h.2.1
    have hb : Fits32 b.int := by cases b <;> first | exact h.2.2.1 | exact h.2.2
    have an : (a.int.toNat : Int)=a.int := Int.toNat_of_nonneg ha.1
    have bn : (b.int.toNat : Int)=b.int := Int.toNat_of_nonneg hb.1
    have eq : (Expr.mod a b).bits.toInt=(Expr.mod a b).int := by
      simp only [Expr.bits,wa,wb,Expr.int]
      rw [←an,←bn]
      exact srem_nonnegative _ _ (by have h := ha.2; omega) (by have h := hb.2; omega)
    refine ⟨?_,eq⟩
    apply BitVec.eq_of_toInt_eq
    rw [eq,word_exact _ h.1]

theorem mask_exact (a b : Expr) (ha : a.Bounded) (hb : b.Bounded) :
    a.bits.slt b.bits=decide (a.int<b.int) := by
  simp [BitVec.slt,(expression_exact a ha).2,(expression_exact b hb).2]

def n (x : Nat) : Expr := .value x
def cdiv (x y : Expr) : Expr := .div (.sub (.add x y) (n 1)) y
def coord (r c : Expr) (t : Host.Tensor) : Expr := .add (.mul r (n t.rowStride)) (.mul c (n t.columnStride))

def matmulRoots (a : Host.Arguments) (cfg : Host.FirstConfig) (second : Bool) (pid : Nat) : List Expr :=
  let bm := if second then 64 else cfg.bm
  let bn := if second then 128 else cfg.bn
  let nm := cdiv (n a.M) (n bm)
  let nn := cdiv (n a.N) (n bn)
  let nk := cdiv (n a.K) (n 32)
  let ng := Expr.mul (n 8) nn
  let gid := Expr.div (n pid) ng
  let first := Expr.mul gid (n 8)
  let gs := Expr.min (.sub nm first) (n 8)
  let pm := Expr.add first (.mod (n pid) gs)
  let pn := Expr.div (.mod (n pid) ng) gs
  let offm := fun i => Expr.add (.mul pm (n bm)) (n i)
  let offn := fun j => Expr.add (.mul pn (n bn)) (n j)
  let im := fun i => if second then Expr.mod (offm i) (n a.M) else offm i
  let jn := fun j => if second then Expr.mod (offn j) (n a.N) else offn j
  let ek := fun e => if second then Expr.add (.mul (n 0) (n 32)) (n e) else n e
  let stepk := fun t => if second then Expr.mul (.mul (n t) (n 32)) (n 1) else Expr.mul (n t) (n 32)
  let ki := fun t e => Expr.add (ek e) (stepk t)
  let ai := if second then Expr.mul (.mul (n 32) (n 1)) (n a.a.columnStride) else Expr.mul (n 32) (n a.a.columnStride)
  let bi := if second then Expr.div (.mul (.mul (n 32) (n 1)) (n a.q.rowStride)) (n 8)
    else Expr.mul (.div (n 32) (n 8)) (n a.q.rowStride)
  let pointerRoots := (List.range ((a.K+31)/32+1)).flatMap fun t =>
    ((List.range bm).flatMap fun i => (List.range 32).flatMap fun e =>
      [im i,ek e,ki t e,.add (coord (im i) (ek e) a.a) (.mul (n t) ai)]) ++
    ((List.range 32).flatMap fun e => (List.range bn).flatMap fun j =>
      [jn j,ki t e,.add (coord (.div (ek e) (n 8)) (jn j) a.q) (.mul (n t) bi)])
  let scaleRoots := (List.range ((a.K+31)/32)).flatMap fun t =>
    (List.range 32).flatMap fun e => (List.range bn).flatMap fun j =>
      let group := if second then Expr.div (ki t e) (n a.group)
        else if a.group=a.K then n 0 else Expr.div (n t) (.div (n a.group) (n 32))
      [group,coord group (jn j) a.s,coord group (.div (jn j) (n 8)) a.z,
       .mul (.mod (ek e) (n 8)) (n 4),.mul (.mod (jn j) (n 8)) (n 4)]
  let initialScales := (List.range bn).flatMap fun j =>
    [Expr.mul (jn j) (n a.s.columnStride),.mul (.div (jn j) (n 8)) (n a.z.columnStride)]
  let outputs := (List.range bm).flatMap fun i => (List.range bn).flatMap fun j =>
    [offm i,offn j,coord (offm i) (offn j) a.c]
  [n pid,n a.M,n a.N,n a.K,n a.group,n bm,n bn,n 32,n 8,n 4,n 15,n 1,n 0,
   nm,nn,nk,ng,gid,first,gs,pm,pn,ai,bi,cdiv (n a.K) (.mul (n 32) (n 1)),
   Expr.mul (n (a.K/8)) (n 8)]++pointerRoots++scaleRoots++initialScales++outputs

def dequantRoots (a : Host.Arguments) (pid : Nat) : List Expr :=
  let g := DequantGrid32.geometry (Host.dequant32 a)
  let kp := Geometry.blockRow g pid
  let np := Geometry.blockColumn g pid
  [n a.K,n a.N,n a.group,n kp,n np,cdiv (n a.K) (n 64),cdiv (n a.N) (n 32),
   .mul (n (a.K/8)) (n 8)] ++
  (List.range 64).flatMap fun i => (List.range 32).flatMap fun j =>
    let k := Expr.add (.mul (n kp) (n 64)) (n i)
    let c := Expr.add (.mul (n np) (n 32)) (n j)
    [k,c,coord k c a.d,coord (.div k (n 8)) c a.q,
     coord (.div k (n a.group)) (.div c (n 8)) a.z,
     coord (.div k (n a.group)) c a.s,
     .mul (.mod k (n 8)) (n 4),.mul (.mod c (n 8)) (n 4)]

def count (a : Host.Arguments) (route : Host.Route) (cfg : Host.FirstConfig) : Nat :=
  if route=.gptq then Geometry.count (FirstGrid.geometry (Host.first a cfg))
  else if route=.s2 then Geometry.count (SecondGrid.geometry (Host.second a))
  else Geometry.count (DequantGrid32.geometry (Host.dequant32 a))
def roots (a : Host.Arguments) (route : Host.Route) (cfg : Host.FirstConfig) (pid : Nat) : List Expr :=
  if route=.gptq then matmulRoots a cfg Bool.false pid
  else if route=.s2 then matmulRoots a cfg Bool.true pid else dequantRoots a pid

def Expr.nodes : Expr → List Expr
  | e@(.value _) => [e]
  | e@(.add a b) | e@(.sub a b) | e@(.mul a b) | e@(.min a b)
    | e@(.div a b) | e@(.mod a b) => e::(a.nodes++b.nodes)
def expressions (a : Host.Arguments) (route : Host.Route) (cfg : Host.FirstConfig) (pid : Nat) : List Expr :=
  (roots a route cfg pid).flatMap Expr.nodes

structure Access where
  tensor : Host.Tensor
  offset : Nat
  active : Bool

def matmulAccesses (a : Host.Arguments) (cfg : Host.FirstConfig) (second : Bool) (pid : Nat) : List Access :=
  let g := if second then SecondGrid.geometry (Host.second a) else FirstGrid.geometry (Host.first a cfg)
  let bm := g.blockRows
  let bn := g.blockColumns
  let rm := Geometry.blockRow g pid*bm
  let cn := Geometry.blockColumn g pid*bn
  let im := fun i => if second then (rm+i)%a.M else rm+i
  let jn := fun j => if second then (cn+j)%a.N else cn+j
  let reads := (List.range ((a.K+31)/32)).flatMap fun t =>
    ((List.range bm).flatMap fun i => (List.range 32).map fun e =>
      ⟨a.a,im i*a.a.rowStride+e*a.a.columnStride+t*(32*a.a.columnStride),
       decide (if second then t*32+e<a.K else rm+i<a.M)⟩) ++
    ((List.range 32).flatMap fun e => (List.range bn).flatMap fun j =>
      let group := if second then (e+t*32)/a.group else if a.group=a.K then 0 else t/(a.group/32)
      let active := decide (if second then t*32+e<a.K else True)
      [⟨a.q,e/8*a.q.rowStride+jn j*a.q.columnStride+
          t*(if second then 32*a.q.rowStride/8 else 32/8*a.q.rowStride),active⟩,
       ⟨a.s,group*a.s.rowStride+jn j*a.s.columnStride,active⟩,
       ⟨a.z,group*a.z.rowStride+jn j/8*a.z.columnStride,active⟩])
  let writes := (List.range bm).flatMap fun i => (List.range bn).map fun j =>
    ⟨a.c,(rm+i)*a.c.rowStride+(cn+j)*a.c.columnStride,decide (rm+i<a.M ∧ cn+j<a.N)⟩
  reads++writes

def dequantAccesses (a : Host.Arguments) (pid : Nat) : List Access :=
  let g := DequantGrid32.geometry (Host.dequant32 a)
  (List.range 64).flatMap fun i => (List.range 32).flatMap fun j =>
    let k := Geometry.blockRow g pid*64+i
    let n := Geometry.blockColumn g pid*32+j
    let active := decide (k<a.K ∧ n<a.N)
    [⟨a.q,k/8*a.q.rowStride+n*a.q.columnStride,active⟩,
     ⟨a.z,k/a.group*a.z.rowStride+n/8*a.z.columnStride,active⟩,
     ⟨a.s,k/a.group*a.s.rowStride+n*a.s.columnStride,active⟩,
     ⟨a.d,k*a.d.rowStride+n*a.d.columnStride,active⟩]
def accesses (a : Host.Arguments) (route : Host.Route) (cfg : Host.FirstConfig) (pid : Nat) : List Access :=
  if route=.gptq then matmulAccesses a cfg Bool.false pid
  else if route=.s2 then matmulAccesses a cfg Bool.true pid else dequantAccesses a pid

/-- Actual source arithmetic, including all inactive lanes and the final
pointer updates. No grid-product, execution or numerical-output premise. -/
structure IndexABI (a : Host.Arguments) (route : Host.Route) (cfg : Host.FirstConfig) : Prop where
  formed : ∀ pid<count a route cfg, ∀ e∈expressions a route cfg pid, e.Bounded
  allocated : ∀ pid<count a route cfg, ∀ x∈accesses a route cfg pid, x.active →
    x.tensor.storageOffset+x.offset<x.tensor.allocation
  pointerRoom : ∀ t∈Host.tensors a, ∀ x : Int, Fits32 x →
    0≤(t.base : Int)+Host.bytes t*((t.storageOffset : Int)+x) ∧
    (t.base : Int)+Host.bytes t*((t.storageOffset : Int)+x)<2^64
structure SourceExact (a : Host.Arguments) (route : Host.Route) (cfg : Host.FirstConfig) : Prop where
  operands : ∀ pid<count a route cfg, ∀ e∈expressions a route cfg pid, e.bits.toInt=e.int
  activeAllocation : ∀ pid<count a route cfg, ∀ x∈accesses a route cfg pid, x.active →
    x.tensor.storageOffset+x.offset<x.tensor.allocation
  masks : ∀ pid<count a route cfg, ∀ e∈expressions a route cfg pid,
    ∀ f∈expressions a route cfg pid, e.bits.slt f.bits=decide (e.int<f.int)
  pointers : ∀ t∈Host.tensors a, ∀ x : Int, Fits32 x →
    (BitVec.ofNat 64 t.base+BitVec.ofNat 64 (Host.bytes t)*
      (BitVec.ofNat 64 t.storageOffset+BitVec.ofInt 64 x)).toNat=
      ((t.base : Int)+Host.bytes t*((t.storageOffset : Int)+x)).toNat
  signedWords : ∀ w : BitVec 32, ∀ slot,
    VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.nibble w.toInt slot=
      ((w >>> (slot%8*4)) &&& (15 : BitVec 32)).toInt

theorem source_exact (a : Host.Arguments) (route : Host.Route) (cfg : Host.FirstConfig)
    (abi : IndexABI a route cfg) : SourceExact a route cfg := by
  refine ⟨?_,abi.allocated,?_,?_,?_⟩
  · intro pid hp e he; exact (expression_exact e (abi.formed pid hp e he)).2
  · intro pid hp e he f hf; exact mask_exact e f (abi.formed pid hp e he) (abi.formed pid hp f hf)
  · intro t ht x hx
    have room := abi.pointerRoom t ht x hx
    change (BitVec.ofInt 64 t.base+BitVec.ofInt 64 (Host.bytes t)*
      (BitVec.ofInt 64 t.storageOffset+BitVec.ofInt 64 x)).toNat=_
    rw [←BitVec.ofInt_add,←BitVec.ofInt_mul,←BitVec.ofInt_add,BitVec.toNat_ofInt]
    congr 1
    exact Int.emod_eq_of_lt room.1 room.2
  · intro w slot
    exact VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.nibble_packed_word w slot

theorem nibble_range (w : BitVec 32) (slot : Nat) :
    0≤VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.nibble w.toInt slot ∧
    VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.nibble w.toInt slot≤15 := by
  exact VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPBits.mask_bound _

theorem difference_exact (q z : BitVec 32) (k n : Nat) :
    let d := VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.nibble q.toInt k-
      VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.nibble z.toInt n
    (BitVec.ofInt 32 d).toInt=d ∧ -15≤d ∧ d≤15 := by
  dsimp
  have hq := nibble_range q k
  have hz := nibble_range z n
  refine ⟨BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega),by omega,by omega⟩

theorem rejects_masked_stride_overflow : ¬(Expr.mul (n 64) (n (2^26))).Bounded := by
  norm_num [Expr.Bounded,Expr.int,Fits32,n]
#axiomsClean expression_exact
#axiomsClean source_exact
#axiomsClean difference_exact
#axiomsClean rejects_masked_stride_overflow
end
end VeriTile.Bench.Repairs.MatmulDequantizeWhole001.ABI
