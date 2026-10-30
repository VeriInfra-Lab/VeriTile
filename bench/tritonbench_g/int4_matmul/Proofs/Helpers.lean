import bench.tritonbench_g.int4_matmul.Proofs.Packing
import bench.tritonbench_g.int4_matmul.Proofs.Bounds
namespace VeriTile.Bench.Repairs.Int4Matmul.Helpers
set_option maxHeartbeats 5000000

/-- Scalar Torch arithmetic ports. Max/min reductions are built from scalar nodes;
there is no quantizer, reduction, packed result or matrix output oracle. -/
structure Arithmetic (α : Type) where
  zero : α
  fifteen : α
  fromInt : Int → α
  neg : α → α
  add : α → α → α
  sub : α → α → α
  mul : α → α → α
  div : α → α → α
  max : α → α → α
  min : α → α → α
  clipLowerZero : α → α
  clipUpperZero : α → α
  roundClamp : α → BitVec 4

/-- Source finite group reduction order is supplied as an ordinary Tensor
execution order. It contains each element once; scalar max/min are the only nodes. -/
inductive ReductionTree (g : Nat) where
  | leaf (i : Fin g)
  | node (left right : ReductionTree g)
def ReductionTree.leaves : ReductionTree g → List (Fin g)
  | .leaf i => [i]
  | .node l r => l.leaves++r.leaves
def ReductionTree.eval (op : α → α → α) (input : Fin g → α) : ReductionTree g → α
  | .leaf i => input i
  | .node l r => op (l.eval op input) (r.eval op input)
structure ReductionOrder (g : Nat) where
  tree : ReductionTree g
  complete : tree.leaves.Perm (List.finRange g)

/-- Independent logical grouped values from original [K,N] weights. -/
def input (s : Shape) (initial : State α) (g n i : Nat) := initial.values .weight ((g*s.group+i)*s.n+n)
def groupStats (s : Shape) (a : Arithmetic α) (order : ReductionOrder s.group)
    (initial : State α) (g n : Nat) : α × α :=
  let hi := order.tree.eval a.max (fun i => input s initial g n i)
  let lo := order.tree.eval a.min (fun i => input s initial g n i)
  (a.clipLowerZero hi,a.clipUpperZero lo)
def scale (s : Shape) (a : Arithmetic α) (order : ReductionOrder s.group)
    (initial : State α) (g n : Nat) :=
  let bounds := groupStats s a order initial g n
  a.div (a.sub bounds.1 bounds.2) a.fifteen
def zeroNibble (s : Shape) (a : Arithmetic α) (order : ReductionOrder s.group)
    (initial : State α) (g n : Nat) :=
  a.roundClamp (a.div (a.neg (groupStats s a order initial g n).2) (scale s a order initial g n))
def weightNibble (s : Shape) (a : Arithmetic α) (order : ReductionOrder s.group)
    (initial : State α) (k n : Nat) :=
  a.roundClamp (a.add (a.div (initial.values .weight (k*s.n+n)) (scale s a order initial (k/s.group) n))
    (a.fromInt ((zeroNibble s a order initial (k/s.group) n).toNat : Int)))

/-- Independent packed layout objective, indexed by output matrix coordinates. -/
def weightSpec (s : Shape) (a : Arithmetic α) (order : ReductionOrder s.group)
    (initial : State α) (p n : Nat) :=
  packedSpec (fun i => weightNibble s a order initial (p*8+i) n)
def zeroSpec (s : Shape) (a : Arithmetic α) (order : ReductionOrder s.group)
    (initial : State α) (g p : Nat) :=
  packedSpec (fun i => zeroNibble s a order initial g (p*8+i))

/-- Source [N,K] contiguous transpose/view intermediate. -/
def quantizedTemporary (s : Shape) (a : Arithmetic α) (order : ReductionOrder s.group)
    (initial : State α) (address : Nat) :=
  weightNibble s a order initial (address%s.k) (address/s.k)
def sourceWeightWord (s : Shape) (a : Arithmetic α) (order : ReductionOrder s.group)
    (initial : State α) (p n : Nat) :=
  packLoop (fun i => quantizedTemporary s a order initial (n*s.k+(p*8+i))) 8
/-- Zero points pack across columns; their transpose changes both packed axes. -/
def sourceZeroWord (s : Shape) (a : Arithmetic α) (order : ReductionOrder s.group)
    (initial : State α) (g p : Nat) :=
  packLoop (fun i => zeroNibble s a order initial g (p*8+i)) 8

theorem packed_weight (s : Shape) (hs : s.Valid) (a : Arithmetic α) (order : ReductionOrder s.group)
    (initial : State α) (p n : Nat) (hp : p<s.k/8) :
    sourceWeightWord s a order initial p n=weightSpec s a order initial p n := by
  unfold sourceWeightWord weightSpec
  rw [packLoop_correct]
  congr 1
  funext i
  have h : p*8+i.val<s.k := by
    have bound := rowMajor_bound (s.k/8) 8 p i.val hp i.isLt
    simpa [Nat.div_mul_cancel hs.k8] using bound
  simp [quantizedTemporary,Nat.add_mod,Nat.mod_eq_of_lt h,div_coordinate n s.k _ hs.kpos h]

theorem packed_zero (s : Shape) (a : Arithmetic α) (order : ReductionOrder s.group)
    (initial : State α) (g p : Nat) : sourceZeroWord s a order initial g p=zeroSpec s a order initial g p :=
  packLoop_correct _

theorem every_quantized_nibble (s : Shape) (hs : s.Valid) (a : Arithmetic α) (order : ReductionOrder s.group)
    (initial : State α) (p n : Nat) (hp : p<s.k/8) (i : Fin 8) :
    ((sourceWeightWord s a order initial p n >>> (4*i.val)) &&& (15 : Word)).toNat=
      (weightNibble s a order initial (p*8+i) n).toNat := by
  rw [packed_weight s hs a order initial p n hp]
  exact unpack_packed _ i

theorem every_zero_nibble (s : Shape) (a : Arithmetic α) (order : ReductionOrder s.group)
    (initial : State α) (g p : Nat) (i : Fin 8) :
    ((sourceZeroWord s a order initial g p >>> (4*i.val)) &&& (15 : Word)).toNat=
      (zeroNibble s a order initial g (p*8+i)).toNat := by
  rw [packed_zero]
  exact unpack_packed _ i

structure Quantized (α : Type) where
  weights : Nat → Word
  scales : Nat → α
  zeros : Nat → Word
  weightShape : Nat × Nat
  scaleShape : Nat × Nat
  zeroShape : Nat × Nat
  groupSize : Nat

def quantize (s : Shape) (a : Arithmetic α) (order : ReductionOrder s.group) (initial : State α) : Quantized α :=
  ⟨(fun addr => sourceWeightWord s a order initial (addr/s.n) (addr%s.n)),
   (fun addr => scale s a order initial (addr/s.n) (addr%s.n)),
   (fun addr => sourceZeroWord s a order initial (addr/(s.n/8)) (addr%(s.n/8))),
   (s.k/8,s.n),(s.k/s.group,s.n),(s.k/s.group,s.n/8),s.group⟩

theorem quantize_result (s : Shape) (hs : s.Valid) (a : Arithmetic α) (order : ReductionOrder s.group)
    (initial : State α) (p n : Nat) (hp : p<s.k/8) (hn : n<s.n) :
    (quantize s a order initial).weights (p*s.n+n)=weightSpec s a order initial p n := by
  simp only [quantize,div_coordinate p s.n n hs.npos hn]
  simp only [Nat.mul_add_mod_self_right,Nat.mod_eq_of_lt hn]
  exact packed_weight s hs a order initial p n hp

/-- Unpack explicitly stores half nibbles, promotes to FP32 for subtraction
and scaling, then rounds back to the half output. β is the 16-bit carrier. -/
structure HalfArithmetic (α β : Type) where
  fromInt : Int → α
  narrow : α → β
  widen : β → α
  sub : α → α → α
  mul : α → α → α

def unpackSpec (s : Shape) (a : HalfArithmetic α β) (initial : State α) (k n : Nat) : β :=
  let q := nibble (initial.packed .weight (k/8*s.n+n)) k
  let z := nibble (initial.packed .zero (k/s.group*(s.n/8)+n/8)) n
  a.narrow (a.mul (a.sub (a.widen (a.narrow (a.fromInt q))) (a.fromInt z))
    (initial.values .scale (k/s.group*s.n+n)))

/-- Physical half destination is [N,K] before the returned transpose view. -/
def sourceUnpack (s : Shape) (a : HalfArithmetic α β) (initial : State α) (address : Nat) :=
  unpackSpec s a initial (address%s.k) (address/s.k)

theorem unpack_result (s : Shape) (hs : s.Valid) (a : HalfArithmetic α β)
    (initial : State α) (k n : Nat) (hk : k<s.k) :
    sourceUnpack s a initial (n*s.k+k)=unpackSpec s a initial k n := by
  simp [sourceUnpack,div_coordinate n s.k k hs.kpos hk,Nat.add_mod,Nat.mod_eq_of_lt hk]

theorem unpack_address (s : Shape) (k n : Nat) (hk : k<s.k) (hn : n<s.n) :
    n*s.k+k<s.n*s.k := rowMajor_bound _ _ _ _ hn hk

/-- Source unpack first populates every half nibble, then each group slice is
read from current half storage, scaled in FP32, and stored back once. -/
def initialHalf (s : Shape) (a : HalfArithmetic α β) (initial : State α) (k n : Nat) :=
  a.narrow (a.fromInt (nibble (initial.packed .weight (k/8*s.n+n)) k))
def unpackLoop (s : Shape) (a : HalfArithmetic α β) (initial : State α) (k n : Nat) : Nat → β
  | 0 => initialHalf s a initial k n
  | g+1 => if k/s.group=g then
      a.narrow (a.mul (a.sub (a.widen (unpackLoop s a initial k n g))
        (a.fromInt (nibble (initial.packed .zero (g*(s.n/8)+n/8)) n)))
        (initial.values .scale (g*s.n+n)))
    else unpackLoop s a initial k n g

theorem unpack_loop_invariant (s : Shape) (a : HalfArithmetic α β) (initial : State α)
    (k n count : Nat) :
    unpackLoop s a initial k n count=
      if k/s.group<count then unpackSpec s a initial k n else initialHalf s a initial k n := by
  induction count with
  | zero => simp [unpackLoop]
  | succ count ih =>
    simp only [unpackLoop,ih]
    by_cases he : k/s.group=count
    · have hn : ¬k/s.group<count := by omega
      have hy : k/s.group<count+1 := by omega
      simp [he,hn,hy,unpackSpec,initialHalf]
    · by_cases hl : k/s.group<count
      · have hy : k/s.group<count+1 := by omega
        simp [he,hl,hy]
      · have hn : ¬k/s.group<count+1 := by omega
        simp [he,hl,hn]

theorem whole_unpack_loop (s : Shape) (hs : s.Valid) (a : HalfArithmetic α β)
    (initial : State α) (k n : Nat) (hk : k<s.k) :
    unpackLoop s a initial k n (s.k/s.group)=unpackSpec s a initial k n := by
  have hg : k/s.group<s.k/s.group := (Nat.div_lt_iff_lt_mul hs.gpos).mpr
    (by simpa [Nat.div_mul_cancel hs.groups] using hk)
  rw [unpack_loop_invariant,if_pos hg]

theorem quantize_scales (s : Shape) (hs : s.Valid) (a : Arithmetic α) (order : ReductionOrder s.group)
    (initial : State α) (g n : Nat) (hn : n<s.n) :
    (quantize s a order initial).scales (g*s.n+n)=scale s a order initial g n := by
  simp only [quantize,div_coordinate _ _ _ hs.npos hn,Nat.mul_add_mod_self_right,Nat.mod_eq_of_lt hn]

theorem quantize_zeros (s : Shape) (hs : s.Valid) (a : Arithmetic α) (order : ReductionOrder s.group)
    (initial : State α) (g p : Nat) (hp : p<s.n/8) :
    (quantize s a order initial).zeros (g*(s.n/8)+p)=zeroSpec s a order initial g p := by
  have hn : 0<s.n/8 := by omega
  simp only [quantize,div_coordinate _ _ _ hn hp,Nat.mul_add_mod_self_right,Nat.mod_eq_of_lt hp]
  exact packed_zero s a order initial g p

#axiomsClean unpack_loop_invariant
#axiomsClean whole_unpack_loop
#axiomsClean quantize_scales
#axiomsClean quantize_zeros
#axiomsClean packed_weight
#axiomsClean packed_zero
#axiomsClean every_quantized_nibble
#axiomsClean every_zero_nibble
#axiomsClean quantize_result
#axiomsClean unpack_result
#axiomsClean unpack_address
end VeriTile.Bench.Repairs.Int4Matmul.Helpers
