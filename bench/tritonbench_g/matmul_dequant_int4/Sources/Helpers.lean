import bench.tritonbench_g.matmul_dequant_int4.Sources.Packing
import bench.tritonbench_g.matmul_dequant_int4.Sources.Bounds
namespace VeriTile.Bench.Repairs.MatmulDequantInt4.Helpers
set_option maxHeartbeats 5000000

/-- Scalar Torch arithmetic ports. Max/min reductions are built from scalar nodes;
there is no quantizer, reduction, packed result or matrix output oracle. -/
structure Arithmetic (α : Type) where
  zero : α
  /-- Source Tensor / Python scalar 15; separate from Tensor/Tensor division. -/
  scaleDivide15 : α → α
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
  a.scaleDivide15 (a.sub bounds.1 bounds.2)
def zeroNibble (s : Shape) (a : Arithmetic α) (order : ReductionOrder s.group)
    (initial : State α) (g n : Nat) :=
  a.roundClamp (a.div (a.neg (groupStats s a order initial g n).2) (scale s a order initial g n))
def weightNibble (s : Shape) (a : Arithmetic α) (order : ReductionOrder s.group)
    (initial : State α) (k n : Nat) :=
  a.roundClamp (a.add (a.div (initial.values .weight (k*s.n+n)) (scale s a order initial (k/s.group) n))
    (a.fromInt ((zeroNibble s a order initial (k/s.group) n).toNat : Int)))

/-- Independent packed layout objective, indexed by output matrix coordinates. -/
@[standalone_spec] def weightSpec (s : Shape) (a : Arithmetic α) (order : ReductionOrder s.group)
    (initial : State α) (p n : Nat) :=
  packedSpec (fun i => weightNibble s a order initial (p*8+i) n)
@[standalone_spec] def zeroSpec (s : Shape) (a : Arithmetic α) (order : ReductionOrder s.group)
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


#axiomsClean packed_weight
#axiomsClean packed_zero
#axiomsClean every_quantized_nibble
#axiomsClean every_zero_nibble
#axiomsClean quantize_result
#axiomsClean quantize_scales
#axiomsClean quantize_zeros
end VeriTile.Bench.Repairs.MatmulDequantInt4.Helpers
