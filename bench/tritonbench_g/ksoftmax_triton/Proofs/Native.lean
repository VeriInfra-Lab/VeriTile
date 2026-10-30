import bench.tritonbench_g.var_len_copy.Proofs.Index64
import Mathlib.Tactic
import Init.Data.Nat.Power2.Basic

namespace VeriTile.Bench.Repairs.Ksoftmax.Native
open VeriTile.Bench.Repairs

/-- Every output coordinate has a unique program/lane owner. -/
def flat (Q K batch row lane : Nat) : Nat := batch*(Q*K)+row*K+lane

theorem row_bound {B Q batch row : Nat} (hb : batch<B) (hr : row<Q) :
    batch*Q+row<B*Q := by
  have h:=Nat.mul_le_mul_right Q hb
  rw [Nat.succ_mul] at h
  omega

theorem active_bound {B Q K batch row lane : Nat} (hb : batch<B) (hr : row<Q)
    (hk : lane<K) : flat Q K batch row lane<B*Q*K := by
  have h:=Nat.mul_le_mul_right K (row_bound hb hr)
  rw [Nat.succ_mul] at h
  unfold flat
  nlinarith

theorem padded_bound {B Q K D batch row lane : Nat} (hb : batch<B) (hr : row<Q)
    (hd : D≤2*K) (hk : lane<D) : flat Q K batch row lane<B*Q*K+K := by
  have h:=Nat.mul_le_mul_right K (row_bound hb hr)
  rw [Nat.succ_mul] at h
  unfold flat
  nlinarith

/-- The ordinary next-power-of-two library contract: the returned positive
power covers K, and halving it would not cover a positive K (except D=1). -/
structure DepthContract (K D : Nat) : Prop where
  positive : 0<D
  covers : K≤D
  minimal : D=1 ∨ D<2*K

theorem depth_bounds (K D : Nat) (hK : 0<K) (library : DepthContract K D) :
    0<D ∧ K≤D ∧ D≤2*K := by
  refine ⟨library.positive,library.covers,?_⟩
  rcases library.minimal with h|h <;> omega

/-- Signed-i64 arithmetic as used by the widened source expressions. -/
def mul (a b : Int) : Int := VarLenCopy.signed64 (a*b)
def add (a b : Int) : Int := VarLenCopy.signed64 (a+b)
def offset (Q K batch row lane : Nat) : Int :=
  add (add (mul batch (Q*K)) (mul row K)) lane

theorem offset_exact (B Q K D batch row lane : Nat)
    (hb : batch<B) (hr : row<Q) (hl : lane<D) (depth : D≤2*K)
    (allocation : B*Q*K<2^61) (hK : 0<K) :
    offset Q K batch row lane=(flat Q K batch row lane : Int) := by
  have total : K≤B*Q*K := by
    have bb : 1≤B := by omega
    have qq : 1≤Q := by omega
    have one : 1≤B*Q := by nlinarith
    simpa using Nat.mul_le_mul_right K one
  have bound := padded_bound hb hr depth hl
  have large : flat Q K batch row lane<2^62 := by omega
  have first : batch*(Q*K)<2^63 := by unfold flat at large;omega
  have second : row*K<2^63 := by unfold flat at large;omega
  have middle : batch*(Q*K)+row*K<2^63 := by unfold flat at large;omega
  have last : flat Q K batch row lane<2^63 := by omega
  have castExact (x : Nat) (bound : x<2^63) : VarLenCopy.signed64 (x : Int)=(x : Int) :=
    VarLenCopy.signed64_exact _ (by omega) (by exact_mod_cast bound)
  have p1 := castExact (batch*(Q*K)) first
  have p2 := castExact (row*K) second
  have p3 := castExact (batch*(Q*K)+row*K) middle
  have p4 := castExact (flat Q K batch row lane) last
  simp only [Nat.cast_mul,Nat.cast_add,flat] at p1 p2 p3 p4
  simp only [offset,mul,add,p1,p2,p3,p4,flat,Nat.cast_mul,Nat.cast_add]

/-- The FP32 allocator's signed byte-capacity limit implies the arithmetic
bound. It is not an extra native-safe size restriction on user tensors. -/
theorem allocation_count_bound (count capacity : Nat)
    (fits : 4*count≤capacity) (allocator : capacity<2^63) : count<2^61 := by omega

def pointer (base : Nat) (i : Int) : BitVec 64 := BitVec.ofInt 64 ((base : Int)+4*i)

theorem pointer_modulo (base i : Nat) :
    (pointer base (i : Int)).toNat=(base+4*i)%2^64 := by
  unfold pointer
  have h : (base : Int)+4*(i : Int)=((base+4*i : Nat) : Int) := by push_cast;ring
  rw [h]
  rfl

/-- Every byte of an active FP32 access fits the same supplied allocation.
Inactive formed pointers are kept modular and never assumed dereferenced. -/
theorem active_pointer (base capacity count i : Nat) (fits : 4*count≤capacity)
    (allocation : base+capacity≤2^64) (active : i<count) :
    (pointer base (i : Int)).toNat=base+4*i ∧
    base≤(pointer base (i : Int)).toNat ∧
    (pointer base (i : Int)).toNat+4≤base+capacity := by
  rw [pointer_modulo,Nat.mod_eq_of_lt (by omega : base+4*i<2^64)]
  exact ⟨rfl,by omega,by omega⟩

/-- Exact sign extension of the nonnegative i32 program-ID/arange primitive.
The primitive's representable range is retained explicitly; launch-grid
compatibility is a separately deferred runtime question. -/
def widened32 (n : Nat) : Int := VarLenCopy.signed64 (BitVec.ofInt 32 (n : Int)).toInt

theorem widened32_exact (n : Nat) (primitive : n<2^31) : widened32 n=(n : Int) := by
  unfold widened32
  rw [BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by exact_mod_cast primitive)]
  exact VarLenCopy.signed64_exact _ (by omega) (by exact_mod_cast (show n<2^63 by omega))

def readMask (K row lane : Int) (causal : Bool) : Bool :=
  decide (lane<K) && (!causal || decide (lane≤row))

theorem mask_exact (K row lane : Nat) (causal : Bool) :
    readMask (K : Int) (row : Int) (lane : Int) causal=
      (decide (lane<K) && (!causal || decide (lane≤row))) := by simp [readMask]

/-- The corrected mask stride is K for the checked contiguous mask, independently
of whether mask rows are indexed by query or by batch. -/
theorem mask_address_bound (rows K row lane : Nat) (hr : row<rows) (hl : lane<K) :
    row*K+lane<rows*K := by
  have h:=Nat.mul_le_mul_right K hr
  rw [Nat.succ_mul] at h
  omega

/-- Different logical output coordinates cannot race, even across programs. -/
theorem flat_injective (Q K b r k b' r' k' : Nat) (hr : r<Q) (hr' : r'<Q)
    (hk : k<K) (hk' : k'<K) (same : flat Q K b r k=flat Q K b' r' k') :
    b=b' ∧ r=r' ∧ k=k' := by
  have lanes:=congrArg (fun n => n%K) same
  simp [flat,Nat.add_mod,Nat.mul_mod,Nat.mod_eq_of_lt hk,Nat.mod_eq_of_lt hk'] at lanes
  have bases : (b*Q+r)*K=(b'*Q+r')*K := by unfold flat at same;nlinarith
  have rows : b*Q+r=b'*Q+r' := Nat.eq_of_mul_eq_mul_right (by omega : 0<K) bases
  have coordinates:=congrArg (fun n => n%Q) rows
  simp [Nat.add_mod,Nat.mul_mod,Nat.mod_eq_of_lt hr,Nat.mod_eq_of_lt hr'] at coordinates
  have batches : b*Q=b'*Q := by omega
  exact ⟨Nat.eq_of_mul_eq_mul_right (by omega : 0<Q) batches,coordinates,lanes⟩

#axiomsClean flat_injective
#axiomsClean widened32_exact
#axiomsClean mask_exact
#axiomsClean mask_address_bound
#axiomsClean row_bound
#axiomsClean active_bound
#axiomsClean padded_bound
#axiomsClean depth_bounds
#axiomsClean offset_exact
#axiomsClean allocation_count_bound
#axiomsClean pointer_modulo
#axiomsClean active_pointer
end VeriTile.Bench.Repairs.Ksoftmax.Native
