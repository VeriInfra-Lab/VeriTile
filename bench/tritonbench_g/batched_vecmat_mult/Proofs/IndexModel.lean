import bench.tritonbench_g.batched_vecmat_mult.Kernels

namespace VeriTile.Bench.Repairs.BatchedVecmatMult
namespace Index64
abbrev word (x : Nat) : BitVec 64 := BitVec.ofNat 64 x

def coordinate (pid block lane : Nat) : BitVec 64 := word pid * word block + word lane

def outOffset (p : Parameters) (pm pn i j : Nat) : BitVec 64 :=
  coordinate pm p.BM i * word p.N + coordinate pn p.BN j

def aOffset (p : Parameters) (pm c i e : Nat) : BitVec 64 :=
  coordinate pm p.BM i * word p.K + coordinate c p.BK e

def bOffset (p : Parameters) (pm pn c i j e : Nat) : BitVec 64 :=
  coordinate pm p.BM i * word p.N * word p.K +
  coordinate pn p.BN j * word p.K + coordinate c p.BK e

def bytePointer (base : Nat) (offset : BitVec 64) : BitVec 64 :=
  word base + offset * word 4

/-- Ordinary representable dense float32 storage, including byte extent. This
is an allocation ABI, not a new caller restriction: Torch rejects storage-size
integer overflow before launch. No i32-native-safe shape condition is imposed.
Zero-sized dimensions remain allowed; setup errors retain the host outcome. -/
structure StorageLimits (p : Parameters) : Prop where
  M : p.M < 2^63
  N : p.N < 2^63
  K : p.K < 2^63
  A : p.M*p.K*4 < 2^63
  B : p.M*p.N*p.K*4 < 2^63
  output : p.M*p.N*4 < 2^63

/-- Agreement of every actual finite-width memory offset with the reused
complete source model. Output is independent of K, including K=0. -/
structure AddressAgreement (p : Parameters) : Prop where
  output : ∀ pm : Fin (p.M/p.BM), ∀ pn : Fin (p.N/p.BN),
    ∀ i : Fin p.BM, ∀ j : Fin p.BN,
    (outOffset p pm.val pn.val i.val j.val).toInt =
      (((pm.val*p.BM+i.val)*p.N+(pn.val*p.BN+j.val) : Nat) : Int)
  inputs : ∀ pm : Fin (p.M/p.BM), ∀ pn : Fin (p.N/p.BN), ∀ c : Fin (p.K/p.BK),
    ∀ i : Fin p.BM, ∀ j : Fin p.BN, ∀ e : Fin p.BK,
    (aOffset p pm.val c.val i.val e.val).toInt =
      (((pm.val*p.BM+i.val)*p.K+(c.val*p.BK+e.val) : Nat) : Int) ∧
    (bOffset p pm.val pn.val c.val i.val j.val e.val).toInt =
      (((pm.val*p.BM+i.val)*p.N*p.K+(pn.val*p.BN+j.val)*p.K+(c.val*p.BK+e.val) : Nat) : Int)

end Index64
end VeriTile.Bench.Repairs.BatchedVecmatMult
