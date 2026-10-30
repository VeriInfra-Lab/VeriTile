import bench.tritonbench_g.attention_fwd_triton3.Proofs.Contract
import bench.tritonbench_g.attention_fwd_triton3.Kernels

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Contract
open VeriTile Triton
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open Host Checked

/-- Only the CUDA y-axis admission changes. Original metadata, live storage,
strides, six-region separation and formed-address checks are retained. -/
def check (c : Call) (A : FlatAlloc) (p : Pointers A) (args : Arguments) : Bool :=
  metadataChecks c args && setupChecks c A p && decide (c.Z*c.H≤65535)

structure Admission (c : Call) (A : FlatAlloc) (p : Pointers A) (args : Arguments) : Prop where
  metadata : metadataChecks c args=Bool.true
  setup : setupChecks c A p=Bool.true
  grid_y : c.Z*c.H≤65535

theorem check_iff (c : Call) (A : FlatAlloc) (p : Pointers A) (args : Arguments) :
    check c A p args=Bool.true ↔ Admission c A p args := by
  simp only [check,Bool.and_eq_true,decide_eq_true_eq]
  constructor
  · rintro ⟨⟨hm,hs⟩,hy⟩;exact ⟨hm,hs,hy⟩
  · rintro ⟨hm,hs,hy⟩;exact ⟨⟨hm,hs⟩,hy⟩

theorem safety (h : Admission c A p args) : Safety c A p := admitted c A p h.setup

/-- Both constexpr and runtime NK use the same representable signed64 scalar.
There is no NK=1 exclusion or extra numerical hypothesis. -/
theorem scalar_cast (n : Nat) (bound : n≤2^31-65) :
    (BitVec.ofNat 64 n).toInt=(n : Int) := by
  have hn : (n : Int)<2^63 := by exact_mod_cast (show n<2^63 by omega)
  change (BitVec.ofInt 64 (n : Int)).toInt=(n : Int)
  apply BitVec.toInt_ofInt_eq_self (by decide)
  · norm_num
  · exact hn

theorem scalar_cast_one : (BitVec.ofNat 64 1).toInt=(1 : Int) := by decide

theorem grid_y_boundary (y : Nat) : (decide (y≤65535) : Bool)=Bool.true ↔ y≤65535 := by simp

#axiomsClean check_iff
#axiomsClean safety
#axiomsClean scalar_cast
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Contract
