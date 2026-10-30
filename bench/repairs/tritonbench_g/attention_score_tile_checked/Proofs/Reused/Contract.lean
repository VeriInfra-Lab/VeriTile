import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.CurrentMachine
namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.Contract
open VeriTile Triton Geometry Physical Whole002 Representation
noncomputable section

/-- Classified ordinary metadata. Shapes and supported dimensions are `Shape`;
these flags state same-device contiguous strided CUDA FP32 inputs with resolved
logical views. The pointers include the newly allocated output. -/
structure Observations where
  ordinaryTensors : Bool
  cudaFP32 : Bool
  oneDevice : Bool
  contiguous : Bool
  strided : Bool
  resolvedViews : Bool
  pointer : RegionName  →  Nat

def Metadata (o : Observations) : Prop :=
  o.ordinaryTensors=Bool.true  ∧  o.cudaFP32=Bool.true  ∧  o.oneDevice=Bool.true  ∧ 
  o.contiguous=Bool.true  ∧  o.strided=Bool.true  ∧  o.resolvedViews=Bool.true

def InputContract (c : Shape) (o : Observations) (offset size : Int) (window : Bool) : Prop :=
  Metadata o  ∧  (window=Bool.true  →  Signed32 offset  ∧  Signed32 size)  ∧ 
  (c.N ≠ 0  →  c.outputSize ≠ 0  → 
    (∀ x ∈ metadata c, x  ≤  i32max)  ∧  qPad c+c.B ≤ i32max  ∧  kPad c ≤ i32max  ∧ 
    c.programs ≤ i32max  ∧  c.Z*c.H ≤ 65535  ∧ 
    (∀ r,tracked r  →  4*extent c r ≤ i64max)  ∧ 
    (∀ r,tracked r  →  o.pointer r % 4=0  ∧  o.pointer r+4*extent c r < 18446744073709551616))

def check (c : Shape) (o : Observations) (offset size : Int) (window : Bool) : Bool := by
  classical
  exact decide (InputContract c o offset size window)

theorem check_iff (c : Shape) (o : Observations) (offset size : Int) (window : Bool) :
    check c o offset size window=Bool.true ↔ InputContract c o offset size window := by
  classical
  simp [check]

/-- Physical allocations are word-indexed; the byte pointer is exactly four
 times the word base. Alignment is checked before the launch. -/
def PointerBinding (c : Shape) (abi : ABI c) (o : Observations) : Prop :=
  c.N ≠ 0  →  c.outputSize ≠ 0  →  ∀ r,tracked r  →  4*(abi.alloc r).base=o.pointer r

theorem domain (c : Shape) (abi : ABI c) (o : Observations) (offset size : Int)
    (window : Bool) (h : InputContract c o offset size window) (binding : PointerBinding c abi o) :
    Domain c abi offset size window := by
  refine ⟨h.2.1,?_,?_⟩
  · intro hn ho
    rcases h.2.2 hn ho with ⟨args,hq,hk,hx,hy,hbytes,_⟩
    exact ⟨args,hq,hk,hx,hy,hbytes⟩
  · intro hn ho r hr
    have pointer := (h.2.2 hn ho).2.2.2.2.2.2 r hr
    rw [Nat.mul_add,binding hn ho r hr]
    exact pointer.2

theorem aligned (c : Shape) (o : Observations) (offset size : Int) (window : Bool)
    (h : InputContract c o offset size window) (hn : c.N ≠ 0) (ho : c.outputSize ≠ 0)
    (r : RegionName) (hr : tracked r) (i : Nat) : (o.pointer r+4*i) % 4=0 := by
  have hp:=((h.2.2 hn ho).2.2.2.2.2.2 r hr).1
  omega

/-- Direct pre-call logical input correspondence from contiguous Tensor API
semantics. There is no normalization allocation or copied-input snapshot. -/
abbrev DirectInputs := @Normalized

#specNonCircular InputContract avoiding [FP.Structural.exec]
#axiomsClean check_iff
#axiomsClean domain
#axiomsClean aligned
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.Contract
