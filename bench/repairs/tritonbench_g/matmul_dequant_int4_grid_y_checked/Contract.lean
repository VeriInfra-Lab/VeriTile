import bench.repairs.tritonbench_g.matmul_dequant_int4.Contract

namespace VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked
open MatmulDequantInt4
open MatmulDequantInt4.Direct
noncomputable section
open Classical

/-- All standalone source configurations have BN=32 or BN=64. -/
def Axis (p : PackedInfo) : Prop := ceil p.weight.tensor.columns 32≤65535

def PackedContract (p : PackedInfo) (m : Nat) (dtype : Host.DType) : Prop :=
  PackedAccepted p m dtype ∧ Axis p

inductive Error where
  | original : Direct.Error → Error
  | gridY
  deriving DecidableEq, Repr

/-- Same successful domain as the original guard sequence, followed by exactly
its one new grid-y branch. Existing failures occur before this new branch. -/
def checkPacked (p : PackedInfo) (m : Nat) (dtype : Host.DType) : Except Error Unit :=
  match Direct.checkPacked p m dtype with
  | .error e => .error (.original e)
  | .ok _ => if Axis p then .ok () else .error .gridY

theorem checkPacked_iff (p : PackedInfo) (m : Nat) (dtype : Host.DType) :
    checkPacked p m dtype=.ok () ↔ PackedContract p m dtype := by
  unfold checkPacked PackedContract
  rw [←Direct.checkPacked_iff]
  cases h : Direct.checkPacked p m dtype with
  | error e => simp [h]
  | ok u => cases u; simp [h,ite_eq_iff]

def checkDequant (p : PackedInfo) (dtype : Host.DType) (device : Option Nat) : Except Error Unit :=
  match checkPacked p 1 dtype with
  | .error e => .error e
  | .ok _ => if device=some p.weight.tensor.device then .ok () else .error (.original .device)

def DequantContract (p : PackedInfo) (dtype : Host.DType) (device : Option Nat) : Prop :=
  PackedContract p 1 dtype ∧ device=some p.weight.tensor.device

theorem checkDequant_iff (p : PackedInfo) (dtype : Host.DType) (device : Option Nat) :
    checkDequant p dtype device=.ok () ↔ DequantContract p dtype device := by
  unfold checkDequant DequantContract
  rw [←checkPacked_iff]
  cases h : checkPacked p 1 dtype with
  | error e => simp [h]
  | ok u => cases u; simp [h,ite_eq_iff]

def S1Contract (p : S1Info) : Prop := S1Accepted p ∧ Axis p.packedInfo

theorem packed_domain_exact (p : PackedInfo) (m : Nat) (dtype : Host.DType) :
    PackedContract p m dtype ↔ PackedAccepted p m dtype ∧
      (∀ c∈Source.dequantConfigs,ceil p.weight.tensor.columns c.bn≤65535) := by
  have alternatives : ∀ c∈Source.dequantConfigs,c.bn=32 ∨ c.bn=64 := by
    intro c hc
    simp only [Source.dequantConfigs,List.mem_cons,List.not_mem_nil,or_false] at hc
    rcases hc with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> decide
  have axis_iff : Axis p ↔ ∀ c∈Source.dequantConfigs,ceil p.weight.tensor.columns c.bn≤65535 := by
    constructor
    · intro h c hc
      rcases alternatives c hc with hn|hn
      · simpa [hn] using h
      · rw [hn]
        dsimp [Axis,ceil] at h ⊢
        omega
    · intro h
      exact h ⟨64,32,4,4⟩ (by simp [Source.dequantConfigs])
  exact and_congr_right (fun _ => axis_iff)

theorem axis_exact_bound (p : PackedInfo) : Axis p ↔ p.weight.tensor.columns≤2097120 := by
  dsimp [Axis,ceil]
  omega

/-- Empty shapes were already rejected by _ordinary; no new empty case is
introduced and the old empty domain is preserved exactly. -/
theorem empty_unchanged (p : PackedInfo) (m : Nat) (dtype : Host.DType)
    (empty : p.weight.tensor.rows=0 ∨ p.weight.tensor.columns=0) :
    ¬PackedAccepted p m dtype ∧ ¬PackedContract p m dtype := by
  have original : ¬PackedAccepted p m dtype := by
    intro h
    have k := h.2.1.shape.kpos
    have n := h.2.1.shape.npos
    rcases empty with empty|empty <;> simp [Direct.packed,empty] at k n
  exact ⟨original,fun h => original h.1⟩

/-- No new product or x-axis guard is needed. Existing K<2^31 implies the
positive architectural x bound; y is the only new admission restriction. -/
theorem launch_axes (p : PackedInfo) (m : Nat) (dtype : Host.DType)
    (h : PackedContract p m dtype) (c : Source.DequantConfig) (hc : c∈Source.dequantConfigs) :
    0<ceil (p.weight.tensor.rows*8) c.bk ∧ ceil (p.weight.tensor.rows*8) c.bk≤2^31-1 ∧
    0<ceil p.weight.tensor.columns c.bn ∧ ceil p.weight.tensor.columns c.bn≤65535 := by
  have shape := h.1.2.1.shape
  have limits := h.1.2.1.indices
  have valid := Source.dequant_valid c hc
  have kb : p.weight.tensor.rows*8<2^31 := limits.k
  have casesK : c.bk=32 ∨ c.bk=64 := by
    simp only [Source.dequantConfigs,List.mem_cons,List.not_mem_nil,or_false] at hc
    rcases hc with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> decide
  exact ⟨ceil_positive _ _ shape.kpos valid.1,by
    rcases casesK with hk|hk <;> rw [hk] <;> dsimp [ceil] <;> omega,
    ceil_positive _ _ shape.npos valid.2.1,((packed_domain_exact p m dtype).mp h).2 c hc⟩

#axiomsClean checkPacked_iff
#axiomsClean checkDequant_iff
#axiomsClean packed_domain_exact
#axiomsClean axis_exact_bound
#axiomsClean empty_unchanged
#axiomsClean launch_axes
end
end VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked
