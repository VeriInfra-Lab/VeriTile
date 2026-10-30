import bench.tritonbench_g.bmm_optimized.Sources.Contract

namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked
open BmmOptimized

/-- The wrapper launches for a nonempty output, including a zero contraction. -/
def Live (s : Shape) : Prop := 0 < s.batch ∧ 0 < s.m ∧ 0 < s.n
instance (s : Shape) : Decidable (Live s) := inferInstanceAs (Decidable (_ ∧ _ ∧ _))

/-- The exact additional domain restriction. TILE_N=32 occurs in the unchanged
autotune list and is its smallest N tile. Empty returns bypass this check. -/
def AxisChecks (s : Shape) : Prop := Live s → s.batch ≤ 65535 ∧ blocks s.n 32 ≤ 65535
instance (s : Shape) : Decidable (AxisChecks s) := inferInstanceAs (Decidable (_ → _ ∧ _))

structure InputContract (p : Checked.Metadata) : Prop where
  original : Checked.InputContract p
  axes : AxisChecks p.call.shape

def checkInputs (p : Checked.Metadata) : Except String Unit :=
  match Checked.checkInputs p with
  | .error _ => .error "original input checks"
  | .ok _ => if AxisChecks p.call.shape then .ok () else .error "live grid axes"

theorem checkInputs_iff (p : Checked.Metadata) :
    checkInputs p = .ok () ↔ InputContract p := by
  have contract : InputContract p ↔ Checked.InputContract p ∧ AxisChecks p.call.shape :=
    ⟨fun h => ⟨h.original,h.axes⟩,fun h => ⟨h.1,h.2⟩⟩
  rw [contract, ← Checked.checkInputs_iff]
  cases h : Checked.checkInputs p <;> simp [checkInputs,h]

theorem empty_domain_unchanged (p : Checked.Metadata) (empty : ¬Live p.call.shape) :
    InputContract p ↔ Checked.InputContract p := by
  constructor
  · exact fun h => h.original
  · exact fun h => ⟨h,fun live => False.elim (empty live)⟩

theorem n_bound_iff (n : Nat) : blocks n 32 ≤ 65535 ↔ n ≤ 2097120 := by
  unfold blocks
  omega

theorem grid_y_le (s : Shape) (c : Config) : s.gy c ≤ blocks s.n 32 := by
  rcases c with ⟨t,st⟩
  cases t <;> simp only [Shape.gy,Config.tn,blocks] <;> omega

theorem axes_exact (s : Shape) : AxisChecks s ↔
    (Live s → ∀ c : Config, s.batch ≤ 65535 ∧ s.gy c ≤ 65535) := by
  constructor
  · intro h live c
    exact ⟨(h live).1,(grid_y_le s c).trans (h live).2⟩
  · intro h live
    exact h live ⟨.t32x32,Bool.false⟩

theorem launch_axes (p : Checked.Metadata) (contract : InputContract p)
    (live : Live p.call.shape) (c : Config) :
    p.call.shape.batch ≤ 65535 ∧ p.call.shape.gy c ≤ 65535 :=
  (axes_exact _).1 contract.axes live c

theorem old_z_witness_rejected : ¬AxisChecks ⟨65536,1,1,1⟩ := by
  norm_num [AxisChecks,Live,blocks]

theorem old_y_witness_rejected : ¬AxisChecks ⟨1,1,8388481,1⟩ := by
  norm_num [AxisChecks,Live,blocks]

#axiomsClean checkInputs_iff
#axiomsClean empty_domain_unchanged
#axiomsClean axes_exact
#axiomsClean launch_axes
#axiomsClean old_z_witness_rejected
#axiomsClean old_y_witness_rejected
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked
