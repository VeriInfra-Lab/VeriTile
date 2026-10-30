import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Admission

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked
open AttentionFwdTriton1.Whole001
noncomputable section

/-- Exact original input and formed-address checks, plus the sole live tile
condition. No numerical payload predicate or hardware-capacity promise. -/
def InputContract (c : Shape) (args : Host.Arguments) {abi : ABI c}
    (pointers : AttentionFwdTriton1.Checked.Pointers abi) : Prop :=
  AttentionFwdTriton1.Checked.inputChecks c args=Bool.true ∧
  AttentionFwdTriton1.Checked.setupChecks c pointers=Bool.true ∧
  liveTileCheck c=Bool.true

def checkInput (c : Shape) (args : Host.Arguments) {abi : ABI c}
    (pointers : AttentionFwdTriton1.Checked.Pointers abi) : Bool :=
  AttentionFwdTriton1.Checked.inputChecks c args &&
    AttentionFwdTriton1.Checked.setupChecks c pointers && liveTileCheck c

theorem checks_iff (c : Shape) (args : Host.Arguments) {abi : ABI c}
    (pointers : AttentionFwdTriton1.Checked.Pointers abi) :
    checkInput c args pointers=Bool.true ↔ InputContract c args pointers := by
  simp [checkInput,InputContract,and_assoc]

theorem inherited_checks (c : Shape) (args : Host.Arguments) {abi : ABI c}
    (pointers : AttentionFwdTriton1.Checked.Pointers abi) (h : InputContract c args pointers) :
    AttentionFwdTriton1.Checked.inputChecks c args=Bool.true ∧
      AttentionFwdTriton1.Checked.setupChecks c pointers=Bool.true := ⟨h.1,h.2.1⟩

theorem empty_domain_unchanged (c : Shape) (args : Host.Arguments) {abi : ABI c}
    (pointers : AttentionFwdTriton1.Checked.Pointers abi) (empty : Host.Empty c) :
    InputContract c args pointers ↔ AttentionFwdTriton1.Checked.inputChecks c args=Bool.true := by
  simp [InputContract,AttentionFwdTriton1.Checked.setupChecks,
    show c.T=0 ∨ c.heads=0 from empty,empty_unchanged c empty]

theorem static_tile_bound (c : Shape) (args : Host.Arguments) {abi : ABI c}
    (pointers : AttentionFwdTriton1.Checked.Pointers abi) (h : InputContract c args pointers)
    (live : ¬Host.Empty c) : c.D*c.D≤1048576 :=
  (liveTileCheck_iff c).1 h.2.2 live

theorem machine_domain (c : Shape) (args : Host.Arguments) (abi : ABI c)
    (pointers : AttentionFwdTriton1.Checked.Pointers abi) (h : InputContract c args pointers) :
    Machine.Domain c abi := AttentionFwdTriton1.Checked.admitted_machine c abi pointers h.2.1

theorem machine_closed (c : Shape) (args : Host.Arguments) (abi : ABI c)
    (pointers : AttentionFwdTriton1.Checked.Pointers abi) (h : InputContract c args pointers) :
    Machine.Closed c abi := Machine.closed c abi (machine_domain c args abi pointers h)

#axiomsClean checks_iff
#axiomsClean empty_domain_unchanged
#axiomsClean static_tile_bound
#axiomsClean machine_closed
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked
