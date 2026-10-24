import bench.repairs.tritonbench_g.cross_entropy2.Proofs.HostContract
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.Native

namespace VeriTile.Bench.Repairs.CrossEntropy2.Checked
open VeriTile.Triton Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Step)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

/-- Successful checked calls admit precisely the guard predicate and retain
exactly the original successful source computation and returned state. -/
theorem forward_success {α σ : Type} [Inhabited α] {M : FP32 α} {rt : Runtime α σ}
    {mx my : TensorMetadata} {ms : ScalarMetadata} {splitBool : Bool}
    {o : Options} {x y : Tensor} {rows cols : Nat} {target : Nat → Int}
    {initial final : World α σ} {returned : Return} :
    forward M rt mx my ms splitBool o x y rows cols target initial=.ok returned final ↔
      ForwardContract mx my ms splitBool o x y rows cols target ∧ Host.forward M rt o x y initial=.ok returned final := by
  classical
  unfold forward
  split_ifs with h <;> simp [h]

theorem backward_success {α σ : Type} [Inhabited α] {M : FP32 α} {rt : Runtime α σ}
    {mx my mg ml : TensorMetadata} {ms : ScalarMetadata}
    {o : Options} {g x l y : Tensor} {rows cols : Nat} {target : Nat → Int}
    {initial final : World α σ} {returned : Tensor} :
    backward M rt mx my mg ml ms o g x l y rows cols target initial=.ok returned final ↔
      BackwardContract mx my mg ml ms o g x l y rows cols target ∧ Host.backward M rt o g x l y initial=.ok returned final := by
  classical
  unfold backward
  split_ifs with h <;> simp [h]

theorem native_launch {mx my : TensorMetadata} {ms : ScalarMetadata} {o : Options}
    {x y : Tensor} {rows cols : Nat} {target : Nat → Int}
    (h : CommonContract mx my ms o x y rows cols target) :
    rows=0 ∨ CrossEntropy1.Native.Supported rows (CrossEntropy1.Math.blocks cols o.width) o.width cols := by
  by_cases empty : rows=0
  · exact Or.inl empty
  · have limits := h.2.2.2.2.2.1 (by omega)
    have valid := h.2.2.2.2.1
    have widthPos : 0<o.width := by unfold Host.valid at valid; tauto
    have widthLim : o.width≤65536 := by unfold Host.valid at valid; tauto
    have metadata : o.start+cols≤o.total := by unfold Host.valid at valid; tauto
    have total : o.total<2^63 := by unfold Host.valid at valid; tauto
    exact Or.inr ⟨limits.1,limits.2,widthPos,widthLim,by omega⟩

#axiomsClean forward_success
#axiomsClean backward_success
#axiomsClean native_launch
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Checked
