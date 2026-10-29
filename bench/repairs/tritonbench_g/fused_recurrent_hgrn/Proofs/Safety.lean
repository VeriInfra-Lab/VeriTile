import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.Physical
import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.Ownership

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn.Checked
open Physical
noncomputable section

/-- Reverse source nodes; the previous-output subtraction is reached only at
positive i. Initialization, including T=0, is covered by forwardNodes at step=0. -/
def reverseNodes (c : Shape) (BD block head lane step : Nat) : List Nat :=
  let i := c.T-1-step
  forwardNodes c BD block head lane 0 ++
    [step,c.T-1,i,head*c.T+i,(head*c.T+i)*c.D,(head*c.T+i)*c.D+(block*BD+lane)] ++
    (if 0 < i then [head*c.T+i-1,(head*c.T+i-1)*c.D,
      (head*c.T+i-1)*c.D+(block*BD+lane)] else [])

theorem reverse_nodes_bound (c : Shape) (BD block head lane step : Nat)
    (tile : tileCheck BD=Bool.true) (hb : block < (c.D+BD-1)/BD)
    (hh : head < c.BH) (hl : lane < BD) (hs : step < c.T) :
    ∀ x∈reverseNodes c BD block head lane step,x ≤ c.high := by
  have init := forward_nodes_bound c BD block head lane 0 tile hb hh hl (by omega)
  have bounds := reverse_bounds c BD block head lane (c.T-1-step) tile hb hh hl (by omega)
  have shape := shape_bounds c
  have product := Nat.le_mul_of_pos_right (head*c.T+(c.T-1-step)-1) c.dpos
  intro x hx
  simp only [reverseNodes,List.mem_append,List.mem_cons,List.not_mem_nil,or_false] at hx
  rcases hx with (hx|hx)|hx
  · exact init x hx
  · rcases hx with rfl|rfl|rfl|rfl|rfl|rfl <;> omega
  · split at hx
    · simp only [List.mem_cons,List.not_mem_nil,or_false] at hx
      rcases hx with rfl|rfl|rfl <;> omega
    · simp at hx

theorem pointer_bound (c : Shape) (U S V : Bool) (runtime : Runtime c U S V)
    (accepted : check c U S V runtime=Bool.true) (r : Role)
    (active : present U S V r=Bool.true) (a : Nat) (bound : a ≤ c.high) :
    (runtime.tensor r).base+4*a < 2^64 := by
  have facts := accepted
  simp only [check,Bool.and_eq_true] at facts
  have member : r∈roles := by cases r <;> simp [roles]
  have pointer := List.all_eq_true.mp facts.2 r member
  simp only [active,Bool.not_true,Bool.false_or,decide_eq_true_eq] at pointer
  omega

/-- All finite-width source arithmetic, formed pointers, active words, program
coverage, and writer uniqueness follow from the executable guards. -/
structure Safety (c : Shape) (U S V : Bool) (runtime : Runtime c U S V) (BD : Nat) : Prop where
  grid : c.BH ≤ 65535 ∧ (c.D+BD-1)/BD < 2^31 ∧ runtime.currentDevice=runtime.device
  forwardExact : ∀ block head lane step,block < (c.D+BD-1)/BD → head < c.BH → lane < BD → step ≤ c.T →
    ∀ x∈forwardNodes c BD block head lane step,(BitVec.ofInt 32 (x : Int)).toInt=(x : Int)
  reverseExact : ∀ block head lane step,block < (c.D+BD-1)/BD → head < c.BH → lane < BD → step < c.T →
    ∀ x∈reverseNodes c BD block head lane step,(BitVec.ofInt 32 (x : Int)).toInt=(x : Int)
  reverseIndex : ∀ step,step < c.T → ((c.T : Int)-1-step)=(c.T-1-step : Nat)
  previousIndex : ∀ head i,0 < i → ((head*c.T : Nat) : Int)+i-1=(head*c.T+i-1 : Nat)
  formedPointers : ∀ r,present U S V r=Bool.true → ∀ a,a ≤ c.high →
    (runtime.tensor r).base+4*a < 2^64 ∧
    (BitVec.ofNat 64 ((runtime.tensor r).base+4*a)).toNat=(runtime.tensor r).base+4*a
  activeWords : ∀ x,Active c U S V x →
    (runtime.tensor x.role).allocationBase ≤ (key runtime x).address ∧
    (key runtime x).address+4 ≤ (runtime.tensor x.role).allocationBase+(runtime.tensor x.role).allocationBytes ∧
    (key runtime x).address+4 ≤ 2^64 ∧ (key runtime x).address%4=0
  coverage : ∀ head d,head < c.BH → d < c.D →
    (d/BD,head)∈Execution.programs c.BH c.D BD ∧ d%BD < BD ∧ d/BD*BD+d%BD=d
  forwardUnique : ∀ id other r a,Execution.fWrites c.T c.D BD S id r a →
    Execution.fWrites c.T c.D BD S other r a → id=other
  backwardUnique : ∀ id other r a,Execution.bWrites c.T c.D BD id r a →
    Execution.bWrites c.T c.D BD other r a → id=other

theorem checked_safety (c : Shape) (U S V : Bool) (runtime : Runtime c U S V)
    (accepted : check c U S V runtime=Bool.true) (BD : Nat) (tile : tileCheck BD=Bool.true) :
    Safety c U S V runtime BD := by
  have facts := accepted
  simp only [check,Bool.and_eq_true,decide_eq_true_eq] at facts
  have shapes := shape_bounds c
  have tiles := tile_bounds BD tile
  have gridBound : (c.D+BD-1)/BD ≤ c.D := by
    apply (Nat.div_le_iff_le_mul_add_pred tiles.1).mpr
    have h := Nat.le_mul_of_pos_left c.D tiles.1
    omega
  refine ⟨⟨facts.1.2.1,by omega,facts.1.2.2.2.2⟩,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · intro block head lane step hb hh hl hs x hx
    exact VeriTile.Bench.Repairs.ChunkGateRecurrence.Machine.exact32 _
      (checked_i32 c U S V runtime accepted x
        (forward_nodes_bound c BD block head lane step tile hb hh hl hs x hx))
  · intro block head lane step hb hh hl hs x hx
    exact VeriTile.Bench.Repairs.ChunkGateRecurrence.Machine.exact32 _
      (checked_i32 c U S V runtime accepted x
        (reverse_nodes_bound c BD block head lane step tile hb hh hl hs x hx))
  · intro step hs
    omega
  · intro head i hi
    omega
  · intro r hr a ha
    have h := pointer_bound c U S V runtime accepted r hr a ha
    exact ⟨h,by rw [BitVec.toNat_ofNat]; exact Nat.mod_eq_of_lt h⟩
  · exact key_bounds c U S V runtime accepted
  · intro head d hh hd
    have owner := grid_owner c.D BD d tiles.1 hd
    exact ⟨(Execution.programs_mem c.BH c.D BD _).mpr ⟨owner.1,hh⟩,owner.2⟩
  · exact Execution.forward_unique c.T c.D BD tiles.1 S
  · exact Execution.backward_unique c.T c.D BD tiles.1

#axiomsClean checked_safety
#specNonCircular Safety avoiding [Execution.gridRun,state,dx,dg]
end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn.Checked
