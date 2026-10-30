import bench.repairs.tritonbench_g.swiglu_triton.Proofs.DirectSource
import bench.repairs.tritonbench_g.swiglu_triton.Proofs.NativeProfile
namespace VeriTile.Bench.Repairs.SwigluTriton.DirectNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward.Observed (Term graphAlgebra)
open Memory DirectPhysical Native
noncomputable section
def graphState {M β : Type} (ops : Ops M Word β) (backward : Bool)
    (p : Layout) (first : Nat) (s : M) : State (Term Word) where
  mem := fun reg offset => .mk .real (.input (
    if reg="A" then ops.read s ((p.allocation .a).key (first*p.cols+offset))
    else if reg="B" then ops.read s ((p.allocation .b).key (first*p.cols+offset))
    else if reg=DirectSource.region backward .third then ops.read s ((p.allocation .third).key (first*p.cols+offset))
    else positiveZero))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun dtype _ _ => defaultValue dtype

theorem source_row_native {M β : Type} (ops : Ops M Word β) (instructions : InstructionSemantics)
    (backward : Bool) (p : Layout) (valid : Valid p) (first pid : Nat)
    (rowBound : first+pid<p.rows) (s : M) (ch : Channel) (on : enabled backward ch)
    (col : Nat) (active : col<p.cols) :
    let execution := DirectTile.sourceRun graphAlgebra backward p.block p.cols pid (graphState ops backward p first s)
    interpret (compiledPorts instructions) ((execution.val.mem (outputRegion ch) (pid*p.cols+col)).read .real)=
      ops.read (DirectPhysical.row ops (sourceScalar (compiledPorts instructions)) backward p (first+pid) s)
        (p.key (outputRole ch) (first+pid) col) := by
  dsimp only
  rw [DirectTile.readback graphAlgebra backward p.block p.cols pid (graphState ops backward p first s) ch on
    (⟨⟨col,lt_of_lt_of_le active valid.cols⟩,PUnit.unit⟩) active]
  rw [DirectPhysical.row_values ops _ backward p valid (first+pid) s rowBound ch on col active]
  cases backward <;> cases ch <;> try simp [enabled] at on
  all_goals simp [graphState,DirectSource.region,DirectPhysical.value,Layout.key,Nat.add_mul,Nat.add_assoc,
    sourceScalar,DirectTile.scalar,Tile.scalar,graphAlgebra,interpret,compiledPorts,sigmoidGraph]

theorem graphState_read {M β : Type} (ops : Ops M Word β) (backward : Bool)
    (p : Layout) (first : Nat) (s : M) (role : Role) (offset : Nat) :
    (graphState ops backward p first s).mem (DirectSource.region backward role) offset=
      Cell.mk .real (.input (ops.read s ((p.allocation role).key (first*p.cols+offset)))) := by
  cases backward <;> cases role <;> simp [graphState,DirectSource.region]

theorem source_row_native_full {M β : Type} (ops : Ops M Word β) (instructions : InstructionSemantics)
    (backward : Bool) (p : Layout) (valid : Valid p) (first pid : Nat)
    (rowBound : first+pid<p.rows) (s : M) :
    ∃ t,run graphAlgebra (directSource backward p.block p.cols).surfaceBody
        (DirectTile.rowState pid (graphState ops backward p first s))=some t ∧
      ∀ role offset,first*p.cols+offset<(p.allocation role).words →
        interpret (compiledPorts instructions) ((t.mem (DirectSource.region backward role) offset).read .real)=
          ops.read (DirectPhysical.row ops (sourceScalar (compiledPorts instructions)) backward p (first+pid) s)
            ((p.allocation role).key (first*p.cols+offset)) := by
  refine ⟨_,(DirectTile.sourceRun graphAlgebra backward p.block p.cols pid (graphState ops backward p first s)).property,?_⟩
  intro role offset backed
  by_cases hit : ∃ ch,enabled backward ch ∧ role=outputRole ch ∧
      ∃ col,col<p.cols ∧ offset=pid*p.cols+col
  · obtain ⟨ch,on,same,col,active,equal⟩ := hit
    subst role offset
    rw [DirectSource.output_region backward ch on]
    have keyEq : ((p.allocation (outputRole ch)).key (first*p.cols+(pid*p.cols+col)))=
        p.key (outputRole ch) (first+pid) col := by simp [Layout.key,Nat.add_mul,Nat.add_assoc]
    rw [keyEq]
    exact source_row_native ops instructions backward p valid first pid rowBound s ch on col active
  · have structural : DirectTile.Outside backward p.block p.cols pid (DirectSource.region backward role) offset := by
      intro ch on
      by_cases different : DirectSource.region backward role≠outputRegion ch
      · exact Or.inl different
      · right
        intro lane active equal
        have sameRole : role=outputRole ch := DirectSource.region_injective backward
          ((not_not.mp different).trans (DirectSource.output_region backward ch on).symm)
        exact hit ⟨ch,on,sameRole,lane.1.val,active,equal⟩
    rw [DirectTile.frame graphAlgebra backward p.block p.cols pid (graphState ops backward p first s) _ _ structural,
      graphState_read]
    change ops.read s _=ops.read _ _
    symm
    apply DirectPhysical.row_frame
    intro ch on col active
    by_cases sameRole : role=outputRole ch
    · subst role
      apply Allocation.indices_disjoint
      intro words
      have sameOffset : offset=pid*p.cols+col := by simp only [Nat.add_mul] at words; omega
      exact hit ⟨ch,on,rfl,col,active,sameOffset⟩
    · exact Allocation.spans_separate _ _ (valid.separate role (outputRole ch) sameRole) _ _
        (Allocation.key_contains _ _ backed)
        (key_backed p valid (outputRole ch) (first+pid) col rowBound active)

#axiomsClean source_row_native_full
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluTriton.DirectNative
