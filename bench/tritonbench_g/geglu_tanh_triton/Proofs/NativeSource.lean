import bench.tritonbench_g.geglu_tanh_triton.Proofs.Native
import bench.tritonbench_g.geglu_tanh_triton.Proofs.Source

namespace VeriTile.Bench.Repairs.GegluTanhTriton.Native
open VeriTile.Triton FP.Structural Physical
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open VeriTile.Bench.TritonBenchG.SwigluBackward.Observed (Term graphAlgebra)
noncomputable section
open Classical
set_option maxHeartbeats 8000000

/-- Actual local physical reads become atoms in the complete typed source.
Only subsequent graph interpretation supplies the native instructions. -/
def graphState {M β : Type} (ops : Ops M Word β) (backward : Bool)
    (p : Layout) (s : M) : State (Term Word) where
  mem := fun reg offset => .mk .real (.input (
    if reg="A" then ops.read s (p.key .a offset)
    else if reg="B" then ops.read s (p.key .b offset)
    else if reg=Source.region backward .third then ops.read s (p.key .third offset)
    else word 0))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => Grid.grid p.count
  undef := fun dtype _ _ => defaultValue dtype

theorem graphState_read {M β : Type} (ops : Ops M Word β) (backward : Bool)
    (p : Layout) (s : M) (role : Role) (offset : Nat) :
    (graphState ops backward p s).mem (Source.region backward role) offset=
      Cell.mk .real (.input (ops.read s (p.key role offset))) := by
  cases backward <;> cases role <;> simp [graphState,Source.region]

theorem tile_native_values {M β : Type} (ops : Ops M Word β) (instructions : Instructions)
    (backward : Bool) (p : Layout) (valid : Valid p) (pid : Nat) (s : M)
    (ch : Channel) (on : enabled backward ch) (lane : Nat)
    (small : lane<128) (active : pid*128+lane<p.count) :
    let execution := Tile.sourceRun graphAlgebra backward 128 p.count pid (graphState ops backward p s)
    interpret (compiledPorts instructions)
      ((execution.val.mem (outputRegion ch) (pid*128+lane)).read .real)=
      ops.read (Physical.tile ops (sourceScalar (compiledPorts instructions)) backward p pid s)
        (p.key (outputRole ch) (pid*128+lane)) := by
  dsimp only
  rw [Tile.readback graphAlgebra backward 128 p.count pid (graphState ops backward p s) ch on
    (⟨⟨lane,small⟩,PUnit.unit⟩) active]
  rw [Physical.tile_values ops _ backward p valid pid s ch on lane
    ((lanes_mem _ _ _).mpr ⟨small,active⟩)]
  cases backward <;> cases ch <;> try simp [enabled] at on
  all_goals simp [graphState,Source.region,Physical.value,sourceScalar,Tile.scalar,
    graphAlgebra,interpret,compiledPorts]

/-- Full source execution and physical word frame on the same arbitrary
prior heap, including masked lanes and every untouched private backing word. -/
theorem tile_native_refines {M β : Type} (ops : Ops M Word β) (instructions : Instructions)
    (backward : Bool) (p : Layout) (valid : Valid p) (pid : Nat) (s : M) :
    ∃ t,run graphAlgebra (source backward 128 p.count).surfaceBody
        (Tile.rowState pid (graphState ops backward p s))=some t ∧
      ∀ role offset,offset<(p.allocation role).words →
        interpret (compiledPorts instructions) ((t.mem (Source.region backward role) offset).read .real)=
          ops.read (Physical.tile ops (sourceScalar (compiledPorts instructions)) backward p pid s)
            (p.key role offset) := by
  refine ⟨_,(Tile.sourceRun graphAlgebra backward 128 p.count pid (graphState ops backward p s)).property,?_⟩
  intro role offset backedWord
  by_cases hit : ∃ ch,enabled backward ch ∧ role=outputRole ch ∧
      ∃ lane,lane<128 ∧ pid*128+lane<p.count ∧ offset=pid*128+lane
  · obtain ⟨ch,on,same,lane,small,active,equal⟩ := hit
    subst role offset
    rw [Source.output_region backward ch on]
    exact tile_native_values ops instructions backward p valid pid s ch on lane small active
  · have structural : Tile.Outside backward 128 p.count pid (Source.region backward role) offset := by
      intro ch on
      by_cases different : Source.region backward role≠outputRegion ch
      · exact Or.inl different
      · right
        intro lane active equal
        have sameRole : role=outputRole ch := Source.region_injective backward
          ((not_not.mp different).trans (Source.output_region backward ch on).symm)
        exact hit ⟨ch,on,sameRole,lane.1.val,lane.1.isLt,active,equal⟩
    rw [Tile.frame graphAlgebra backward 128 p.count pid (graphState ops backward p s) _ _ structural,
      graphState_read]
    change ops.read s _=ops.read _ _
    symm
    apply Physical.tile_frame
    intro ch on lane member
    obtain ⟨small,active⟩ := (lanes_mem _ _ _).mp member
    by_cases sameRole : role=outputRole ch
    · subst role
      apply Allocation.indices_disjoint
      intro equal
      exact hit ⟨ch,on,rfl,lane,small,active,equal⟩
    · exact Allocation.spans_separate _ _ (valid.separate role (outputRole ch) sameRole) _ _
        (Allocation.key_contains _ _ backedWord) (backed p valid (outputRole ch) _ active)

def Certificates {M β : Type} (ops : Ops M Word β) (instructions : Instructions)
    (backward : Bool) (p : Layout) (schedule : List Nat) (before : M) : Prop :=
  ∀ prior pid suffix,schedule=prior++pid::suffix →
    let entry := Physical.run ops (sourceScalar (compiledPorts instructions)) backward p prior before
    ∃ t,run graphAlgebra (source backward 128 p.count).surfaceBody
        (Tile.rowState pid (graphState ops backward p entry))=some t ∧
      ∀ role offset,offset<(p.allocation role).words →
        interpret (compiledPorts instructions) ((t.mem (Source.region backward role) offset).read .real)=
          ops.read (Physical.tile ops (sourceScalar (compiledPorts instructions)) backward p pid entry)
            (p.key role offset)

theorem certificates {M β : Type} (ops : Ops M Word β) (instructions : Instructions)
    (backward : Bool) (p : Layout) (valid : Valid p) (schedule : List Nat) (before : M) :
    Certificates ops instructions backward p schedule before := by
  intro prior pid suffix same
  exact tile_native_refines ops instructions backward p valid pid _

#axiomsClean tile_native_refines
#axiomsClean certificates
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.GegluTanhTriton.Native
