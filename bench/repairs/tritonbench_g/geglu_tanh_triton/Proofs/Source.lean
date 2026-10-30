import bench.repairs.tritonbench_g.geglu_tanh_triton.Proofs.Physical
import bench.repairs.tritonbench_g.geglu_tanh_triton.Proofs.Grid

namespace VeriTile.Bench.Repairs.GegluTanhTriton.Source
open VeriTile.Triton FP.Structural Physical
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical
set_option maxHeartbeats 6000000

def region (backward : Bool) : Role → RegionName
  | .a => "A" | .b => "B" | .third => if backward then "DC" else "C"

theorem region_injective (backward : Bool) : Function.Injective (region backward) := by
  intro a b same
  cases backward <;> cases a <;> cases b <;> try rfl
  all_goals simp [region] at same

theorem output_region (backward : Bool) (ch : Channel) (on : enabled backward ch) :
    region backward (outputRole ch)=outputRegion ch := by
  cases backward <;> cases ch <;> try rfl
  all_goals simp [enabled] at on

/-- Reads of this very prior heap, rather than a detached entry array. -/
def state {M α β : Type} [Inhabited α] (ops : Ops M α β) (backward : Bool)
    (p : Layout) (s : M) : State α where
  mem := fun reg offset => .mk .real (
    if reg="A" then ops.read s (p.key .a offset)
    else if reg="B" then ops.read s (p.key .b offset)
    else if reg=region backward .third then ops.read s (p.key .third offset)
    else default)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => Grid.grid p.count
  undef := fun dtype _ _ => defaultValue dtype

theorem state_read {M α β : Type} [Inhabited α] (ops : Ops M α β) (backward : Bool)
    (p : Layout) (s : M) (role : Role) (offset : Nat) :
    (state ops backward p s).mem (region backward role) offset=
      Cell.mk .real (ops.read s (p.key role offset)) := by
  cases backward <;> cases role <;> simp [state,region]

theorem tile_values {M α β : Type} [Inhabited α] (ops : Ops M α β) (algebra : Algebra α)
    (backward : Bool) (p : Layout) (pid : Nat) (s : M)
    (ch : Channel) (on : enabled backward ch) (lane : Nat)
    (small : lane<128) (active : pid*128+lane<p.count) :
    (Tile.sourceRun algebra backward 128 p.count pid (state ops backward p s)).val.mem
      (outputRegion ch) (pid*128+lane)=
      Cell.mk .real (Physical.value ops (Tile.scalar algebra) backward p s ch (pid*128+lane)) := by
  rw [Tile.readback algebra backward 128 p.count pid (state ops backward p s) ch on
    (⟨⟨lane,small⟩,PUnit.unit⟩) active]
  cases backward <;> cases ch <;> try simp [enabled] at on
  all_goals simp [state,region,Physical.value,Tile.scalar]

/-- Complete local refinement for every physical scratch word, including
inactive words. The full source executes from the actual local entry heap. -/
theorem tile_refines {M α β : Type} [Inhabited α] (ops : Ops M α β) (algebra : Algebra α)
    (backward : Bool) (p : Layout) (valid : Valid p) (pid : Nat) (s : M) :
    ∃ t,run algebra (source backward 128 p.count).surfaceBody
        (Tile.rowState pid (state ops backward p s))=some t ∧
      ∀ role offset,offset<(p.allocation role).words →
        t.mem (region backward role) offset=Cell.mk .real
          (ops.read (Physical.tile ops (Tile.scalar algebra) backward p pid s) (p.key role offset)) := by
  refine ⟨_,(Tile.sourceRun algebra backward 128 p.count pid (state ops backward p s)).property,?_⟩
  intro role offset backedWord
  by_cases hit : ∃ ch,enabled backward ch ∧ role=outputRole ch ∧
      ∃ lane,lane<128 ∧ pid*128+lane<p.count ∧ offset=pid*128+lane
  · obtain ⟨ch,on,same,lane,small,active,equal⟩ := hit
    subst role offset
    rw [output_region backward ch on,tile_values ops algebra backward p pid s ch on lane small active,
      Physical.tile_values ops (Tile.scalar algebra) backward p valid pid s ch on lane
        ((lanes_mem _ _ _).mpr ⟨small,active⟩)]
  · have structural : Tile.Outside backward 128 p.count pid (region backward role) offset := by
      intro ch on
      by_cases different : region backward role≠outputRegion ch
      · exact Or.inl different
      · right
        intro lane active equal
        have sameRole : role=outputRole ch := region_injective backward
          ((not_not.mp different).trans (output_region backward ch on).symm)
        exact hit ⟨ch,on,sameRole,lane.1.val,lane.1.isLt,active,equal⟩
    rw [Tile.frame algebra backward 128 p.count pid (state ops backward p s) _ _ structural,state_read]
    congr 1
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

/-- Every scheduled invocation is tied to the physical heap after its actual
preceding invocations. This includes arbitrary partial/failure schedules. -/
def Certificates {M α β : Type} [Inhabited α] (ops : Ops M α β) (algebra : Algebra α)
    (backward : Bool) (p : Layout) (schedule : List Nat) (before : M) : Prop :=
  ∀ prior pid suffix,schedule=prior++pid::suffix →
    let entry := Physical.run ops (Tile.scalar algebra) backward p prior before
    ∃ t,run algebra (source backward 128 p.count).surfaceBody
        (Tile.rowState pid (state ops backward p entry))=some t ∧
      ∀ role offset,offset<(p.allocation role).words →
        t.mem (region backward role) offset=Cell.mk .real
          (ops.read (Physical.tile ops (Tile.scalar algebra) backward p pid entry) (p.key role offset))

theorem certificates {M α β : Type} [Inhabited α] (ops : Ops M α β) (algebra : Algebra α)
    (backward : Bool) (p : Layout) (valid : Valid p) (schedule : List Nat) (before : M) :
    Certificates ops algebra backward p schedule before := by
  intro prior pid suffix same
  exact tile_refines ops algebra backward p valid pid _

#axiomsClean tile_refines
#axiomsClean certificates
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.GegluTanhTriton.Source
