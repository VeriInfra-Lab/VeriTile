import bench.tritonbench_g.swiglu_triton.Proofs.Physical

namespace VeriTile.Bench.Repairs.SwigluTriton.Source
open VeriTile.Triton FP.Structural
open Memory Physical
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

/-- The actual chunk slice shifts each symbolic base by first*block words.
All loaded cells are reads of this very physical memory, including in-place
A/B. No independent entry array or assumed kernel output is introduced. -/
def state {M α β : Type} [Inhabited α] (ops : Ops M α β) (backward : Bool)
    (p : Layout) (first : Nat) (s : M) : State α where
  mem := fun reg offset => .mk .real (
    if reg="A" then ops.read s ((p.allocation .a).key (first*p.block+offset))
    else if reg="B" then ops.read s ((p.allocation .b).key (first*p.block+offset))
    else if reg=region backward .third then ops.read s ((p.allocation .third).key (first*p.block+offset))
    else default)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun dtype _ _ => defaultValue dtype

theorem state_read {M α β : Type} [Inhabited α] (ops : Ops M α β) (backward : Bool)
    (p : Layout) (first : Nat) (s : M) (role : Role) (offset : Nat) :
    (state ops backward p first s).mem (region backward role) offset=
      Cell.mk .real (ops.read s ((p.allocation role).key (first*p.block+offset))) := by
  cases backward <;> cases role <;> simp [state,region]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluTriton.Source
