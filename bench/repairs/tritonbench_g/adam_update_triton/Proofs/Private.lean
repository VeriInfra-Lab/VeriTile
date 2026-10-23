import bench.repairs.tritonbench_g.adam_update_triton.Proofs.Grid
import bench.repairs.tritonbench_g.adam_update_triton.Proofs.Index
import bench.repairs.tritonbench_g.swiglu_triton.Proofs.Physical
namespace VeriTile.Bench.Repairs.AdamUpdateTriton.Private
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open VeriTile.Bench.Repairs.SwigluTriton (Channel)
namespace Shared
abbrev Physical := VeriTile.Bench.Repairs.SwigluTriton.Physical.Layout
end Shared
open VeriTile.Bench.Repairs.SwigluTriton.Physical (Role)
noncomputable section
set_option maxHeartbeats 6000000

structure Layout where
  allocation : Role → Allocation
  total : Nat

def Layout.flat (p : Layout) : VeriTile.Bench.Repairs.SwigluTriton.Physical.Layout :=
  ⟨p.allocation,p.total,1,1⟩

abbrev Valid (p : Layout) := VeriTile.Bench.Repairs.SwigluTriton.Physical.Valid p.flat

def region : Role → RegionName
  | .a => "P" | .b => "M" | .third => "G"

def channel : Output → Channel
  | .parameter => .da | .momentum => .db

def outputRole : Output → Role
  | .parameter => .a | .momentum => .b

def compute {α : Type} [Inhabited α] (sem : Tile.Semantics α) (params : Scalars α) :
    Channel → α → α → α → α
  | .c,_,_,_ => default
  | .da,p,m,g => Tile.scalar sem params .parameter p g m
  | .db,p,m,g => Tile.scalar sem params .momentum p g m

def run {M α β : Type} [Inhabited α] (ops : Ops M α β) (sem : Tile.Semantics α)
    (params : Scalars α) (p : Layout) (indices : List Nat) (s : M) : M :=
  VeriTile.Bench.Repairs.SwigluTriton.Physical.run ops (compute sem params) Bool.true p.flat indices s

def state {M α β : Type} [Inhabited α] (ops : Ops M α β) (p : Layout)
    (first : Nat) (s : M) : State α where
  mem := fun region offset => Cell.mk .real
    (if region="P" then ops.read s ((p.allocation .a).key (first+offset))
     else if region="M" then ops.read s ((p.allocation .b).key (first+offset))
     else if region="G" then ops.read s ((p.allocation .third).key (first+offset))
     else default)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun dtype _ _ => defaultValue dtype

theorem state_read {M α β : Type} [Inhabited α] (ops : Ops M α β) (p : Layout)
    (first : Nat) (s : M) (role : Role) (offset : Nat) :
    (state ops p first s).mem (region role) offset=
      Cell.mk .real (ops.read s ((p.allocation role).key (first+offset))) := by
  cases role <;> simp [state,region]

theorem values {M α β : Type} [Inhabited α] (ops : Ops M α β) (sem : Tile.Semantics α)
    (params : Scalars α) (p : Layout) (valid : Valid p) (indices : List Nat)
    (unique : indices.Nodup) (bounded : ∀ i∈indices,i<p.total) (s : M)
    (output : Output) (i : Nat) (member : i∈indices) :
    ops.read (run ops sem params p indices s) ((p.allocation (outputRole output)).key i)=
      Tile.scalar sem params output (ops.read s ((p.allocation .a).key i))
        (ops.read s ((p.allocation .third).key i)) (ops.read s ((p.allocation .b).key i)) := by
  have h := VeriTile.Bench.Repairs.SwigluTriton.Physical.run_values ops (compute sem params)
    Bool.true p.flat valid indices unique bounded s (channel output)
    (by cases output <;> decide) i 0 member (by change 0<1; decide)
  cases output <;> simpa [run,Layout.flat,channel,outputRole,compute,
    VeriTile.Bench.Repairs.SwigluTriton.Physical.Layout.key,
    VeriTile.Bench.Repairs.SwigluTriton.Physical.value,
    VeriTile.Bench.Repairs.SwigluTriton.Physical.outputRole] using h

theorem byte_frame {M α β : Type} [Inhabited α] (ops : Ops M α β) (sem : Tile.Semantics α)
    (params : Scalars α) (p : Layout) (indices : List Nat) (s : M) (owner address : Nat)
    (outside : ∀ i∈indices,∀ output,¬((p.allocation (outputRole output)).key i).covers owner address) :
    ops.observe (run ops sem params p indices s) owner address=ops.observe s owner address := by
  apply VeriTile.Bench.Repairs.SwigluTriton.Physical.run_bytes
  intro i member ch on col active
  have zero : col=0 := by change col<1 at active; omega
  subst col
  cases ch
  · simp [VeriTile.Bench.Repairs.SwigluTriton.enabled] at on
  · simpa [Layout.flat,VeriTile.Bench.Repairs.SwigluTriton.Physical.Layout.key,
      VeriTile.Bench.Repairs.SwigluTriton.Physical.outputRole,outputRole] using outside i member .parameter
  · simpa [Layout.flat,VeriTile.Bench.Repairs.SwigluTriton.Physical.Layout.key,
      VeriTile.Bench.Repairs.SwigluTriton.Physical.outputRole,outputRole] using outside i member .momentum

theorem read_frame {M α β : Type} [Inhabited α] (ops : Ops M α β) (sem : Tile.Semantics α)
    (params : Scalars α) (p : Layout) (valid : Valid p) (indices : List Nat)
    (bounded : ∀ j∈indices,j<p.total) (s : M) (role : Role) (i : Nat) (active : i<p.total)
    (outside : ∀ j∈indices,∀ output,role≠outputRole output ∨ i≠j) :
    ops.read (run ops sem params p indices s) ((p.allocation role).key i)=
      ops.read s ((p.allocation role).key i) := by
  apply VeriTile.Bench.Repairs.SwigluTriton.Physical.run_frame
  intro j member ch on col lane
  have zero : col=0 := by change col<1 at lane; omega
  subst col
  have separate (output : Output) :
      ((p.allocation role).key i).disjoint ((p.allocation (outputRole output)).key j) := by
    have h := VeriTile.Bench.Repairs.SwigluTriton.Physical.keys_disjoint p.flat valid
      role (outputRole output) i 0 j 0 active (by change 0<1; decide)
      (bounded j member) (by change 0<1; decide)
    have different : p.flat.key role i 0≠p.flat.key (outputRole output) j 0 := by
      intro equal
      have unique := VeriTile.Bench.Repairs.SwigluTriton.Physical.keys_unique p.flat valid
        role (outputRole output) i 0 j 0 active (by change 0<1; decide)
        (bounded j member) (by change 0<1; decide) equal
      rcases outside j member output with different | different
      · exact different unique.1
      · exact different unique.2.1
    simpa [Layout.flat,VeriTile.Bench.Repairs.SwigluTriton.Physical.Layout.key] using h different
  cases ch
  · simp [VeriTile.Bench.Repairs.SwigluTriton.enabled] at on
  · simpa [Layout.flat,VeriTile.Bench.Repairs.SwigluTriton.Physical.Layout.key,
      VeriTile.Bench.Repairs.SwigluTriton.Physical.outputRole,outputRole] using separate .parameter
  · simpa [Layout.flat,VeriTile.Bench.Repairs.SwigluTriton.Physical.Layout.key,
      VeriTile.Bench.Repairs.SwigluTriton.Physical.outputRole,outputRole] using separate .momentum

/-- Actual bound scalar registers and all nineteen source statements run for
every grid program. The resulting role memories agree with the physical-byte
execution on every backed private coordinate, including untouched coordinates.
Masked out-of-backing pointer formation is not treated as a dereference. -/
theorem chunk_source {M α β : Type} [Inhabited α] (ops : Ops M α β) (sem : Tile.Semantics α)
    (params : Scalars α) (p : Layout) (valid : Valid p) (first count block : Nat)
    (fits : first+count≤p.total) (positive : 0<block) (schedule : List Nat)
    (complete : schedule.Perm (List.range ((count+block-1)/block))) (s : M) :
    ∃ final,Grid.runPrograms sem params p.total count block schedule (state ops p first s)=some final ∧
      ∀ role i,first+i<p.total →
        final.mem (region role) i=Cell.mk .real
          (ops.read (run ops sem params p (List.range' first count) s)
            ((p.allocation role).key (first+i))) := by
  obtain ⟨final,ran,valuesSource,frameSource⟩ := Grid.whole sem params p.total count block
    positive schedule complete (state ops p first s)
  have bounded : ∀ j∈List.range' first count,j<p.total := by
    intro j member
    have := List.mem_range'_1.mp member
    omega
  refine ⟨final,ran,?_⟩
  intro role i active
  by_cases inside : i<count
  · have value (output : Output) :
        final.mem (Tile.outputRegion output) i=Cell.mk .real
          (ops.read (run ops sem params p (List.range' first count) s)
            ((p.allocation (outputRole output)).key (first+i))) := by
      rw [valuesSource output i inside,values ops sem params p valid _
        (List.nodup_range' (s:=first) (n:=count)) bounded s output (first+i)
        (List.mem_range'_1.mpr (by omega))]
      simp [state]
    cases role
    · exact value .parameter
    · exact value .momentum
    · rw [frameSource (region .third) i (Or.inl (by decide)),state_read,
        read_frame ops sem params p valid _ bounded s .third (first+i) active]
      intro j member output
      left
      cases output <;> decide
  · rw [frameSource (region role) i (Or.inr (by omega)),state_read,
      read_frame ops sem params p valid _ bounded s role (first+i) active]
    intro j member output
    right
    have := List.mem_range'_1.mp member
    omega

#axiomsClean values
#axiomsClean byte_frame
#axiomsClean chunk_source
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.Private
