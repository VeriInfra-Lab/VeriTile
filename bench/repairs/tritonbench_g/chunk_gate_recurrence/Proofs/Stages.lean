import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.Binding

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.Binding
open Checked Machine
open VeriTile.Bench.Repairs.SwigluTriton.Memory (Key)
noncomputable section
open Classical

/-- Forward and backward have distinct physical entry states. In particular,
the incoming gradient can alias the output returned by the forward call. -/
def stageCells (c : Shape) (seed forward : Bool) : List Cell :=
  (cells c seed).filter fun x => if forward then x.region=.output else x.region≠.output

def stageCommit {M α β : Type} (api : API M α β) (c : Shape) (seed : Bool)
    (runtime : Runtime c seed) (forward : Bool) (values : Memory α) (m : M) : M :=
  VeriTile.Bench.Repairs.SwigluTriton.Memory.stores api (key c seed runtime)
    (fun x => values x.region x.program x.time x.lane) (stageCells c seed forward) m

theorem stage_mem (c : Shape) (seed forward : Bool) (x : Cell) :
    x∈stageCells c seed forward ↔ Source.writes c seed x.region x.program x.time x.lane ∧
      (if forward then x.region=.output else x.region≠.output) := by
  cases forward <;> simp [stageCells,cells_mem]

theorem stage_subset (c : Shape) (seed forward : Bool) (x : Cell)
    (member : x∈stageCells c seed forward) : x∈cells c seed :=
  (List.mem_filter.mp member).1

theorem stage_read {M α β : Type} (api : API M α β) (c : Shape) (seed : Bool)
    (runtime : Runtime c seed) (accepted : check c seed runtime=Bool.true)
    (forward : Bool) (values : Memory α) (m : M) (x : Cell)
    (member : x∈stageCells c seed forward) :
    api.read (stageCommit api c seed runtime forward values m) (key c seed runtime x)=
      values x.region x.program x.time x.lane := by
  apply VeriTile.Bench.Repairs.SwigluTriton.Memory.readback api _ _ _ m x member
  · intro y hy equal
    have same : y=x := by
      by_contra different
      exact Key.disjoint_ne (keys_disjoint c seed runtime accepted y x
        (stage_subset c seed forward y hy) (stage_subset c seed forward x member) different) equal
    rw [same]
  · intro y hy different
    exact keys_disjoint c seed runtime accepted x y (stage_subset c seed forward x member)
      (stage_subset c seed forward y hy) (by intro same; subst y; exact different rfl)

theorem backward_keeps_output {M α β : Type} (api : API M α β) (c : Shape) (seed : Bool)
    (runtime : Runtime c seed) (accepted : check c seed runtime=Bool.true)
    (values : Memory α) (m : M) (p t j : Nat)
    (hp : p<c.programs) (ht : t<c.N) (hj : j<1024) :
    api.read (stageCommit api c seed runtime Bool.false values m)
      (key c seed runtime ⟨.output,p,t,j⟩)=api.read m (key c seed runtime ⟨.output,p,t,j⟩) := by
  apply VeriTile.Bench.Repairs.SwigluTriton.Memory.read_frame
  intro y hy
  have facts := (stage_mem c seed Bool.false y).mp hy
  exact keys_disjoint c seed runtime accepted _ y
    ((cells_mem c seed _).mpr (Or.inl ⟨rfl,hp,ht,hj⟩))
    (stage_subset c seed Bool.false y hy) (by intro same; exact facts.2 (congrArg Cell.region same).symm)

def StageOutside (c : Shape) (seed : Bool) (runtime : Runtime c seed)
    (forward : Bool) (owner address : Nat) : Prop :=
  ∀ x,x∈stageCells c seed forward → ¬(key c seed runtime x).covers owner address

theorem stage_frame {M α β : Type} (api : API M α β) (c : Shape) (seed : Bool)
    (runtime : Runtime c seed) (forward : Bool) (values : Memory α) (m : M) (owner address : Nat)
    (outside : StageOutside c seed runtime forward owner address) :
    api.observe (stageCommit api c seed runtime forward values m) owner address=api.observe m owner address :=
  VeriTile.Bench.Repairs.SwigluTriton.Memory.bytes_frame api _ _ _ m owner address outside

#axiomsClean stage_read
#axiomsClean backward_keeps_output
#axiomsClean stage_frame
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.Binding
