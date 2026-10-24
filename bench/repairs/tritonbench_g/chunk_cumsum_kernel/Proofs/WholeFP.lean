import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.FPGrid
import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.DirectHost

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.WholeFP
open VeriTile.Triton Layout FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]
abbrev Padding (α : Type) := (d : TileDType) → RegionName → Nat → Value α d

def entry (c : Shape) (p : Storage.Placement c) (padding : Padding α) (heap : Nat → Cell α) : State α where
  mem := Storage.lift c p (.mk .real default) heap
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun ax => if ax=0 then c.B*c.H else 1
  undef := padding

/-- Complete actual execution and independently executed logical-index source.
Every configured FP32 scan/reduction/cast retains its exact argument order. -/
theorem physical (ops : Config → FPRow.Ops α) (c : Shape) (p : Storage.Placement c)
    (padding : Padding α) (trials : List Config) (selected : Config) (heap : Nat → Cell α) :
    ∃ actual reference,
      Host.sourceLaunches (FPGrid.source ops c p.input p.output) (trials++[selected]) (entry c p padding heap)=some actual ∧
      Host.sourceLaunches (FPGrid.referenceSource ops c p.input p.output) (trials++[selected]) (entry c p padding heap)=some reference ∧
      (∀ row time, row<c.B*c.H → time<c.T →
        Storage.commit c p heap actual.mem (Storage.address p.backing.output (row*c.T+time))=
          reference.mem p.output (row*c.T+time)) ∧
      (∀ addr, ¬Storage.writes c p addr → Storage.commit c p heap actual.mem addr=heap addr) := by
  obtain ⟨actual,he,hf⟩ := FPGrid.choices_complete ops c p.input p.output (trials++[selected]) (entry c p padding heap)
  obtain ⟨x,reference,hx,hr,rel⟩ := FPGrid.choices_equivalent ops c p.input p.output (trials++[selected])
    (entry c p padding heap) (entry c p padding heap) ⟨rfl,rfl⟩
  have same : x=actual := Option.some.inj (hx.symm.trans he)
  subst x
  refine ⟨actual,reference,he,hr,?_,?_⟩
  · intro row time hrow htime
    rw [Storage.commit_output c p heap actual.mem _ ((Storage.offset_bound c row time hrow htime).trans_le p.backing.outputCells)]
    exact congrFun (congrFun rel.1 p.output) _
  · apply Storage.commit_frame c p (.mk .real default) heap actual.mem
    intro off hn
    exact hf p.output off (Or.inr (fun row hr time ht => hn row (by simpa using hr) time ht))

#axiomsClean physical
end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.WholeFP
