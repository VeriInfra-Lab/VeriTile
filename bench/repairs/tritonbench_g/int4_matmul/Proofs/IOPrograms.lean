import bench.repairs.tritonbench_g.int4_matmul.Proofs.IOAccess
import bench.repairs.tritonbench_g.int4_matmul.Proofs.IOCoordinates
import bench.repairs.tritonbench_g.int4_matmul.Proofs.IOBinding
import bench.repairs.tritonbench_g.int4_matmul.Proofs.Kernel.GraphCells

namespace VeriTile.Bench.Repairs.Int4Matmul.IOProof
open VeriTile.Triton FP.Structural KernelExecution KernelSource
noncomputable section
set_option maxHeartbeats 5000000
variable {α : Type} [Inhabited α]

theorem invocation_frame (x : Raw.Arguments) (t : Tile) (o : Ops α) (rest : Algebra α)
    (lanes : List Lane) (v : (IO.programIO x t o rest lanes).Inputs)
    (p : IO.Program) (initial final : RegisterState (SourceGraph.Node α))
    (ran : (IO.invocation x t o rest p).run initial=some final) :
    (IO.programIO x t o rest lanes).Frame v initial final := by
  intro r a outside
  have unchanged := SourceGraph.source_frame o rest (Raw.parameters x t)
    ((IO.invocation x t o rest p).initial initial) final ran r a
  rw [Invocation.initial_mem] at unchanged
  apply unchanged
  intro i live
  have hr : row x t ((IO.invocation x t o rest p).initial initial) i<x.shape.m := live.1
  have hc : column x t ((IO.invocation x t o rest p).initial initial) i<x.shape.n := live.2
  have out := outside.1 0 (⟨_,hr⟩,⟨_,hc⟩,()) (by trivial)
  simpa only [output_address] using out

theorem programs (x : Raw.Arguments) (geometry : Raw.Geometry x) (t : Tile)
    (member : t∈Host.sourceTiles) (o : Ops α) (rest : Algebra α)
    (lanes : List Lane) (v : (IO.programIO x t o rest lanes).Inputs)
    (base : RegisterState (SourceGraph.Node α)) :
    (IO.programIO x t o rest lanes).Programs v base := by
  intro current p _
  refine ⟨fun bounds inside=>invocation_safe x geometry t member o rest lanes v bounds inside p _,?_⟩
  obtain ⟨final,ran,_⟩ := SourceGraph.source_memory o rest (Raw.parameters x t)
    ((IO.invocation x t o rest p).initial ((IO.programIO x t o rest lanes).patchOutputs v current base))
  exact ⟨final,ran,invocation_frame x t o rest lanes v p _ final ran⟩

/-- Every contributing complete CTA gives the independent finite-operand
partial at this output, using its actual current atomic destination. -/
theorem source_value (x : Raw.Arguments) (geometry : Raw.Geometry x) (t : Tile)
    (member : t∈Host.sourceTiles) (o : Ops α) (rest : Algebra α)
    (lanes : List Lane) (v : (IO.programIO x t o rest lanes).Inputs)
    (domain : IO.domain x.shape o v) (base : RegisterState (SourceGraph.Node α))
    (bound : (IO.programIO x t o rest lanes).InputCells v base)
    (current : (IO.programIO x t o rest lanes).RawOutputs)
    (p : IO.Program) (listed : p∈IO.programs x.shape t)
    (j : TileIndex [x.shape.m,x.shape.n]) (own : IO.owns x.shape t p j.1.val j.2.1.val)
    (final : RegisterState (SourceGraph.Node α))
    (ran : (IO.invocation x t o rest p).run
      ((IO.programIO x t o rest lanes).patchOutputs v current base)=some final) :
    final.mem "out" (j.1.val*x.shape.n+j.2.1.val)=Cell.mk .real (.value
      (if t.splits=1 then Formula.partialSum x.shape t o v j.1.val j.2.1.val p.2 else
        o.atomicAdd (SourceGraph.observe o (current ⟨0,j⟩))
          (Formula.partialSum x.shape t o v j.1.val j.2.1.val p.2))) := by
  let state := (IO.invocation x t o rest p).initial
    ((IO.programIO x t o rest lanes).patchOutputs v current base)
  obtain ⟨i,hr,hc⟩ := owner_index x t (Host.all_tiles_valid t member) o rest p
    ((IO.programIO x t o rest lanes).patchOutputs v current base) j.1.val j.2.1.val own
  have live : active (Raw.parameters x t) state i := by
    change row x t state i<x.shape.m ∧ column x t state i<x.shape.n
    exact ⟨hr ▸ j.1.isLt,hc ▸ j.2.1.isLt⟩
  have addr : outputOffset (Raw.parameters x t) state i=j.1.val*x.shape.n+j.2.1.val := by
    rw [output_address,hr,hc]
  have cell := SourceGraph.source_cell o rest (Raw.parameters x t) state final ran i live
    (fun k hk same=>active_unique x t state i k live hk same)
  have operand := operand_binding x t o rest lanes v domain base bound current p
  have result := source_partial x geometry t member o v (SourceGraph.observeState o state) operand i
  have hw : Raw.work x t (SourceGraph.observeState o state) i=⟨j.1.val,j.2.1.val,p.2⟩ := by
    change Work.mk (row x t state i) (column x t state i) p.2=_
    rw [hr,hc]
  rw [hw] at result
  rw [partial_reference x.shape t o (operands x.shape o v) (operands x.shape o v)
    ⟨j.1.val,j.2.1.val,p.2⟩ ⟨j.1.isLt,j.2.1.isLt,(program_mem _ _ p).mp listed |>.2⟩
    ⟨fun _ _ _=>rfl,rfl⟩,formula_partial] at result
  have currentCell := (IO.programIO x t o rest lanes).toMemoryIO.patchOutputs_read v
    (output_unique x t o rest lanes v) current base ⟨0,j⟩ (by trivial)
  change ((IO.programIO x t o rest lanes).patchOutputs v current base).mem "out"
    (j.1.val*x.shape.n+j.2.1.val)=Cell.mk .real (current ⟨0,j⟩) at currentCell
  rw [result,addr] at cell
  change final.mem "out" _=Cell.mk .real (.value (if t.splits=1 then _ else
    o.atomicAdd (SourceGraph.observe o ((state.mem "out" _).read .real)) _)) at cell
  have original : state.mem "out" (j.1.val*x.shape.n+j.2.1.val)=Cell.mk .real (current ⟨0,j⟩) :=
    currentCell
  rw [original] at cell
  exact cell

/-- Every omitted CTA provably preserves this actual selected cell. -/
theorem source_missing (x : Raw.Arguments) (t : Tile) (ht : t.Valid)
    (o : Ops α) (rest : Algebra α) (lanes : List Lane)
    (v : (IO.programIO x t o rest lanes).Inputs) (base : RegisterState (SourceGraph.Node α))
    (current : (IO.programIO x t o rest lanes).RawOutputs)
    (p : IO.Program) (j : TileIndex [x.shape.m,x.shape.n])
    (miss : ¬IO.owns x.shape t p j.1.val j.2.1.val)
    (final : RegisterState (SourceGraph.Node α))
    (ran : (IO.invocation x t o rest p).run
      ((IO.programIO x t o rest lanes).patchOutputs v current base)=some final) :
    final.mem "out" (j.1.val*x.shape.n+j.2.1.val)=Cell.mk .real (current ⟨0,j⟩) := by
  have frame := SourceGraph.source_frame o rest (Raw.parameters x t)
    ((IO.invocation x t o rest p).initial ((IO.programIO x t o rest lanes).patchOutputs v current base))
    final ran "out" (j.1.val*x.shape.n+j.2.1.val)
  have unchanged := frame (by
    intro i live
    right
    intro same
    rw [output_address] at same
    have coords := address_injective x.shape ⟨j.1.val,j.2.1.val,0⟩
      ⟨row x t ((IO.invocation x t o rest p).initial _) i,
        column x t ((IO.invocation x t o rest p).initial _) i,0⟩ j.2.1.isLt live.2 same
    dsimp only at coords
    have own := source_owns x t ht o rest p ((IO.programIO x t o rest lanes).patchOutputs v current base) i
    rw [←coords.1,←coords.2] at own
    exact miss own)
  rw [Invocation.initial_mem] at unchanged
  rw [unchanged]
  exact (IO.programIO x t o rest lanes).toMemoryIO.patchOutputs_read v
    (output_unique x t o rest lanes v) current base ⟨0,j⟩ (by trivial)

#axiomsClean programs
#axiomsClean source_value
#axiomsClean source_missing
end
end VeriTile.Bench.Repairs.Int4Matmul.IOProof
