import bench.tritonbench_g.context_attn_nopad.Proofs.IOScatter

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.IOLogical
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
open IOData
noncomputable section
set_option maxHeartbeats 3000000
variable {α : Type} [Inhabited α]

theorem tile_agreement (c : Call) (r : Regions) (o : FPBackend.Ops α) (idx : GridIndex c.grid)
    (s : State α) (memory : Memory α) (agree : Agreement c r memory s)
    (startLabel : r.out≠r.start) (lengthLabel : r.out≠r.sequence) :
    Agreement c r (step c r o idx memory) (FPReference.tile c r o idx s) := by
  constructor
  · exact (FPReference.grid_frame c r o [idx] s agree.metadata startLabel lengthLabel).1
  · intro region off hit
    rw [tile_memory c r o idx s agree.metadata,
      output_congr c r o idx _ memory agree.words]
    exact scatter_reads r.out (FPReference.laneOffset c idx) (FPReference.laneActive c idx)
      (output c r o idx memory) _ s memory region off (agree.words region off hit)

theorem tile_cell (c : Call) (r : Regions) (o : FPBackend.Ops α) (idx : GridIndex c.grid)
    (s : State α) (memory : Memory α) (agree : Agreement c r memory s)
    (region : RegionName) (off : Nat)
    (known : s.mem region off=Cell.mk .real (memory region off) ∨
      ∃ lane,FPReference.laneActive c idx lane ∧ region=r.out ∧ off=FPReference.laneOffset c idx lane) :
    (FPReference.tile c r o idx s).mem region off=Cell.mk .real (step c r o idx memory region off) := by
  rw [tile_memory c r o idx s agree.metadata,
    output_congr c r o idx _ memory agree.words]
  apply scatter_cells
  rcases known with eq|⟨lane,on,hr,ho⟩
  · exact Or.inl eq
  · exact Or.inr ⟨lane,TileShape.mem_allIndices _ lane,on,hr,ho⟩

theorem grid_agreement (c : Call) (r : Regions) (o : FPBackend.Ops α)
    (order : List (GridIndex c.grid)) (s : State α) (memory : Memory α) (agree : Agreement c r memory s)
    (startLabel : r.out≠r.start) (lengthLabel : r.out≠r.sequence) :
    Agreement c r (grid c r o order memory) (FPReference.grid c r o order s) ∧
    ∀ region off,s.mem region off=Cell.mk .real (memory region off) ∨ FPReference.Writes c r order region off →
      (FPReference.grid c r o order s).mem region off=Cell.mk .real (grid c r o order memory region off) := by
  induction order generalizing s memory with
  | nil =>
    refine ⟨agree,?_⟩
    intro region off known
    exact known.elim id (by simp [FPReference.Writes])
  | cons idx rest ih =>
    obtain ⟨finalAgree,finalCell⟩ := ih (FPReference.tile c r o idx s) (step c r o idx memory)
      (tile_agreement c r o idx s memory agree startLabel lengthLabel)
    refine ⟨finalAgree,?_⟩
    intro region off known
    apply finalCell
    rcases known with eq|⟨p,hp,lane,on,hr,ho⟩
    · exact Or.inl (tile_cell c r o idx s memory agree region off (Or.inl eq))
    · rcases List.mem_cons.mp hp with rfl|hp
      · exact Or.inl (tile_cell c r o p s memory agree region off (Or.inr ⟨lane,on,hr,ho⟩))
      · exact Or.inr ⟨p,hp,lane,on,hr,ho⟩

#axiomsClean grid_agreement
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.IOLogical
