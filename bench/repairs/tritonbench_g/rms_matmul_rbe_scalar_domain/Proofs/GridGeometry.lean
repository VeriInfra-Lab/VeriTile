import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.StoreTrace
import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.CallerSafety

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.PhysicalKernel
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

structure Commit where
  id : Program
  phase : Fin 3
  lane : TileIndex [16,64]
  deriving DecidableEq

def Commit.Valid (i : Contract.Inputs) (c : Commit) : Prop :=
  c.id.Valid i ∧ activeLane i c.id c.lane

def commits (i : Contract.Inputs) (id : Program) : List Commit :=
  (List.finRange 3).flatMap (fun phase => (lanes i id).map (fun lane => ⟨id,phase,lane⟩))

def allCommits (i : Contract.Inputs) : List Commit := (programs i).flatMap (commits i)

theorem lanes_mem (i : Contract.Inputs) (id : Program) (lane : TileIndex [16,64]) :
    lane∈lanes i id ↔ activeLane i id lane := by
  rw [lanes,List.mem_filter]
  simp only [TileShape.mem_allIndices [16,64] lane,true_and,decide_eq_true_eq]

theorem commits_mem (i : Contract.Inputs) (id : Program) (c : Commit) :
    c∈commits i id ↔ c.id=id ∧ activeLane i id c.lane := by
  constructor
  · intro member
    obtain ⟨phase,_,hm⟩ := List.mem_flatMap.mp member
    obtain ⟨lane,hl,eq⟩ := List.mem_map.mp hm
    subst c
    exact ⟨rfl,(lanes_mem i id lane).mp hl⟩
  · rintro ⟨same,live⟩
    apply List.mem_flatMap.mpr
    refine ⟨c.phase,List.mem_finRange _,List.mem_map.mpr ⟨c.lane,(lanes_mem i id c.lane).mpr live,?_⟩⟩
    cases c
    simp_all

theorem allCommits_mem (i : Contract.Inputs) (c : Commit) : c∈allCommits i ↔ c.Valid i := by
  simp [allCommits,commits_mem,programs_mem,Commit.Valid]

theorem output_index_lt (i : Contract.Inputs) (checked : Contract.InputContract i)
    (q : RmsMatmulRbe.Contract.Tensor) (hq : ExistingQ i q) (c : Commit) (hc : c.Valid i) :
    logicalOffset i c.id c.lane < (RmsMatmulRbe.Checked.output i.original q c.phase).metadata.numel := by
  rw [caller_output_count i checked q hq]
  have bound := RmsMatmulRbe.Checked.flat3_lt _ _ _ c.id.batch (row i c.id c.lane.1)
    (col i c.id c.lane.2.1) hc.1.1 hc.2.1 hc.2.2
  simpa [logicalOffset,RmsMatmulRbe.Checked.flat3,Nat.add_mul,Nat.mul_assoc] using bound

theorem output_inBacking (i : Contract.Inputs) (checked : Contract.InputContract i)
    (ordinary : OrdinaryInputs i) (q : RmsMatmulRbe.Contract.Tensor) (hq : ExistingQ i q)
    (c : Commit) (hc : c.Valid i) :
    inBacking (RmsMatmulRbe.Checked.output i.original q c.phase) (logicalOffset i c.id c.lane) :=
  inBacking_of_lt _ (caller_output_ordinary i ordinary q hq c.phase) _
    (output_index_lt i checked q hq c hc)

theorem output_pointer_valid (i : Contract.Inputs) (checked : Contract.InputContract i)
    (ordinary : OrdinaryInputs i) (q : RmsMatmulRbe.Contract.Tensor) (hq : ExistingQ i q)
    (c : Commit) (hc : c.Valid i) :
    RmsMatmulRbe.Checked.PointerValid (RmsMatmulRbe.Checked.output i.original q c.phase)
      (logicalOffset i c.id c.lane) :=
  RmsMatmulRbe.Checked.pointer_valid _ (caller_output_ordinary i ordinary q hq c.phase)
    (caller_output_admitted i checked q hq c.phase) _ (output_index_lt i checked q hq c hc)

def Commit.cell (i : Contract.Inputs) (c : Commit) (hc : c.Valid i) :
    RmsMatmulRbe.Cell (RmsMatmulRbe.Checked.shape i.original) :=
  (c.phase,⟨c.id.batch,hc.1.1⟩,⟨row i c.id c.lane.1,hc.2.1⟩,⟨col i c.id c.lane.2.1,hc.2.2⟩)

theorem coordinates_unique (i : Contract.Inputs) (a b : Commit) (ha : a.Valid i) (hb : b.Valid i)
    (phase : a.phase=b.phase) (offset : logicalOffset i a.id a.lane=logicalOffset i b.id b.lane) : a=b := by
  have coords := RmsMatmulRbe.Checked.flat3_injective (B:=(RmsMatmulRbe.Checked.shape i.original).B)
    (M:=(RmsMatmulRbe.Checked.shape i.original).M) (N:=(RmsMatmulRbe.Checked.shape i.original).N)
    (a₁:=(a.cell i ha).2) (a₂:=(b.cell i hb).2)
    (by simpa [Commit.cell,logicalOffset,RmsMatmulRbe.Checked.flat3,Nat.add_mul,Nat.mul_assoc] using offset)
  have batch : a.id.batch=b.id.batch := congrArg (fun x => x.1.val) coords
  have rows : row i a.id a.lane.1=row i b.id b.lane.1 := congrArg (fun x => x.2.1.val) coords
  have cols : col i a.id a.lane.2.1=col i b.id b.lane.2.1 := congrArg (fun x => x.2.2.val) coords
  have ar := a.lane.1.isLt
  have br := b.lane.1.isLt
  have ac := a.lane.2.1.isLt
  have bc := b.lane.2.1.isLt
  dsimp [row] at rows
  dsimp [col] at cols
  have divs : a.id.block/RmsMatmulRbe.tilesN (RmsMatmulRbe.Checked.shape i.original)=
      b.id.block/RmsMatmulRbe.tilesN (RmsMatmulRbe.Checked.shape i.original) := by omega
  have mods : a.id.block%RmsMatmulRbe.tilesN (RmsMatmulRbe.Checked.shape i.original)=
      b.id.block%RmsMatmulRbe.tilesN (RmsMatmulRbe.Checked.shape i.original) := by omega
  have blocks : a.id.block=b.id.block := by
    have ad := Nat.mod_add_div a.id.block (RmsMatmulRbe.tilesN (RmsMatmulRbe.Checked.shape i.original))
    have bd := Nat.mod_add_div b.id.block (RmsMatmulRbe.tilesN (RmsMatmulRbe.Checked.shape i.original))
    rw [divs,mods] at ad
    omega
  have r : a.lane.1=b.lane.1 := Fin.ext (by omega)
  have n : a.lane.2.1=b.lane.2.1 := Fin.ext (by omega)
  cases a with | mk aid ap al =>
    cases b with | mk bid bp bl =>
      cases aid; cases bid
      simp_all only [Program.mk.injEq,Commit.mk.injEq,Prod.mk.injEq]
      exact ⟨trivial,trivial,Prod.ext r (Prod.ext n (Subsingleton.elim _ _))⟩

theorem same_tensor_apart (t : RmsMatmulRbe.Contract.Tensor) (j k : Nat) (hne : j≠k) :
    Apart (address t j) (RmsMatmulRbe.Contract.width t.metadata.dtype)
      (address t k) (RmsMatmulRbe.Contract.width t.metadata.dtype) := by
  have positive := RmsMatmulRbe.Checked.width_positive t.metadata.dtype
  unfold Apart address
  rcases lt_or_gt_of_ne hne with hj|hk
  · left; nlinarith
  · right; nlinarith

theorem commit_apart (i : Contract.Inputs) (checked : Contract.InputContract i)
    (ordinary : OrdinaryInputs i) (q : RmsMatmulRbe.Contract.Tensor) (hq : ExistingQ i q)
    (a b : Commit) (ha : a.Valid i) (hb : b.Valid i) (hne : a≠b) :
    Apart (outputAddress i q a.id a.phase a.lane) (outputWidth (qkvOutputFP8 (useFP8 i) a.phase))
      (outputAddress i q b.id b.phase b.lane) (outputWidth (qkvOutputFP8 (useFP8 i) b.phase)) := by
  rw [caller_output_width i checked q hq,caller_output_width i checked q hq]
  by_cases phase : a.phase=b.phase
  · have offset : logicalOffset i a.id a.lane≠logicalOffset i b.id b.lane := by
      intro h
      exact hne (coordinates_unique i a b ha hb phase h)
    simpa only [outputAddress,phase] using same_tensor_apart
      (RmsMatmulRbe.Checked.output i.original q b.phase) _ _ offset
  · exact backing_apart _ _ _ _ (output_inBacking i checked ordinary q hq a ha)
      (output_inBacking i checked ordinary q hq b hb) (caller_outputs_apart i checked q hq a.phase b.phase phase)

def owner (i : Contract.Inputs) (c : RmsMatmulRbe.Cell (RmsMatmulRbe.Checked.shape i.original)) : Commit :=
  let l := RmsMatmulRbe.owner c
  ⟨⟨l.batch,l.tileM*RmsMatmulRbe.tilesN (RmsMatmulRbe.Checked.shape i.original)+l.tileN⟩,
    l.phase,l.row,l.col,PUnit.unit⟩

theorem owner_valid (i : Contract.Inputs) (c : RmsMatmulRbe.Cell (RmsMatmulRbe.Checked.shape i.original)) :
    (owner i c).Valid i := by
  have grid := RmsMatmulRbe.lane_grid (RmsMatmulRbe.owner c)
  have decode := RmsMatmulRbe.flat_pid_decode (RmsMatmulRbe.owner c)
  refine ⟨⟨(RmsMatmulRbe.owner c).batch.isLt,grid.2.2⟩,?_,?_⟩
  · simpa only [owner,activeLane,row,decode.1] using (RmsMatmulRbe.owner c).mActive
  · simpa only [owner,activeLane,col,decode.2] using (RmsMatmulRbe.owner c).nActive

theorem owner_cell (i : Contract.Inputs) (c : RmsMatmulRbe.Cell (RmsMatmulRbe.Checked.shape i.original)) :
    (owner i c).cell i (owner_valid i c)=c := by
  have decode := RmsMatmulRbe.flat_pid_decode (RmsMatmulRbe.owner c)
  simpa only [Commit.cell,owner,row,col,decode.1,decode.2,RmsMatmulRbe.Lane.cell] using RmsMatmulRbe.owner_cell c

theorem coverage (i : Contract.Inputs) (c : RmsMatmulRbe.Cell (RmsMatmulRbe.Checked.shape i.original)) :
    owner i c∈allCommits i := (allCommits_mem i _).mpr (owner_valid i c)

#axiomsClean output_index_lt
#axiomsClean output_pointer_valid
#axiomsClean coordinates_unique
#axiomsClean commit_apart
#axiomsClean coverage
#axiomsClean owner_cell
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.PhysicalKernel
