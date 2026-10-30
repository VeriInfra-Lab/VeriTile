import bench.tritonbench_g.parallel_attention.InputCongruence
import bench.tritonbench_g.parallel_attention.RealObserved
import bench.tritonbench_g.parallel_attention.Proofs.Contract

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
set_option maxHeartbeats 3000000

/-- The actual three independent grid coordinates. No flattened-grid product
or target-runtime launch bound is added to this logical schedule. -/
structure Program (shape : Shape) (L : Nat) where
  head : Fin (shape.B*shape.H)
  chunk : Fin (blocks shape.T L)
  value : Fin (blocks shape.V shape.BV)
  deriving DecidableEq

def Program.tile (p : Program shape L) : Tile := ⟨p.head.val,p.chunk.val,0,p.value.val⟩

def programs (shape : Shape) (L : Nat) : List (Program shape L) :=
  (List.finRange (shape.B*shape.H)).flatMap fun head =>
    (List.finRange (blocks shape.T L)).flatMap fun chunk =>
      (List.finRange (blocks shape.V shape.BV)).map fun value => ⟨head,chunk,value⟩

theorem program_member (shape : Shape) (L : Nat) (p : Program shape L) : p∈programs shape L := by
  rcases p with ⟨head,chunk,value⟩
  simp [programs]

def withProgram (p : Program shape L) (s : State α) : State α :=
  {s with pids := fun ax => if ax=0 then p.value.val else if ax=1 then p.chunk.val
    else if ax=2 then p.head.val else 0}

theorem program_tile (p : Program shape L) (s : State α) :
    launchTile shape (withProgram p s)=p.tile := by
  have h := p.value.isLt
  unfold blocks at h
  simp [launchTile,withProgram,Program.tile,Nat.div_eq_of_lt h,Nat.mod_eq_of_lt h]

theorem block_start (n b j : Nat) (positive : 0<b) (bound : j<blocks n b) : j*b<n := by
  have hdiv := Nat.div_mul_le_self (n+b-1) b
  unfold blocks at bound
  have hmul : (j+1)*b≤((n+b-1)/b)*b := Nat.mul_le_mul_right b bound
  nlinarith [Nat.sub_add_cancel (show 1≤n+b by omega)]

theorem program_domain (shape : Shape) (h : ShapeContract shape) (L : Nat) (positive : 0<L)
    (p : Program shape L) : TileDomain shape L p.tile where
  head := p.head.isLt
  row := block_start shape.T L p.chunk.val positive p.chunk.isLt
  key := rfl
  value := block_start shape.V shape.BV p.value.val (h.value_tile_positive shape) p.value.isLt
  keyFits := h.key_fits shape
  keyPositive := h.key_positive
  valueWidth := h.value_tile_positive shape

/-- The source grid executor is ordinary sequential composition of complete
quoted programs. A disjoint-store schedule can be permuted independently of
floating expression order within each program. -/
def runPrograms (launch : Program shape L → State α → Option (State α)) :
    List (Program shape L) → State α → Option (State α)
  | [],s => some s
  | p::rest,s => (launch p s).bind (runPrograms launch rest)

/-- Generic source-grid progress and complete cell frame, derived from each
program's proven execution; there is no run-success premise. -/
theorem runPrograms_total_frame
    (launch : Program shape L → State α → Option (State α))
    (writes : Program shape L → RegionName → Nat → Prop)
    (steps : ∀ p s,∃ final,launch p s=some final ∧
      ∀ R off,¬writes p R off → final.mem R off=s.mem R off)
    (ps : List (Program shape L)) (s : State α) :
    ∃ final,runPrograms launch ps s=some final ∧
      ∀ R off,(∀ p∈ps,¬writes p R off) → final.mem R off=s.mem R off := by
  induction ps generalizing s with
  | nil => exact ⟨s,rfl,by intros; rfl⟩
  | cons p rest ih =>
    obtain ⟨middle,first,firstFrame⟩ := steps p s
    obtain ⟨final,tail,tailFrame⟩ := ih middle
    refine ⟨final,by simp only [runPrograms,first,Option.bind_some,tail],?_⟩
    intro R off outside
    rw [tailFrame R off (by intro q hq; exact outside q (by simp [hq])),
      firstFrame R off (outside p (by simp))]

def queryLaunch (a : Scalars α) [Inhabited α] (shape : Shape) (σ : ℝ)
    (p : Program shape 64) (s : State α) : Option (State α) :=
  FP.Structural.exec (withDot (scalarAlgebra a) (scalarDotPrimitive a))
    (queryKernel (dqConfig shape σ p.tile)) (withProgram p s)

def keyValueLaunch (a : Scalars α) [Inhabited α] (shape : Shape) (σ : ℝ)
    (p : Program shape 64) (s : State α) : Option (State α) :=
  FP.Structural.exec (withDot (scalarAlgebra a) (scalarDotPrimitive a))
    (backwardKernel (backwardConfig shape σ p.tile)) (withProgram p s)

def forwardLaunch (a : Scalars α) [Inhabited α] (shape : Shape) (σ : ℝ)
    (p : Program shape 128) (s : State α) : Option (State α) :=
  FP.Structural.exec (withDot (scalarAlgebra a) (scalarDotPrimitive a))
    (forwardKernel (forwardConfig shape σ)) (withProgram p s)

theorem query_grid_progress {α : Type} [Inhabited α] (a : Scalars α) (shape : Shape) (σ : ℝ) (s : State α) :
    ∃ final,runPrograms (queryLaunch a shape σ) (programs shape 64) s=some final ∧
      ∀ R off,(∀ p∈programs shape 64,¬(queryRectangle shape p.tile).writes 64 shape.BK R off) →
        final.mem R off=s.mem R off := by
  apply runPrograms_total_frame
  intro p before
  obtain ⟨final,executed,_,frame⟩ := query_source a shape σ p.tile (withProgram p before)
  exact ⟨final,executed,frame⟩

theorem keyValue_grid_progress {α : Type} [Inhabited α] (a : Scalars α) (shape : Shape) (σ : ℝ)
    (s : State α) (keyWidth : 0<shape.BK) (valueWidth : 0<shape.BV) :
    ∃ final,runPrograms (keyValueLaunch a shape σ) (programs shape 64) s=some final ∧
      ∀ R off,(∀ p∈programs shape 64,¬((keyRectangle shape p.tile).writes 64 shape.BK R off ∨
        (valueRectangle shape p.tile).writes 64 shape.BV R off)) → final.mem R off=s.mem R off := by
  apply runPrograms_total_frame
  intro p before
  obtain ⟨final,executed,_,_,frame⟩ := keyValue_source a shape σ p.tile (withProgram p before) keyWidth valueWidth
  exact ⟨final,executed,frame⟩

theorem forward_grid_progress {α : Type} [Inhabited α] (a : Scalars α) (shape : Shape) (σ : ℝ)
    (s : State α) (valueWidth : 0<shape.BV) :
    ∃ final,runPrograms (forwardLaunch a shape σ) (programs shape 128) s=some final ∧
      ∀ R off,(∀ p∈programs shape 128,¬((outputRectangle shape p.tile 128).writes 128 shape.BV R off ∨
        normalizerWrites shape p.tile R off)) → final.mem R off=s.mem R off := by
  apply runPrograms_total_frame
  intro p before
  obtain ⟨final,executed,_,_,frame⟩ := forward_source a shape σ (withProgram p before) valueWidth
  refine ⟨final,executed,?_⟩
  rw [program_tile] at frame
  exact frame

#axiomsClean program_member
#axiomsClean program_tile
#axiomsClean block_start
#axiomsClean program_domain
#axiomsClean runPrograms_total_frame
#axiomsClean query_grid_progress
#axiomsClean keyValue_grid_progress
#axiomsClean forward_grid_progress
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel
