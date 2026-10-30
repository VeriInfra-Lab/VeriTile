import bench.repairs.tritonbench_g.l2_norm_triton2.Proofs.Tile
import bench.repairs.tritonbench_g.l2_norm_triton2.Proofs.Physical

namespace VeriTile.Bench.Repairs.L2NormTriton2.Source
open VeriTile.Triton FP.Structural Physical
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical
set_option maxHeartbeats 6000000

/-- An observation adapter can erase native compiler provenance after each
store. It does not assume anything about scalar arithmetic or reductions. -/
structure Adapter (α γ : Type) where
  algebra : Algebra γ
  inject : α → γ
  project : Nat → γ → α
  input : ∀ i x,project i (inject x)=x

def Adapter.compute {α γ : Type} (a : Adapter α γ) (p : Layout) (eps : γ)
    (xs dys : Nat → α) (i : Nat) : α :=
  a.project i (Tile.scalar a.algebra p.cols p.block eps
    (fun j => a.inject (xs j)) (fun j => a.inject (dys j)) i)

def region : Option Bool → RegionName
  | none => "DX" | some Bool.false => "X" | some Bool.true => "DY"
def allocation (p : Layout) : Option Bool → Allocation
  | none => p.output | some gradient => p.inputFor gradient

def state {M α β γ : Type} [Inhabited α] [Inhabited γ] (ops : Ops M α β)
    (a : Adapter α γ) (p : Layout) (s : M) : State γ where
  mem := fun name offset => .mk .real (a.inject (
    if name="X" then ops.read s (p.input.key offset)
    else if name="DY" then ops.read s (p.gradient.key offset)
    else if name="DX" then ops.read s (p.output.key offset) else default))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => p.rows
  undef := fun dtype _ _ => defaultValue dtype

theorem state_read {M α β γ : Type} [Inhabited α] [Inhabited γ] (ops : Ops M α β)
    (a : Adapter α γ) (p : Layout) (s : M) (output : Option Bool) (offset : Nat) :
    (state ops a p s).mem (region output) offset=
      Cell.mk .real (a.inject (ops.read s ((allocation p output).key offset))) := by
  cases output with
  | none => simp [state,region,allocation]
  | some gradient => cases gradient <;> simp [state,region,allocation,Layout.inputFor]

theorem active_scalar {M α β γ : Type} [Inhabited α] [Inhabited γ] (ops : Ops M α β)
    (a : Adapter α γ) (p : Layout) (row : Nat) (eps : γ) (s : M)
    (i : Nat) (hi : i<p.cols) :
    a.project i (Tile.scalar a.algebra p.cols p.block eps
      (Tile.inputs (state ops a p s) "X" p.cols row)
      (Tile.inputs (state ops a p s) "DY" p.cols row) i)=
      a.compute p eps (rowInputs ops p s row Bool.false) (rowInputs ops p s row Bool.true) i := by
  unfold Adapter.compute
  congr 1
  apply Tile.scalar_congr _ _ _ _ _ _ _ _ ?_ ?_ i hi
  all_goals
    intro j hj
    simp [Tile.inputs,state,rowInputs,hj,Layout.xkey,Layout.inputFor,Cell.read]

/-- Complete execution and refinement for both actual allocations, including
inactive words. Projection is taken at each word's logical column, preserving
the native lane-dependent butterfly order for sub-warp rows. -/
theorem tile_refines {M α β γ : Type} [Inhabited α] [Inhabited γ] (ops : Ops M α β)
    (a : Adapter α γ) (p : Layout) (v : Valid p) (row : Nat) (hr : row<p.rows)
    (eps : γ) (s : M) :
    ∃ t,FP.Structural.run a.algebra (backward p.cols p.block).surfaceBody
        (Tile.rowState row eps (state ops a p s))=some t ∧
      ∀ output offset,offset<(allocation p output).words →
        a.project (offset%p.cols) ((t.mem (region output) offset).read .real)=
          ops.read (Physical.tile ops (a.compute p eps) p row s)
            ((allocation p output).key offset) := by
  refine ⟨_,(Tile.sourceRun a.algebra p.cols p.block row eps (state ops a p s)).property,?_⟩
  intro output offset inside
  cases output with
  | some gradient =>
    have different : region (some gradient)≠"DX" := by cases gradient <;> simp [region]
    rw [Tile.frame _ _ _ _ _ _ _ _ (Or.inl different),state_read]
    simp only [Cell.read,dite_true,a.input,allocation]
    symm
    apply tile_frame
    intro col hc
    exact Allocation.spans_separate _ _ (Valid.validFor p v gradient).separate _ _
      (Allocation.key_contains _ _ inside) (ybacked p v row col hr hc)
  | none =>
    by_cases hit : ∃ col,col<p.cols ∧ offset=row*p.cols+col
    · obtain ⟨col,hc,rfl⟩ := hit
      have cb : col<p.block := lt_of_lt_of_le hc v.main.blockCovers
      have remainder : (row*p.cols+col)%p.cols=col := by simp [Nat.add_mod,Nat.mod_eq_of_lt hc]
      change a.project _ (((Tile.sourceRun _ _ _ _ _ _).val.mem "DX" _).read .real)=_
      rw [Tile.readback a.algebra p.cols p.block row eps (state ops a p s)
        (⟨⟨col,cb⟩,PUnit.unit⟩) hc]
      simp only [Cell.read,dite_true,remainder]
      rw [active_scalar ops a p row eps s col hc]
      exact (tile_values ops (a.compute p eps) p row s col hc).symm
    · have outside : ∀ i : TileIndex [p.block],i.1.val<p.cols → offset≠row*p.cols+i.1.val := by
        intro i hi equal
        exact hit ⟨i.1.val,hi,equal⟩
      rw [Tile.frame _ _ _ _ _ _ _ _ (Or.inr outside),state_read]
      simp only [Cell.read,dite_true,a.input,allocation]
      symm
      apply tile_frame
      intro col hc
      apply Allocation.indices_disjoint
      intro equal
      exact hit ⟨col,hc,equal⟩

/-- Each invocation is certified at its actual preceding physical heap,
including schedules which later stop or fail. -/
def Certificates {M α β γ : Type} [Inhabited α] [Inhabited γ] (ops : Ops M α β)
    (a : Adapter α γ) (p : Layout) (eps : γ) (schedule : List Nat) (s : M) : Prop :=
  ∀ prior row suffix,schedule=prior++row::suffix →
    let entry := Physical.run ops (a.compute p eps) p prior s
    ∃ t,FP.Structural.run a.algebra (backward p.cols p.block).surfaceBody
        (Tile.rowState row eps (state ops a p entry))=some t ∧
      ∀ output offset,offset<(allocation p output).words →
        a.project (offset%p.cols) ((t.mem (region output) offset).read .real)=
          ops.read (Physical.tile ops (a.compute p eps) p row entry)
            ((allocation p output).key offset)

theorem certificates {M α β γ : Type} [Inhabited α] [Inhabited γ] (ops : Ops M α β)
    (a : Adapter α γ) (p : Layout) (v : Valid p) (eps : γ) (schedule : List Nat)
    (bounded : ∀ row∈schedule,row<p.rows) (s : M) : Certificates ops a p eps schedule s := by
  intro prior row suffix equal
  exact tile_refines ops a p v row (bounded row (by rw [equal];simp)) eps _

#axiomsClean tile_refines
#axiomsClean certificates
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton2.Source
