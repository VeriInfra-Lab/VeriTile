import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Tile
import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Native

namespace VeriTile.Bench.Repairs.L2NormBwd.Bridge
open VeriTile.Triton FP.Structural Physical
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

/-- Each role starts at its actual descriptor's storage offset. X and DY can
alias because both are read from the same physical memory snapshot. -/
def roleKey (p : Layout) (region : RegionName) (offset : Nat) : Key :=
  if region="X" then p.x.view.storage.key (p.x.view.storageOffset+offset)
  else if region="DY" then p.dy.view.storage.key (p.dy.view.storageOffset+offset)
  else p.dx.view.storage.key (p.dx.view.storageOffset+offset)

def sourceState {M α γ β : Type} [Inhabited γ] (ops : Ops M α β) (embed : α → γ)
    (p : Layout) (memory : M) : State γ where
  mem := fun region offset => .mk .real (embed (ops.read memory (roleKey p region offset)))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun ax => if ax=0 then p.x.rows else 1
  undef := fun dtype _ _ => defaultValue dtype

theorem actual_source_cell {M α γ β : Type} [Inhabited γ] (ops : Ops M α β)
    (alg : Algebra γ) (embed : α → γ) (p : Layout) (valid : Valid p)
    (eps : γ) (memory : M) (r c : Nat) (hr : r<p.x.rows) (hc : c<p.x.cols) :
    ((Tile.sourceRun alg p.strides p.x.cols p.block r eps (sourceState ops embed p memory)).val.mem
      "DX" (r*p.dx.rowStride+c*p.dx.colStride)).read .real =
      Tile.scalar alg p.x.cols p.block eps (fun j => embed (inputs ops memory p.x r j))
        (fun j => embed (inputs ops memory p.dy r j)) c := by
  let target : TileIndex [p.block] := (⟨c,lt_of_lt_of_le hc valid.block⟩,PUnit.unit)
  have unique : ∀ i j,i<p.x.cols → j<p.x.cols → i*p.dx.colStride=j*p.dx.colStride → i=j := by
    intro i j hi hj same
    exact (valid.dxUnique r i r j (by rwa [valid.dxRows]) (by rwa [valid.dxCols])
      (by rwa [valid.dxRows]) (by rwa [valid.dxCols])
      (by simp only [Matrix.key,Matrix.index,same])).2
  have read := Tile.readback alg p.strides p.x.cols p.block r eps (sourceState ops embed p memory)
    target hc unique
  dsimp only [Layout.strides,target] at read ⊢
  rw [read,Cell.read_same]
  congr 1 <;> funext j <;>
    simp [Tile.input,sourceState,roleKey,inputs,Layout.strides,Matrix.key,Matrix.index,Nat.add_assoc]

/-- Project only the source's active DX stores into the actual allocation.
The preceding theorem constructs every source statement and proves each value.
The source frame theorem rules out every other region/address write. -/
def sourceRow {M α γ β : Type} [Inhabited γ] (ops : Ops M α β) (alg : Algebra γ)
    (embed : α → γ) (project : Nat → γ → α) (p : Layout) (eps : γ) (r : Nat) (memory : M) : M :=
  let result := (Tile.sourceRun alg p.strides p.x.cols p.block r eps (sourceState ops embed p memory)).val
  stores ops (p.dx.key r) (fun c => project c ((result.mem "DX" (r*p.dx.rowStride+c*p.dx.colStride)).read .real))
    (List.range p.x.cols) memory

private theorem stores_ext {M α β : Type} (ops : Ops M α β) (keys : Nat → Key)
    (xs ys : Nat → α) (jobs : List Nat) (memory : M) (same : ∀ i∈jobs,xs i=ys i) :
    stores ops keys xs jobs memory=stores ops keys ys jobs memory := by
  induction jobs generalizing memory with
  | nil => rfl
  | cons i rest ih =>
    rw [stores,stores,same i (by simp)]
    exact ih _ (fun j hj => same j (by simp [hj]))

theorem structural_row {M α β : Type} [Inhabited α] (ops : Ops M α β) (alg : Algebra α)
    (p : Layout) (valid : Valid p) (eps : α) (r : Nat) (memory : M) (hr : r<p.x.rows) :
    sourceRow ops alg id (fun _ => id) p eps r memory=
      Physical.row ops (sourceReduction alg) p eps r memory := by
  unfold sourceRow Physical.row
  apply stores_ext
  intro c hc
  have cell := actual_source_cell ops alg id p valid eps memory r c hr (List.mem_range.mp hc)
  exact cell

theorem native_row {M β : Type} (ops : Ops M Native.Word β) (instructions : Native.Instructions)
    (p : Layout) (valid : Valid p) (eps : Native.Word) (r : Nat) (memory : M) (hr : r<p.x.rows) :
    sourceRow ops (Native.algebra instructions p.block) Native.atom (fun c n => n.value (c%128))
      p (Native.atom eps) r memory=Physical.row ops (Native.reduction instructions) p eps r memory := by
  unfold sourceRow Physical.row
  apply stores_ext
  intro c hc
  rw [actual_source_cell ops (Native.algebra instructions p.block) Native.atom p valid
    (Native.atom eps) memory r c hr (List.mem_range.mp hc)]
  exact Native.source_graph_lowering instructions p.x.cols p.block eps _ _ c

def sourceRows {M α γ β : Type} [Inhabited γ] (ops : Ops M α β) (alg : Algebra γ)
    (embed : α → γ) (project : Nat → γ → α) (p : Layout) (eps : γ) : List Nat → M → M
  | [],memory => memory
  | r::rest,memory => sourceRows ops alg embed project p eps rest (sourceRow ops alg embed project p eps r memory)

theorem structural_rows {M α β : Type} [Inhabited α] (ops : Ops M α β) (alg : Algebra α)
    (p : Layout) (valid : Valid p) (eps : α) (rows : List Nat) (memory : M)
    (bounded : ∀ r∈rows,r<p.x.rows) :
    sourceRows ops alg id (fun _ => id) p eps rows memory=
      Physical.run ops (sourceReduction alg) p eps rows memory := by
  induction rows generalizing memory with
  | nil => rfl
  | cons r rest ih =>
    rw [sourceRows,Physical.run,structural_row ops alg p valid eps r memory (bounded r (by simp))]
    exact ih _ (fun j hj => bounded j (by simp [hj]))

theorem native_rows {M β : Type} (ops : Ops M Native.Word β) (instructions : Native.Instructions)
    (p : Layout) (valid : Valid p) (eps : Native.Word) (rows : List Nat) (memory : M)
    (bounded : ∀ r∈rows,r<p.x.rows) :
    sourceRows ops (Native.algebra instructions p.block) Native.atom (fun c n => n.value (c%128))
      p (Native.atom eps) rows memory=Physical.run ops (Native.reduction instructions) p eps rows memory := by
  induction rows generalizing memory with
  | nil => rfl
  | cons r rest ih =>
    rw [sourceRows,Physical.run,native_row ops instructions p valid eps r memory (bounded r (by simp))]
    exact ih _ (fun j hj => bounded j (by simp [hj]))

#axiomsClean actual_source_cell
#axiomsClean structural_rows
#axiomsClean native_rows
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormBwd.Bridge
