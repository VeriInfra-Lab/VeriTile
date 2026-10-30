import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Kernel
namespace VeriTile.Bench.Repairs.L2NormBwd.RawKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open VeriTile.Bench.Repairs.SwigluTriton
noncomputable section

/-- Raw allocation and strided matrix geometry; no Tensor object or view flag. -/
structure Matrix where
  storage : Allocation
  storageOffset : Nat
  rows : Nat
  cols : Nat
  rowStride : Nat
  colStride : Nat

def Matrix.key (m : Matrix) (r c : Nat) : Key :=
  m.storage.key (m.storageOffset+r*m.rowStride+c*m.colStride)
def Matrix.backed (m : Matrix) : Prop := ∀ r c,r<m.rows → c<m.cols → m.storage.contains (m.key r c)
def Matrix.injective (m : Matrix) : Prop := ∀ r c r' c',r<m.rows → c<m.cols → r'<m.rows → c'<m.cols →
  m.key r c=m.key r' c' → r=r' ∧ c=c'
def Matrix.physical (m : Matrix) : Physical.Matrix :=
  ⟨⟨0,m.storage,[m.rows,m.cols],[m.rowStride,m.colStride],m.storageOffset,Bool.false,rfl⟩,
    m.rows,m.cols,m.rowStride,m.colStride,rfl,rfl⟩
def Matrix.ofPhysical (m : Physical.Matrix) : Matrix :=
  ⟨m.view.storage,m.view.storageOffset,m.rows,m.cols,m.rowStride,m.colStride⟩

structure Layout where
  x : Matrix
  dy : Matrix
  dx : Matrix
  block : Nat

def Layout.physical (p : Layout) : Physical.Layout := ⟨p.x.physical,p.dy.physical,p.dx.physical,p.block⟩
def Layout.ofPhysical (p : Physical.Layout) : Layout :=
  ⟨Matrix.ofPhysical p.x,Matrix.ofPhysical p.dy,Matrix.ofPhysical p.dx,p.block⟩

structure Layout.Valid (p : Layout) : Prop where
  positive : 0<p.x.cols
  block : p.x.cols≤p.block
  dyCols : p.dy.cols=p.x.cols
  dyRows : p.x.rows≤p.dy.rows
  dxCols : p.dx.cols=p.x.cols
  dxRows : p.dx.rows=p.x.rows
  xBacked : p.x.backed
  dyBacked : p.dy.backed
  dxBacked : p.dx.backed
  dxUnique : p.dx.injective
  xFresh : p.x.storage.separate p.dx.storage
  dyFresh : p.dy.storage.separate p.dx.storage

theorem Layout.valid (p : Layout) (h : p.Valid) : Physical.Valid p.physical :=
  ⟨h.positive,h.block,h.dyCols,h.dyRows,h.dxCols,h.dxRows,h.xBacked,h.dyBacked,
    h.dxBacked,h.dxUnique,h.xFresh,h.dyFresh,rfl,rfl,rfl⟩
theorem Layout.ofPhysical_valid (p : Physical.Layout) (h : Physical.Valid p) : (Layout.ofPhysical p).Valid :=
  ⟨h.positive,h.block,h.dyCols,h.dyRows,h.dxCols,h.dxRows,h.xBacked,h.dyBacked,
    h.dxBacked,h.dxUnique,h.xFresh,h.dyFresh⟩

/-- Erasing descriptor-only fields leaves source execution inputs unchanged. -/
theorem sourceState_ofPhysical {M α γ β : Type} [Inhabited γ] (ops : Ops M α β)
    (embed : α → γ) (p : Physical.Layout) (before : M) :
    Bridge.sourceState ops embed (Layout.ofPhysical p).physical before=Bridge.sourceState ops embed p before := rfl

theorem sourceRow_ofPhysical {M α γ β : Type} [Inhabited γ] (ops : Ops M α β)
    (A : Algebra γ) (embed : α → γ) (project : Nat → γ → α)
    (p : Physical.Layout) (eps : γ) (row : Nat) (before : M) :
    Bridge.sourceRow ops A embed project (Layout.ofPhysical p).physical eps row before=
      Bridge.sourceRow ops A embed project p eps row before := rfl

theorem sourceRows_ofPhysical {M α γ β : Type} [Inhabited γ] (ops : Ops M α β)
    (A : Algebra γ) (embed : α → γ) (project : Nat → γ → α)
    (p : Physical.Layout) (eps : γ) (rows : List Nat) (before : M) :
    Bridge.sourceRows ops A embed project (Layout.ofPhysical p).physical eps rows before=
      Bridge.sourceRows ops A embed project p eps rows before := by
  induction rows generalizing before with
  | nil => rfl
  | cons row rest ih =>
    rw [Bridge.sourceRows,Bridge.sourceRows,sourceRow_ofPhysical,ih]

theorem certificates_ofPhysical {M α γ β : Type} [Inhabited γ] (ops : Ops M α β)
    (A : Algebra γ) (embed : α → γ) (project : Nat → γ → α)
    (p : Physical.Layout) (eps : γ) (rows : List Nat) (before : M) :
    Kernel.Certificates ops A embed project (Layout.ofPhysical p).physical eps rows before=
      Kernel.Certificates ops A embed project p eps rows before := by
  induction rows generalizing before with
  | nil => rfl
  | cons row rest ih =>
    simp only [Kernel.Certificates,sourceRow_ofPhysical,ih]
    all_goals rfl


theorem real_row {M β : Type} (ops : Ops M ℝ β)
    (p : Physical.Layout) (valid : Physical.Valid p) (eps : ℝ) (before : M) (row : Nat) (hr : row<p.x.rows) :
    ∃ after, FP.Structural.exec Math.algebra (source p.strides p.x.cols p.block)
      (Tile.rowState row eps (Bridge.sourceState ops id p before))=some after ∧
      (∀ col, col<p.x.cols → (after.mem "DX" (row*p.dx.rowStride+col*p.dx.colStride)).read .real =
        Math.derivative p.x.cols eps (Physical.inputs ops before p.x row) (Physical.inputs ops before p.dy row) col) ∧
      Kernel.RowFrame p row (Bridge.sourceState ops id p before) after := by
  refine ⟨(Tile.sourceRun Math.algebra p.strides p.x.cols p.block row eps (Bridge.sourceState ops id p before)).val,
    (Tile.sourceRun Math.algebra p.strides p.x.cols p.block row eps (Bridge.sourceState ops id p before)).property,
    ?_,Kernel.source_frame Math.algebra p eps row (Bridge.sourceState ops id p before)⟩
  intro col hc
  rw [Bridge.actual_source_cell ops Math.algebra id p valid eps before row col hr hc]
  exact Math.source_derivative p.x.cols p.block valid.block eps _ _ col


theorem native_row {M β : Type} (ops : Ops M Native.Word β) (instructions : Native.Instructions)
    (p : Physical.Layout) (valid : Physical.Valid p) (eps : Native.Word) (before : M) (row : Nat) (hr : row<p.x.rows) :
    ∃ after, FP.Structural.exec (Native.algebra instructions p.block) (source p.strides p.x.cols p.block)
      (Tile.rowState row (Native.atom eps) (Bridge.sourceState ops Native.atom p before))=some after ∧
      (∀ col, col<p.x.cols → ((after.mem "DX" (row*p.dx.rowStride+col*p.dx.colStride)).read .real).value (col%128) =
        Native.row instructions p.x.cols p.block eps (Physical.inputs ops before p.x row) (Physical.inputs ops before p.dy row) col) ∧
      Kernel.RowFrame p row (Bridge.sourceState ops Native.atom p before) after := by
  refine ⟨(Tile.sourceRun (Native.algebra instructions p.block) p.strides p.x.cols p.block row
    (Native.atom eps) (Bridge.sourceState ops Native.atom p before)).val,
    (Tile.sourceRun (Native.algebra instructions p.block) p.strides p.x.cols p.block row
    (Native.atom eps) (Bridge.sourceState ops Native.atom p before)).property,
    ?_,Kernel.source_frame (Native.algebra instructions p.block) p (Native.atom eps) row (Bridge.sourceState ops Native.atom p before)⟩
  intro col hc
  rw [Bridge.actual_source_cell ops (Native.algebra instructions p.block) Native.atom p valid
    (Native.atom eps) before row col hr hc,Native.source_graph_lowering]


theorem real_grid {M β : Type} (ops : Ops M ℝ β)
    (p : Physical.Layout) (valid : Physical.Valid p) (eps : ℝ) (before : M) :
    Kernel.Certificates ops Math.algebra id (fun _ => id) p eps (List.range p.x.rows) before ∧
      let after := Bridge.sourceRows ops Math.algebra id (fun _ => id) p eps (List.range p.x.rows) before
      (∀ row col, row<p.x.rows → col<p.x.cols → ops.read after (p.dx.key row col) =
        Math.derivative p.x.cols eps (Physical.inputs ops before p.x row) (Physical.inputs ops before p.dy row) col) ∧ Kernel.ByteFrame ops p before after := by
  refine ⟨Kernel.certificates _ _ _ _ _ _ _ _,?_⟩
  dsimp only
  rw [Bridge.structural_rows ops Math.algebra p valid eps (List.range p.x.rows) before
    (fun row member => List.mem_range.mp member)]
  constructor
  · intro row col hr hc
    rw [Physical.run_values ops (Physical.sourceReduction Math.algebra) p valid eps
      (List.range p.x.rows) List.nodup_range (fun row member => List.mem_range.mp member)
      before row col (List.mem_range.mpr hr) hc]
    exact Math.source_derivative p.x.cols p.block valid.block eps _ _ col
  · intro owner address outside
    exact Physical.run_bytes ops (Physical.sourceReduction Math.algebra) p eps (List.range p.x.rows) before owner address
      (fun row member col hc => outside row col (List.mem_range.mp member) hc)


theorem native_grid {M : Type} (ops : Ops M Native.Word Words.Byte) (codec : Words.Codec ops)
    (instructions : Native.Instructions) (p : Physical.Layout) (valid : Physical.Valid p) (eps : Native.Word) (before : M) :
    Kernel.Certificates ops (Native.algebra instructions p.block) Native.atom (fun c n => n.value (c%128))
      p (Native.atom eps) (List.range p.x.rows) before ∧
      let after := Bridge.sourceRows ops (Native.algebra instructions p.block) Native.atom (fun c n => n.value (c%128))
        p (Native.atom eps) (List.range p.x.rows) before
      (∀ row col, row<p.x.rows → col<p.x.cols → ops.read after (p.dx.key row col) =
        Native.row instructions p.x.cols p.block eps (Physical.inputs ops before p.x row) (Physical.inputs ops before p.dy row) col ∧
        ∀ byte, byte<4 → ops.observe after (p.dx.key row col).owner ((p.dx.key row col).address+byte) =
          (Native.row instructions p.x.cols p.block eps (Physical.inputs ops before p.x row) (Physical.inputs ops before p.dy row) col).extractLsb' (8*byte) 8) ∧
      Kernel.ByteFrame ops p before after := by
  refine ⟨Kernel.certificates _ _ _ _ _ _ _ _,?_⟩
  dsimp only
  rw [Bridge.native_rows ops instructions p valid eps (List.range p.x.rows) before
    (fun row member => List.mem_range.mp member)]
  constructor
  · intro row col hr hc
    have value : ops.read (Physical.run ops (Native.reduction instructions) p eps (List.range p.x.rows) before)
        (p.dx.key row col)=Native.row instructions p.x.cols p.block eps (Physical.inputs ops before p.x row) (Physical.inputs ops before p.dy row) col := by
      rw [Physical.run_values ops (Native.reduction instructions) p valid eps
        (List.range p.x.rows) List.nodup_range (fun row member => List.mem_range.mp member)
        before row col (List.mem_range.mpr hr) hc]
      rfl
    refine ⟨value,?_⟩
    intro byte small
    rw [codec.byte _ _ byte small,value]
  · intro owner address outside
    exact Physical.run_bytes ops (Native.reduction instructions) p eps (List.range p.x.rows) before owner address
      (fun row member col hc => outside row col (List.mem_range.mp member) hc)


end
end VeriTile.Bench.Repairs.L2NormBwd.RawKernel
