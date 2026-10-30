import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Contract
import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Bridge
import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Real
import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Integer
import bench.repairs.tritonbench_g.swiglu_triton.Proofs.Words

namespace VeriTile.Bench.Repairs.L2NormBwd.Kernel
open VeriTile.Triton FP.Structural Physical
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open VeriTile.Bench.Repairs.SwigluTriton
noncomputable section
open Classical
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

/-- Actual argument descriptors for the two direct input views and fresh DX.
The ordinary Tensor binding identifies metadata, pointers and logical keys.
The view-key laws are allocation/view semantics, with no value, execution or
output premise. Read-only X/DY overlap is allowed. -/
structure Buffers (x dy : TensorMetadata) (input gradient : View) (p : Layout) : Prop where
  inputTensor : TensorBinding x input
  gradientTensor : TensorBinding dy gradient
  valid : Valid p
  columns : p.x.cols = dy.shape.getLastD 0
  inputCount : p.x.rows*p.x.cols = x.numel
  gradientCount : p.dy.rows*p.dy.cols = dy.numel
  inputStorage : p.x.view.storage = input.storage
  gradientStorage : p.dy.view.storage = gradient.storage
  inputKeys : ∀ i, i < x.numel → p.x.view.key i = input.key i
  gradientKeys : ∀ i, i < dy.numel → p.dy.view.key i = gradient.key i
  block : p.block = min 16384 (2^Nat.clog 2 p.x.cols)
  xMetadata : Integer.Metadata p.x
  dyMetadata : Integer.Metadata p.dy
  dxMetadata : Integer.Metadata p.dx
  outputAlignment : p.dx.view.storage.baseByte%4=0

/-- Native launch ABI boundary, separate from Python input admission. -/
def LaunchCompatible (p : Layout) : Prop := Integer.GridCompatible p.x.rows

private theorem flat_bound (rows cols r c : Nat) (hr : r<rows) (hc : c<cols) :
    r*cols+c<rows*cols := by
  have bound := Nat.mul_le_mul_right cols (show r+1≤rows by omega)
  simp only [Nat.add_mul,Nat.one_mul] at bound
  omega

theorem input_key (x dy : TensorMetadata) (input gradient : View) (p : Layout)
    (binding : Buffers x dy input gradient p) (row col : Nat)
    (hr : row<p.x.rows) (hc : col<p.x.cols) :
    p.x.key row col=input.key (row*p.x.cols+col) := by
  rw [←p.x.logical_key row col hc]
  apply binding.inputKeys
  rw [←binding.inputCount]
  exact flat_bound _ _ _ _ hr hc

theorem gradient_key (x dy : TensorMetadata) (input gradient : View) (p : Layout)
    (binding : Buffers x dy input gradient p) (row col : Nat)
    (hr : row<p.x.rows) (hc : col<p.x.cols) :
    p.dy.key row col=gradient.key (row*p.x.cols+col) := by
  rw [←p.dy.logical_key row col (by rw [binding.valid.dyCols];exact hc),binding.valid.dyCols]
  apply binding.gradientKeys
  rw [←binding.gradientCount,binding.valid.dyCols]
  exact flat_bound _ _ _ _ (lt_of_lt_of_le hr binding.valid.dyRows) hc

/-- The Tensor data pointers are precisely the two source-region bases.
No pointer observation is needed for an empty launch. -/
theorem source_input_pointers (x dy : TensorMetadata) (h : InputContract x dy)
    (input gradient : View) (p : Layout) (binding : Buffers x dy input gradient p)
    (nonempty : 0<p.x.rows) :
    x.dataPtr=BitVec.ofNat 64 (p.x.key 0 0).address ∧
    dy.dataPtr=BitVec.ofNat 64 (p.dy.key 0 0).address := by
  have count : input.numel=x.numel := congrArg List.prod binding.inputTensor.shape.symm
  have dcount : gradient.numel=dy.numel := congrArg List.prod binding.gradientTensor.shape.symm
  have active : 0<x.numel := by rw [←binding.inputCount];exact Nat.mul_pos nonempty binding.valid.positive
  have hx := binding.inputTensor.pointer (by rw [count];exact active)
  have hd := binding.gradientTensor.pointer (by rw [dcount];exact lt_of_lt_of_le active h.enough)
  have xkey := input_key x dy input gradient p binding 0 0 nonempty binding.valid.positive
  have dkey := gradient_key x dy input gradient p binding 0 0 nonempty binding.valid.positive
  simp only [Nat.zero_mul,Nat.zero_add] at xkey dkey
  exact And.intro (hx.trans (congrArg (fun k : Key => BitVec.ofNat 64 k.address) xkey).symm)
      (hd.trans (congrArg (fun k : Key => BitVec.ofNat 64 k.address) dkey).symm)

/-- All active scalar FP32 accesses are aligned. Input alignment comes from
Python's actual pointer guards; output alignment is the allocator boundary. -/
theorem access_alignment (x dy : TensorMetadata) (h : InputContract x dy)
    (input gradient : View) (p : Layout) (binding : Buffers x dy input gradient p)
    (row col : Nat) (hr : row<p.x.rows) (_hc : col<p.x.cols) :
    (p.x.key row col).address%4=0 ∧ (p.dy.key row col).address%4=0 ∧ (p.dx.key row col).address%4=0 := by
  have nonempty : 0<p.x.rows := by omega
  have active : 0<x.numel := by rw [←binding.inputCount];exact Nat.mul_pos nonempty binding.valid.positive
  obtain ⟨xp,dp⟩ := source_input_pointers x dy h input gradient p binding nonempty
  obtain ⟨xa,da⟩ := h.alignment active
  have xb := binding.valid.xBacked 0 0 nonempty binding.valid.positive
  have db := binding.valid.dyBacked 0 0 (lt_of_lt_of_le nonempty binding.valid.dyRows)
    (by rw [binding.valid.dyCols];exact binding.valid.positive)
  have xr := p.x.view.storage.representable
  have dr := p.dy.view.storage.representable
  have xs : (p.x.key 0 0).address<2^64 := by rcases xb with ⟨_,_,upper⟩;omega
  have ds : (p.dy.key 0 0).address<2^64 := by rcases db with ⟨_,_,upper⟩;omega
  rw [xp,BitVec.toNat_ofNat,Nat.mod_eq_of_lt xs] at xa
  rw [dp,BitVec.toNat_ofNat,Nat.mod_eq_of_lt ds] at da
  simp only [Matrix.key,Matrix.index,Allocation.key,Nat.add_mod,Nat.mul_mod,Nat.mod_self,
    Nat.zero_mul,Nat.add_zero,Nat.mod_mod] at xa da ⊢
  exact ⟨xa,da,by simpa only [Nat.zero_mod,Nat.mod_mod,Nat.add_zero] using binding.outputAlignment⟩

/-- Actual source loads refer to the original Tensor keys, including offsets
and aliases, through one shared physical memory. -/
theorem source_loads {M α γ β : Type} [Inhabited γ] (ops : Ops M α β)
    (embed : α → γ) (x dy : TensorMetadata) (input gradient : View) (p : Layout)
    (binding : Buffers x dy input gradient p) (before : M) (row col : Nat)
    (hr : row<p.x.rows) (hc : col<p.x.cols) :
    Tile.input (Bridge.sourceState ops embed p before) "X" row p.x.rowStride p.x.colStride col =
      embed (ops.read before (input.key (row*p.x.cols+col))) ∧
    Tile.input (Bridge.sourceState ops embed p before) "DY" row p.dy.rowStride p.dy.colStride col =
      embed (ops.read before (gradient.key (row*p.x.cols+col))) := by
  have hx := input_key x dy input gradient p binding row col hr hc
  have hd := gradient_key x dy input gradient p binding row col hr hc
  constructor <;> simp [Tile.input,Bridge.sourceState,Bridge.roleKey,Cell.read,
    ←hx,←hd,Matrix.key,Matrix.index,Nat.add_assoc]

/-- Checked negative flags identify physical read words with logical Tensor
values. No whole-input value equation is supplied by the caller. -/
theorem logical_loads {M α γ β : Type} [Inhabited γ] (ops : Ops M α β)
    (neg : Negation α) (embed : α → γ) (x dy : TensorMetadata) (h : InputContract x dy)
    (input gradient : View) (p : Layout) (binding : Buffers x dy input gradient p)
    (before : M) (row col : Nat) (hr : row<p.x.rows) (hc : col<p.x.cols) :
    Tile.input (Bridge.sourceState ops embed p before) "X" row p.x.rowStride p.x.colStride col =
      embed (logical ops neg before input (row*p.x.cols+col)) ∧
    Tile.input (Bridge.sourceState ops embed p before) "DY" row p.dy.rowStride p.dy.colStride col =
      embed (logical ops neg before gradient (row*p.x.cols+col)) := by
  have xn : input.negative=Bool.false := binding.inputTensor.negative ▸ h.noNeg.1
  have dn : gradient.negative=Bool.false := binding.gradientTensor.negative ▸ h.noNeg.2
  simpa [logical,Negation.encode,xn,dn] using
    source_loads ops embed x dy input gradient p binding before row col hr hc

/-- Signed products and actual pointer arithmetic, including all four access
bytes. Only the launched prefix of DY needs representable program IDs. -/
def AccessSafe (m : Matrix) (row col : Nat) : Prop :=
  (Integer.pointer m row col).toNat=(m.key row col).address ∧
  ∀ byte, byte<4 → (Integer.pointer m row col).toNat+byte<m.view.storage.baseByte+m.view.storage.bytes ∧
    (Integer.pointer m row col).toNat+byte<2^64

theorem access_safe (x dy : TensorMetadata) (input gradient : View) (p : Layout)
    (binding : Buffers x dy input gradient p) (launch : LaunchCompatible p)
    (row col : Nat) (hr : row<p.x.rows) (hc : col<p.x.cols) :
    AccessSafe p.x row col ∧ AccessSafe p.dy row col ∧ AccessSafe p.dx row col := by
  have pid : row<2^32 := by unfold LaunchCompatible Integer.GridCompatible at launch;omega
  obtain ⟨_,_,_,_,_,xp,xb⟩ := Integer.active_no_wrap_of_pid p.x binding.xMetadata binding.valid.xBacked
    row col pid hr hc
  obtain ⟨_,_,_,_,_,dp,db⟩ := Integer.active_no_wrap_of_pid p.dy binding.dyMetadata binding.valid.dyBacked
    row col pid (lt_of_lt_of_le hr binding.valid.dyRows) (by rw [binding.valid.dyCols];exact hc)
  obtain ⟨_,_,_,_,_,op,ob⟩ := Integer.active_no_wrap_of_pid p.dx binding.dxMetadata binding.valid.dxBacked
    row col pid (by rw [binding.valid.dxRows];exact hr) (by rw [binding.valid.dxCols];exact hc)
  exact ⟨⟨xp,xb⟩,⟨dp,db⟩,⟨op,ob⟩⟩

/-- Original Tensor coordinates, independent of the source and reductions. -/
def formula {M β : Type} (ops : Ops M ℝ β) (input gradient : View) (N : Nat)
    (eps : ℝ) (before : M) (row col : Nat) : ℝ :=
  Math.derivative N eps (fun j => ops.read before (input.key (row*N+j)))
    (fun j => ops.read before (gradient.key (row*N+j))) col

def nativeFormula {M β : Type} (ops : Ops M Native.Word β) (instructions : Native.Instructions)
    (input gradient : View) (N B : Nat) (eps : Native.Word) (before : M) (row col : Nat) : Native.Word :=
  Native.row instructions N B eps (fun j => ops.read before (input.key (row*N+j)))
    (fun j => ops.read before (gradient.key (row*N+j))) col

/-- Full exterior logical-memory frame at every source invocation. -/
def RowFrame {α : Type} (p : Layout) (row : Nat) (before after : State α) : Prop :=
  ∀ (region : RegionName) (offset : Nat), region≠"DX" ∨
    (∀ col, col<p.x.cols → offset≠row*p.dx.rowStride+col*p.dx.colStride) →
    after.mem region offset=before.mem region offset

theorem source_frame {α : Type} [Inhabited α] (A : Algebra α) (p : Layout)
    (eps : α) (row : Nat) (before : State α) :
    RowFrame p row before (Tile.sourceRun A p.strides p.x.cols p.block row eps before).val := by
  intro region offset outside
  apply Tile.frame
  rcases outside with different | outside
  · exact Or.inl different
  · exact Or.inr (fun lane active => outside lane.1.val active)

/-- A complete successful source body at each actual preceding physical heap.
The recursion advances through source-derived stores, not a supplied result. -/
def Certificates {M α γ β : Type} [Inhabited γ] (ops : Ops M α β) (A : Algebra γ)
    (embed : α → γ) (project : Nat → γ → α) (p : Layout) (eps : γ) : List Nat → M → Prop
  | [],_ => True
  | row::rest,before =>
      (∃ after, FP.Structural.exec A (source p.strides p.x.cols p.block)
        (Tile.rowState row eps (Bridge.sourceState ops embed p before))=some after ∧
        RowFrame p row (Bridge.sourceState ops embed p before) after) ∧
      Certificates ops A embed project p eps rest
        (Bridge.sourceRow ops A embed project p eps row before)

theorem certificates {M α γ β : Type} [Inhabited γ] (ops : Ops M α β) (A : Algebra γ)
    (embed : α → γ) (project : Nat → γ → α) (p : Layout) (eps : γ) (rows : List Nat) (before : M) :
    Certificates ops A embed project p eps rows before := by
  induction rows generalizing before with
  | nil => trivial
  | cons row rest ih =>
    refine ⟨⟨(Tile.sourceRun A p.strides p.x.cols p.block row eps (Bridge.sourceState ops embed p before)).val,
      (Tile.sourceRun A p.strides p.x.cols p.block row eps (Bridge.sourceState ops embed p before)).property,
      source_frame A p eps row (Bridge.sourceState ops embed p before)⟩,ih _⟩

theorem real_value {M β : Type} (ops : Ops M ℝ β)
    (x dy : TensorMetadata) (input gradient : View) (p : Layout)
    (binding : Buffers x dy input gradient p) (eps : ℝ) (before : M) (row col : Nat)
    (hr : row<p.x.rows) (hc : col<p.x.cols) :
    Physical.value ops (sourceReduction Math.algebra) p eps before row col =
      formula ops input gradient p.x.cols eps before row col := by
  change Tile.scalar Math.algebra p.x.cols p.block eps (inputs ops before p.x row)
    (inputs ops before p.dy row) col = _
  exact (Tile.scalar_congr Math.algebra p.x.cols p.block eps _ _
    (fun j => ops.read before (input.key (row*p.x.cols+j)))
    (fun j => ops.read before (gradient.key (row*p.x.cols+j)))
    (fun j hj => congrArg (ops.read before) (input_key x dy input gradient p binding row j hr hj))
    (fun j hj => congrArg (ops.read before) (gradient_key x dy input gradient p binding row j hr hj)) col hc).trans
      (Math.source_derivative p.x.cols p.block binding.valid.block eps _ _ col)

theorem native_value {M β : Type} (ops : Ops M Native.Word β) (instructions : Native.Instructions)
    (x dy : TensorMetadata) (input gradient : View) (p : Layout)
    (binding : Buffers x dy input gradient p) (eps : Native.Word) (before : M) (row col : Nat)
    (hr : row<p.x.rows) (hc : col<p.x.cols) :
    Physical.value ops (Native.reduction instructions) p eps before row col =
      nativeFormula ops instructions input gradient p.x.cols p.block eps before row col := by
  apply (Native.reduction instructions).congr p.x.cols p.block eps _ _ _ _
    (fun j hj => congrArg (ops.read before) (input_key x dy input gradient p binding row j hr hj))
    (fun j hj => congrArg (ops.read before) (gradient_key x dy input gradient p binding row j hr hj)) col hc

theorem real_row {M β : Type} (ops : Ops M ℝ β)
    (x dy : TensorMetadata) (input gradient : View) (p : Layout)
    (binding : Buffers x dy input gradient p) (eps : ℝ) (before : M) (row : Nat) (hr : row<p.x.rows) :
    ∃ after, FP.Structural.exec Math.algebra (source p.strides p.x.cols p.block)
      (Tile.rowState row eps (Bridge.sourceState ops id p before))=some after ∧
      (∀ col, col<p.x.cols → (after.mem "DX" (row*p.dx.rowStride+col*p.dx.colStride)).read .real =
        formula ops input gradient p.x.cols eps before row col) ∧
      RowFrame p row (Bridge.sourceState ops id p before) after := by
  refine ⟨(Tile.sourceRun Math.algebra p.strides p.x.cols p.block row eps (Bridge.sourceState ops id p before)).val,
    (Tile.sourceRun Math.algebra p.strides p.x.cols p.block row eps (Bridge.sourceState ops id p before)).property,
    ?_,source_frame Math.algebra p eps row (Bridge.sourceState ops id p before)⟩
  intro col hc
  rw [Bridge.actual_source_cell ops Math.algebra id p binding.valid eps before row col hr hc]
  exact real_value ops x dy input gradient p binding eps before row col hr hc

theorem native_row {M β : Type} (ops : Ops M Native.Word β) (instructions : Native.Instructions)
    (x dy : TensorMetadata) (input gradient : View) (p : Layout)
    (binding : Buffers x dy input gradient p) (eps : Native.Word) (before : M) (row : Nat) (hr : row<p.x.rows) :
    ∃ after, FP.Structural.exec (Native.algebra instructions p.block) (source p.strides p.x.cols p.block)
      (Tile.rowState row (Native.atom eps) (Bridge.sourceState ops Native.atom p before))=some after ∧
      (∀ col, col<p.x.cols → ((after.mem "DX" (row*p.dx.rowStride+col*p.dx.colStride)).read .real).value (col%128) =
        nativeFormula ops instructions input gradient p.x.cols p.block eps before row col) ∧
      RowFrame p row (Bridge.sourceState ops Native.atom p before) after := by
  refine ⟨(Tile.sourceRun (Native.algebra instructions p.block) p.strides p.x.cols p.block row
    (Native.atom eps) (Bridge.sourceState ops Native.atom p before)).val,
    (Tile.sourceRun (Native.algebra instructions p.block) p.strides p.x.cols p.block row
    (Native.atom eps) (Bridge.sourceState ops Native.atom p before)).property,
    ?_,source_frame (Native.algebra instructions p.block) p (Native.atom eps) row (Bridge.sourceState ops Native.atom p before)⟩
  intro col hc
  rw [Bridge.actual_source_cell ops (Native.algebra instructions p.block) Native.atom p binding.valid
    (Native.atom eps) before row col hr hc,Native.source_graph_lowering]
  exact native_value ops instructions x dy input gradient p binding eps before row col hr hc

/-- Complete byte frame, including caller buffers, allocation gaps and tails. -/
def ByteFrame {M α β : Type} (ops : Ops M α β) (p : Layout) (before after : M) : Prop :=
  ∀ owner address,
    (∀ row col, row<p.x.rows → col<p.x.cols → ¬(p.dx.key row col).covers owner address) →
    ops.observe after owner address=ops.observe before owner address

theorem real_grid {M β : Type} (ops : Ops M ℝ β)
    (x dy : TensorMetadata) (input gradient : View) (p : Layout)
    (binding : Buffers x dy input gradient p) (eps : ℝ) (before : M) :
    Certificates ops Math.algebra id (fun _ => id) p eps (List.range p.x.rows) before ∧
      let after := Bridge.sourceRows ops Math.algebra id (fun _ => id) p eps (List.range p.x.rows) before
      (∀ row col, row<p.x.rows → col<p.x.cols → ops.read after (p.dx.key row col) =
        formula ops input gradient p.x.cols eps before row col) ∧ ByteFrame ops p before after := by
  refine ⟨certificates _ _ _ _ _ _ _ _,?_⟩
  dsimp only
  rw [Bridge.structural_rows ops Math.algebra p binding.valid eps (List.range p.x.rows) before
    (fun row member => List.mem_range.mp member)]
  constructor
  · intro row col hr hc
    rw [Physical.run_values ops (sourceReduction Math.algebra) p binding.valid eps
      (List.range p.x.rows) List.nodup_range (fun row member => List.mem_range.mp member)
      before row col (List.mem_range.mpr hr) hc]
    exact real_value ops x dy input gradient p binding eps before row col hr hc
  · intro owner address outside
    exact Physical.run_bytes ops (sourceReduction Math.algebra) p eps (List.range p.x.rows) before owner address
      (fun row member col hc => outside row col (List.mem_range.mp member) hc)

theorem native_grid {M : Type} (ops : Ops M Native.Word Words.Byte) (codec : Words.Codec ops)
    (instructions : Native.Instructions) (x dy : TensorMetadata) (input gradient : View) (p : Layout)
    (binding : Buffers x dy input gradient p) (eps : Native.Word) (before : M) :
    Certificates ops (Native.algebra instructions p.block) Native.atom (fun c n => n.value (c%128))
      p (Native.atom eps) (List.range p.x.rows) before ∧
      let after := Bridge.sourceRows ops (Native.algebra instructions p.block) Native.atom (fun c n => n.value (c%128))
        p (Native.atom eps) (List.range p.x.rows) before
      (∀ row col, row<p.x.rows → col<p.x.cols → ops.read after (p.dx.key row col) =
        nativeFormula ops instructions input gradient p.x.cols p.block eps before row col ∧
        ∀ byte, byte<4 → ops.observe after (p.dx.key row col).owner ((p.dx.key row col).address+byte) =
          (nativeFormula ops instructions input gradient p.x.cols p.block eps before row col).extractLsb' (8*byte) 8) ∧
      ByteFrame ops p before after := by
  refine ⟨certificates _ _ _ _ _ _ _ _,?_⟩
  dsimp only
  rw [Bridge.native_rows ops instructions p binding.valid eps (List.range p.x.rows) before
    (fun row member => List.mem_range.mp member)]
  constructor
  · intro row col hr hc
    have value : ops.read (Physical.run ops (Native.reduction instructions) p eps (List.range p.x.rows) before)
        (p.dx.key row col)=nativeFormula ops instructions input gradient p.x.cols p.block eps before row col := by
      rw [Physical.run_values ops (Native.reduction instructions) p binding.valid eps
        (List.range p.x.rows) List.nodup_range (fun row member => List.mem_range.mp member)
        before row col (List.mem_range.mpr hr) hc]
      exact native_value ops instructions x dy input gradient p binding eps before row col hr hc
    refine ⟨value,?_⟩
    intro byte small
    rw [codec.byte _ _ byte small,value]
  · intro owner address outside
    exact Physical.run_bytes ops (Native.reduction instructions) p eps (List.range p.x.rows) before owner address
      (fun row member col hc => outside row col (List.mem_range.mp member) hc)

#specNonCircular formula avoiding [source,Tile.scalar,Physical.run]
#specNonCircular nativeFormula avoiding [source,Tile.scalar,Native.algebra,Physical.run]
#axiomsClean source_input_pointers
#axiomsClean access_alignment
#axiomsClean source_loads
#axiomsClean logical_loads
#axiomsClean access_safe
#axiomsClean real_row
#axiomsClean native_row
#axiomsClean real_grid
#axiomsClean native_grid
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormBwd.Kernel
