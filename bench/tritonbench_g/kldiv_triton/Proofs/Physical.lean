import bench.tritonbench_g.kldiv_triton.Proofs.BackwardLoop
import bench.tritonbench_g.kldiv_triton.Proofs.Layout
import bench.tritonbench_g.kldiv_triton.Proofs.Index

namespace VeriTile.Bench.Repairs.KldivTriton.Physical
open VeriTile.Triton FP.Structural
open L2NormBwd.Physical (Matrix)
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open VeriTile.Bench.TritonBenchG.KldivTriton.FullMath (rounds outputCount)
noncomputable section
set_option autoImplicit false
set_option maxHeartbeats 4000000

theorem matrix_unique (m : Matrix)
    (guard : Layout.Writable m.rows m.cols m.rowStride m.colStride) : m.injective := by
  intro r c r' c' hr hc hr' hc' same
  have address := congrArg Key.address same
  have indices : Layout.index m.view.storageOffset m.rowStride m.colStride r c =
      Layout.index m.view.storageOffset m.rowStride m.colStride r' c' := by
    simp only [Matrix.key, Matrix.index, Allocation.key] at address
    unfold Layout.index
    omega
  have h := Layout.injective m.rows m.cols m.view.storageOffset m.rowStride m.colStride
    guard ⟨r,hr⟩ ⟨r',hr'⟩ ⟨c,hc⟩ ⟨c',hc'⟩ indices
  exact ⟨congrArg Fin.val h.1, congrArg Fin.val h.2⟩

theorem view_unique (m : Matrix) (unique : m.injective)
    (i j : Nat) (hi : i < m.view.numel) (hj : j < m.view.numel)
    (same : m.view.key i = m.view.key j) : i = j := by
  rw [m.numel] at hi hj
  have cols : 0 < m.cols := by
    by_contra h
    have empty : m.cols = 0 := by omega
    simp [empty] at hi
  have ir : i/m.cols<m.rows := (Nat.div_lt_iff_lt_mul cols).mpr (by simpa [Nat.mul_comm] using hi)
  have jr : j/m.cols<m.rows := (Nat.div_lt_iff_lt_mul cols).mpr (by simpa [Nat.mul_comm] using hj)
  have ic := Nat.mod_lt i cols
  have jc := Nat.mod_lt j cols
  have ie : i/m.cols*m.cols+i%m.cols=i := by simpa [Nat.mul_comm] using Nat.div_add_mod i m.cols
  have je : j/m.cols*m.cols+j%m.cols=j := by simpa [Nat.mul_comm] using Nat.div_add_mod j m.cols
  have ik := m.logical_key (i/m.cols) (i%m.cols) ic
  have jk := m.logical_key (j/m.cols) (j%m.cols) jc
  rw [ie] at ik
  rw [je] at jk
  rw [ik,jk] at same
  obtain ⟨row,col⟩ := unique _ _ _ _ ir ic jr jc same
  rw [row,col] at ie
  omega

theorem view_spans (m : Matrix) (unique : m.injective)
    (i j : Nat) (hi : i<m.view.numel) (hj : j<m.view.numel) (different : i≠j) :
    (m.view.key i).disjoint (m.view.key j) := by
  apply Allocation.indices_disjoint
  intro collision
  apply different
  apply view_unique m unique i j hi hj
  exact congrArg m.view.storage.key collision

def jobs (rows cols : Nat) : List (Nat × Nat) := (List.range rows).product (List.range cols)

theorem job_member (rows cols r c : Nat) : (r,c) ∈ jobs rows cols ↔ r<rows ∧ c<cols := by
  simp [jobs]

def rowValues {M α β : Type} (ops : Ops M α β) (s : M) (m : Matrix) (r : Nat) : Nat → α :=
  fun c => ops.read s (m.key r c)

structure ForwardLayout where
  prediction : Matrix
  target : Matrix
  output : Matrix
  block : Nat
  flag : Bool
  reduction : Nat

def forwardState {M α β : Type} [Inhabited α] (ops : Ops M α β) (s : M)
    (p : ForwardLayout) (r : Nat) : State α where
  mem := fun region i => .mk .real (
    if region = "y_ptr" then ops.read s (p.prediction.view.storage.key (p.prediction.view.storageOffset+i))
    else if region = "gt_ptr" then ops.read s (p.target.view.storage.key (p.target.view.storageOffset+i))
    else ops.read s (p.output.view.storage.key (p.output.view.storageOffset+i)))
  regs := fun _ _ _ => none
  pids := fun ax => if ax=0 then r else 0
  numPids := fun _ => p.prediction.rows
  undef := fun d _ _ => defaultValue d

theorem forward_pid {M α β : Type} [Inhabited α] (ops : Ops M α β) (s : M)
    (p : ForwardLayout) (r : Nat) : (forwardState ops s p r).pids 0 = r := rfl

def forwardValue {M α β : Type} [Inhabited α] (ops : Ops M α β) (m : Tile.Model α) (s : M)
    (p : ForwardLayout) (positive : 0<p.block) (r c : Nat) : α :=
  ((Tile.sourceRun m (forwardState ops s p r) p.prediction.rowStride p.target.rowStride p.output.rowStride
    p.prediction.cols p.block p.flag p.reduction positive).val.mem "loss_ptr" (r*p.output.rowStride+c)).read .real

def forwardKernel {M α β : Type} [Inhabited α] (ops : Ops M α β) (m : Tile.Model α) (s : M)
    (p : ForwardLayout) (positive : 0<p.block) : M :=
  stores ops (fun rc => p.output.key rc.1 rc.2) (fun rc => forwardValue ops m s p positive rc.1 rc.2)
    (jobs p.output.rows p.output.cols) s

theorem forward_inputs {M α β : Type} [Inhabited α] (ops : Ops M α β) (s : M)
    (p : ForwardLayout) (r : Nat) (py : Index.UnitColumns p.prediction) (pt : Index.UnitColumns p.target)
    (hy : r < p.prediction.rows) (ht : r < p.target.rows) :
    (∀ c, c < p.prediction.cols → Tile.input (forwardState ops s p r) "y_ptr" (r*p.prediction.rowStride) c = rowValues ops s p.prediction r c) ∧
    (∀ c, c < p.target.cols → Tile.input (forwardState ops s p r) "gt_ptr" (r*p.target.rowStride) c = rowValues ops s p.target r c) := by
  constructor
  · intro c hc
    simp [Tile.input,forwardState,rowValues,Matrix.key,Matrix.index,py.active r c hy hc,Nat.add_assoc]
  · intro c hc
    simp [Tile.input,forwardState,rowValues,Matrix.key,Matrix.index,pt.active r c ht hc,Nat.add_assoc]

theorem forward_element {M α β : Type} [Inhabited α] (ops : Ops M α β) (m : Tile.Model α) (s : M)
    (p : ForwardLayout) (positive : 0<p.block) (py : Index.UnitColumns p.prediction) (pt : Index.UnitColumns p.target)
    (mode : p.reduction=0) (r c : Nat) (active : c<p.prediction.cols) (hy : r<p.prediction.rows)
    (ht : r<p.target.rows) (targetCol : c<p.target.cols) :
    forwardValue ops m s p positive r c = Tile.loss m p.flag
      (m.alg.fp32Load (rowValues ops s p.prediction r c)) (m.alg.fp32Load (rowValues ops s p.target r c)) := by
  unfold forwardValue
  rw [mode]
  have h := Tile.source_element m (forwardState ops s p r) p.prediction.rowStride p.target.rowStride p.output.rowStride
    p.prediction.cols p.block p.flag positive c active
  simp only [forward_pid] at h
  rw [h,Cell.read_same,(forward_inputs ops s p r py pt hy ht).1 c active,
    (forward_inputs ops s p r py pt hy ht).2 c targetCol]

theorem forward_chunk {M α β : Type} [Inhabited α] (ops : Ops M α β) (m : Tile.Model α) (s : M)
    (p : ForwardLayout) (positive : 0<p.block) (py : Index.UnitColumns p.prediction) (pt : Index.UnitColumns p.target)
    (mode : p.reduction≠0) (r c : Nat) (active : c<rounds p.prediction.cols p.block)
    (hy : r<p.prediction.rows) (ht : r<p.target.rows) (cols : p.target.cols=p.prediction.cols) :
    forwardValue ops m s p positive r c =
      Tile.chunkSum m p.flag p.prediction.cols p.block c (rowValues ops s p.prediction r) (rowValues ops s p.target r) := by
  unfold forwardValue
  have h := Tile.source_chunk m (forwardState ops s p r) p.prediction.rowStride p.target.rowStride p.output.rowStride
    p.prediction.cols p.block p.flag p.reduction positive mode c active
  simp only [forward_pid] at h
  rw [h,Cell.read_same]
  apply Tile.chunkSum_congr
  · exact (forward_inputs ops s p r py pt hy ht).1
  · intro i hi
    exact (forward_inputs ops s p r py pt hy ht).2 i (by rw [cols]; exact hi)

theorem forward_readback {M α β : Type} [Inhabited α] (ops : Ops M α β) (m : Tile.Model α) (s : M)
    (p : ForwardLayout) (positive : 0<p.block) (unique : p.output.injective) (r c : Nat)
    (hr : r<p.output.rows) (hc : c<p.output.cols) :
    ops.read (forwardKernel ops m s p positive) (p.output.key r c) = forwardValue ops m s p positive r c := by
  unfold forwardKernel
  apply readback ops _ _ _ s (r,c) ((job_member _ _ _ _).mpr ⟨hr,hc⟩)
  · intro j hj same
    obtain ⟨jr,jc⟩ := (job_member _ _ _ _).mp hj
    obtain ⟨row,col⟩ := unique j.1 j.2 r c jr jc hr hc same
    simp only [row,col]
  · intro j _ different
    exact p.output.disjoint _ _ _ _ (Ne.symm different)

theorem forward_bytes {M α β : Type} [Inhabited α] (ops : Ops M α β) (m : Tile.Model α) (s : M)
    (p : ForwardLayout) (positive : 0<p.block) (owner address : Nat)
    (outside : ∀ r c,r<p.output.rows → c<p.output.cols → ¬(p.output.key r c).covers owner address) :
    ops.observe (forwardKernel ops m s p positive) owner address=ops.observe s owner address := by
  apply bytes_frame
  intro j hj
  obtain ⟨hr,hc⟩ := (job_member _ _ _ _).mp hj
  exact outside j.1 j.2 hr hc

structure BackwardLayout where
  target : Matrix
  work : Matrix
  block : Nat
  flag : Bool

def backwardState {M α β : Type} [Inhabited α] (ops : Ops M α β) (s : M)
    (p : BackwardLayout) (r : Nat) : State α where
  mem := fun region i => .mk .real (
    if region = "target_ptr" then ops.read s (p.target.view.storage.key (p.target.view.storageOffset+i))
    else ops.read s (p.work.view.storage.key (p.work.view.storageOffset+i)))
  regs := fun _ _ _ => none
  pids := fun ax => if ax=0 then r else 0
  numPids := fun _ => p.work.rows
  undef := fun d _ _ => defaultValue d

theorem backward_pid {M α β : Type} [Inhabited α] (ops : Ops M α β) (s : M)
    (p : BackwardLayout) (r : Nat) : (backwardState ops s p r).pids 0 = r := rfl

def backwardValue {M α β : Type} [Inhabited α] (ops : Ops M α β) (m : Tile.Model α) (s : M)
    (p : BackwardLayout) (positive : 0<p.block) (r c : Nat) : α :=
  ((Backward.sourceRun m (backwardState ops s p r) p.work.rowStride p.target.rowStride
    p.work.cols p.block p.flag positive).val.mem "input_ptr" (r*p.work.rowStride+c)).read .real

def backwardKernel {M α β : Type} [Inhabited α] (ops : Ops M α β) (m : Tile.Model α) (s : M)
    (p : BackwardLayout) (positive : 0<p.block) : M :=
  stores ops (fun rc => p.work.key rc.1 rc.2) (fun rc => backwardValue ops m s p positive rc.1 rc.2)
    (jobs p.work.rows p.work.cols) s

theorem backward_formula {M α β : Type} [Inhabited α] (ops : Ops M α β) (m : Tile.Model α) (s : M)
    (p : BackwardLayout) (positive : 0<p.block) (pt : Index.UnitColumns p.target) (r c : Nat) (active : c<p.work.cols)
    (hr : r<p.target.rows) (hc : c<p.target.cols) :
    backwardValue ops m s p positive r c=Tile.gradient m p.flag (m.alg.fp32Load (rowValues ops s p.target r c)) := by
  unfold backwardValue
  have h := Backward.source_cell m (backwardState ops s p r) p.work.rowStride p.target.rowStride
    p.work.cols p.block p.flag positive c active
  simp only [backward_pid] at h
  rw [h,Cell.read_same]
  simp [Tile.input,backwardState,rowValues,Matrix.key,Matrix.index,pt.active r c hr hc,Nat.add_assoc]

theorem backward_readback {M α β : Type} [Inhabited α] (ops : Ops M α β) (m : Tile.Model α) (s : M)
    (p : BackwardLayout) (positive : 0<p.block) (unique : p.work.injective) (r c : Nat)
    (hr : r<p.work.rows) (hc : c<p.work.cols) :
    ops.read (backwardKernel ops m s p positive) (p.work.key r c)=backwardValue ops m s p positive r c := by
  unfold backwardKernel
  apply readback ops _ _ _ s (r,c) ((job_member _ _ _ _).mpr ⟨hr,hc⟩)
  · intro j hj same
    obtain ⟨jr,jc⟩ := (job_member _ _ _ _).mp hj
    obtain ⟨row,col⟩ := unique j.1 j.2 r c jr jc hr hc same
    simp only [row,col]
  · intro j _ different
    exact p.work.disjoint _ _ _ _ (Ne.symm different)

theorem backward_bytes {M α β : Type} [Inhabited α] (ops : Ops M α β) (m : Tile.Model α) (s : M)
    (p : BackwardLayout) (positive : 0<p.block) (owner address : Nat)
    (outside : ∀ r c,r<p.work.rows → c<p.work.cols → ¬(p.work.key r c).covers owner address) :
    ops.observe (backwardKernel ops m s p positive) owner address=ops.observe s owner address := by
  apply bytes_frame
  intro j hj
  obtain ⟨hr,hc⟩ := (job_member _ _ _ _).mp hj
  exact outside j.1 j.2 hr hc

/-- Copy_ modifies the supplied view's actual allocation and preserves every
byte outside its active four-byte spans, even when target/gradient alias it. -/
theorem supplied_copyback {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    (destination : Matrix) (guard : Layout.Writable destination.rows destination.cols destination.rowStride destination.colStride)
    (values : Nat → α) (s : M) :
    (∀ i,i<destination.view.numel →
      logical ops neg (copyback ops neg destination.view values s) destination.view i=values i) ∧
    (∀ owner address,(∀ i,i<destination.view.numel → ¬(destination.view.key i).covers owner address) →
      ops.observe (copyback ops neg destination.view values s) owner address=ops.observe s owner address) := by
  constructor
  · intro i hi
    exact copyback_readback ops neg destination.view values s i hi
      (view_spans destination (matrix_unique destination guard))
  · intro owner address outside
    exact bytes_frame ops destination.view.key _ _ s owner address (fun i hi => outside i (List.mem_range.mp hi))

#axiomsClean matrix_unique
#axiomsClean forward_element
#axiomsClean forward_chunk
#axiomsClean forward_readback
#axiomsClean forward_bytes
#axiomsClean backward_formula
#axiomsClean backward_readback
#axiomsClean backward_bytes
#axiomsClean supplied_copyback
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Physical
