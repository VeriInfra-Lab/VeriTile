import bench.tritonbench_g.kldiv_triton.Proofs.Physical
import bench.tritonbench_g.kldiv_triton.Proofs.Native
import bench.tritonbench_g.kldiv_triton.Proofs.Real
import bench.tritonbench_g.swiglu_triton.Proofs.Words

namespace VeriTile.Bench.Repairs.KldivTriton.Storage
open VeriTile.Triton FP.Structural
open L2NormBwd.Physical (Matrix)
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open VeriTile.Bench.TritonBenchG.KldivTriton.FullMath (rounds)
open Physical (ForwardLayout BackwardLayout rowValues jobs job_member)
noncomputable section
set_option autoImplicit false
set_option maxHeartbeats 4000000

/-- Storage values and interpreter nodes are distinct. Loads embed raw words;
stores project the checked source result, never provenance tags. -/
structure Adapter (α κ : Type) where
  model : Tile.Model κ
  embed : α → κ
  project : κ → α

def native (p : Native.Instructions) : Adapter Native.Word Native.Node :=
  ⟨Native.model p,Native.atom,Native.Node.value⟩
def nativeForward (p : Native.Instructions) (B : Nat) (flag : Bool) : Adapter Native.Word Native.Node :=
  ⟨Native.forwardModel p B flag,Native.atom,Native.Node.value⟩
def real : Adapter ℝ ℝ := ⟨RealModel.model,id,id⟩

def forwardState {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (m : Adapter α κ) (s : M)
    (p : ForwardLayout) (r : Nat) : State κ where
  mem := fun region i => .mk .real (m.embed (
    if region = "y_ptr" then ops.read s (p.prediction.view.storage.key (p.prediction.view.storageOffset+i))
    else if region = "gt_ptr" then ops.read s (p.target.view.storage.key (p.target.view.storageOffset+i))
    else ops.read s (p.output.view.storage.key (p.output.view.storageOffset+i))))
  regs := fun _ _ _ => none
  pids := fun ax => if ax=0 then r else 0
  numPids := fun _ => p.prediction.rows
  undef := fun d _ _ => defaultValue d

theorem forward_pid {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (m : Adapter α κ) (s : M)
    (p : ForwardLayout) (r : Nat) : (forwardState ops m s p r).pids 0 = r := rfl

def forwardValue {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (m : Adapter α κ) (s : M)
    (p : ForwardLayout) (positive : 0<p.block) (r c : Nat) : α :=
  m.project (((Tile.sourceRun m.model (forwardState ops m s p r) p.prediction.rowStride p.target.rowStride p.output.rowStride
    p.prediction.cols p.block p.flag p.reduction positive).val.mem "loss_ptr" (r*p.output.rowStride+c)).read .real)

def forwardKernel {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (m : Adapter α κ) (s : M)
    (p : ForwardLayout) (positive : 0<p.block) : M :=
  stores ops (fun rc => p.output.key rc.1 rc.2) (fun rc => forwardValue ops m s p positive rc.1 rc.2)
    (jobs p.output.rows p.output.cols) s

theorem forward_inputs {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (m : Adapter α κ) (s : M)
    (p : ForwardLayout) (r : Nat) (py : Index.UnitColumns p.prediction) (pt : Index.UnitColumns p.target)
    (hy : r < p.prediction.rows) (ht : r < p.target.rows) :
    (∀ c, c < p.prediction.cols → Tile.input (forwardState ops m s p r) "y_ptr" (r*p.prediction.rowStride) c = m.embed (rowValues ops s p.prediction r c)) ∧
    (∀ c, c < p.target.cols → Tile.input (forwardState ops m s p r) "gt_ptr" (r*p.target.rowStride) c = m.embed (rowValues ops s p.target r c)) := by
  constructor
  · intro c hc
    simp [Tile.input,forwardState,rowValues,Matrix.key,Matrix.index,py.active r c hy hc,Nat.add_assoc]
  · intro c hc
    simp [Tile.input,forwardState,rowValues,Matrix.key,Matrix.index,pt.active r c ht hc,Nat.add_assoc]

theorem forward_element {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (m : Adapter α κ) (s : M)
    (p : ForwardLayout) (positive : 0<p.block) (py : Index.UnitColumns p.prediction) (pt : Index.UnitColumns p.target)
    (mode : p.reduction=0) (r c : Nat) (active : c<p.prediction.cols) (hy : r<p.prediction.rows)
    (ht : r<p.target.rows) (targetCol : c<p.target.cols) :
    forwardValue ops m s p positive r c = m.project (Tile.loss m.model p.flag
      (m.model.alg.fp32Load (m.embed (rowValues ops s p.prediction r c))) (m.model.alg.fp32Load (m.embed (rowValues ops s p.target r c)))) := by
  unfold forwardValue
  rw [mode]
  have h := Tile.source_element m.model (forwardState ops m s p r) p.prediction.rowStride p.target.rowStride p.output.rowStride
    p.prediction.cols p.block p.flag positive c active
  simp only [forward_pid] at h
  rw [h,Cell.read_same,(forward_inputs ops m s p r py pt hy ht).1 c active,
    (forward_inputs ops m s p r py pt hy ht).2 c targetCol]

theorem forward_chunk {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (m : Adapter α κ) (s : M)
    (p : ForwardLayout) (positive : 0<p.block) (py : Index.UnitColumns p.prediction) (pt : Index.UnitColumns p.target)
    (mode : p.reduction≠0) (r c : Nat) (active : c<rounds p.prediction.cols p.block)
    (hy : r<p.prediction.rows) (ht : r<p.target.rows) (cols : p.target.cols=p.prediction.cols) :
    forwardValue ops m s p positive r c =
      m.project (Tile.chunkSum m.model p.flag p.prediction.cols p.block c
        (fun c => m.embed (rowValues ops s p.prediction r c)) (fun c => m.embed (rowValues ops s p.target r c))) := by
  unfold forwardValue
  have h := Tile.source_chunk m.model (forwardState ops m s p r) p.prediction.rowStride p.target.rowStride p.output.rowStride
    p.prediction.cols p.block p.flag p.reduction positive mode c active
  simp only [forward_pid] at h
  rw [h,Cell.read_same]
  apply congrArg m.project
  apply Tile.chunkSum_congr
  · exact (forward_inputs ops m s p r py pt hy ht).1
  · intro i hi
    exact (forward_inputs ops m s p r py pt hy ht).2 i (by rw [cols]; exact hi)

theorem forward_readback {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (m : Adapter α κ) (s : M)
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

theorem forward_bytes {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (m : Adapter α κ) (s : M)
    (p : ForwardLayout) (positive : 0<p.block) (owner address : Nat)
    (outside : ∀ r c,r<p.output.rows → c<p.output.cols → ¬(p.output.key r c).covers owner address) :
    ops.observe (forwardKernel ops m s p positive) owner address=ops.observe s owner address := by
  apply bytes_frame
  intro j hj
  obtain ⟨hr,hc⟩ := (job_member _ _ _ _).mp hj
  exact outside j.1 j.2 hr hc

def backwardState {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (m : Adapter α κ) (s : M)
    (p : BackwardLayout) (r : Nat) : State κ where
  mem := fun region i => .mk .real (m.embed (
    if region = "target_ptr" then ops.read s (p.target.view.storage.key (p.target.view.storageOffset+i))
    else ops.read s (p.work.view.storage.key (p.work.view.storageOffset+i))))
  regs := fun _ _ _ => none
  pids := fun ax => if ax=0 then r else 0
  numPids := fun _ => p.work.rows
  undef := fun d _ _ => defaultValue d

theorem backward_pid {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (m : Adapter α κ) (s : M)
    (p : BackwardLayout) (r : Nat) : (backwardState ops m s p r).pids 0 = r := rfl

def backwardValue {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (m : Adapter α κ) (s : M)
    (p : BackwardLayout) (positive : 0<p.block) (r c : Nat) : α :=
  m.project (((Backward.sourceRun m.model (backwardState ops m s p r) p.work.rowStride p.target.rowStride
    p.work.cols p.block p.flag positive).val.mem "input_ptr" (r*p.work.rowStride+c)).read .real)

def backwardKernel {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (m : Adapter α κ) (s : M)
    (p : BackwardLayout) (positive : 0<p.block) : M :=
  stores ops (fun rc => p.work.key rc.1 rc.2) (fun rc => backwardValue ops m s p positive rc.1 rc.2)
    (jobs p.work.rows p.work.cols) s

theorem backward_formula {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (m : Adapter α κ) (s : M)
    (p : BackwardLayout) (positive : 0<p.block) (pt : Index.UnitColumns p.target) (r c : Nat) (active : c<p.work.cols)
    (hr : r<p.target.rows) (hc : c<p.target.cols) :
    backwardValue ops m s p positive r c=m.project (Tile.gradient m.model p.flag (m.model.alg.fp32Load (m.embed (rowValues ops s p.target r c)))) := by
  unfold backwardValue
  have h := Backward.source_cell m.model (backwardState ops m s p r) p.work.rowStride p.target.rowStride
    p.work.cols p.block p.flag positive c active
  simp only [backward_pid] at h
  rw [h,Cell.read_same]
  simp [Tile.input,backwardState,rowValues,Matrix.key,Matrix.index,pt.active r c hr hc,Nat.add_assoc]

theorem backward_readback {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (m : Adapter α κ) (s : M)
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

theorem backward_bytes {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (m : Adapter α κ) (s : M)
    (p : BackwardLayout) (positive : 0<p.block) (owner address : Nat)
    (outside : ∀ r c,r<p.work.rows → c<p.work.cols → ¬(p.work.key r c).covers owner address) :
    ops.observe (backwardKernel ops m s p positive) owner address=ops.observe s owner address := by
  apply bytes_frame
  intro j hj
  obtain ⟨hr,hc⟩ := (job_member _ _ _ _).mp hj
  exact outside j.1 j.2 hr hc


#axiomsClean forward_element
#axiomsClean forward_chunk
#axiomsClean forward_readback
#axiomsClean forward_bytes
#axiomsClean backward_formula
#axiomsClean backward_readback
#axiomsClean backward_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Storage
