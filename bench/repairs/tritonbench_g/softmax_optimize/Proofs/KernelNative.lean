import bench.repairs.tritonbench_g.softmax_optimize.Proofs.KernelAlgebra
import bench.repairs.tritonbench_g.softmax_optimize.Proofs.FPFrame
import bench.repairs.tritonbench_g.softmax_optimize.Proofs.TensorBinding
import bench.repairs.tritonbench_g.softmax_optimize.Proofs.NativeOrder

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.KernelNative
open VeriTile Triton FP.Structural Binding
open VeriTile.Bench.Repairs.SoftmaxFlaggems.Native (Term Word Instructions Schedule Reductions)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000

def seed (x : Object) (read : Nat → Word) (s : State Term) : State Term :=
  {s with mem := fun r i => if r=x.region then
    Cell.mk TileDType.real (Term.leaf x.region (x.tensor.storage.address i) (read (x.tensor.storage.address i)))
    else s.mem r i}

def rebase (x y : Object) (s : State Term) : State Term :=
  {s with mem := fun r i => if r="x" then s.mem x.region (x.tensor.storageOffset+i)
    else if r="out" then s.mem y.region (y.tensor.storageOffset+i)
    else s.mem r i}

/-- The alternative view names the original Tensor logical element. It keeps
its actual storage address and its provenance leaf, including exceptional words. -/
def logicalView (x y : Object) (read : Nat → Word)
    (row : Fin (rows x.metadata)) (s : State Term) : State Term :=
  let initial := rebase x y (seed x read s)
  {initial with
    mem := fun r i =>
      if active : r="x" ∧ row.val*columns x.metadata≤ i ∧ i<row.val*columns x.metadata+columns x.metadata then
        Cell.mk TileDType.real (Term.leaf x.region (x.tensor.storage.address (x.tensor.storageOffset+i))
          (logicalValue (fun w => w ^^^ 0x80000000) read x row ⟨i-row.val*columns x.metadata,by omega⟩))
      else initial.mem r i}

theorem logicalView_eq (I : Instructions) (S : Schedule) (O : NativeOrder.AddOrder)
    (x y : Object) (h : InputContract x.metadata) (read : Nat → Word)
    (row : Fin (rows x.metadata)) (s : State Term) :
    logicalView x y read row s=rebase x y (seed x read s) := by
  have binding : MemoryBinding (NativeOrder.observe I S O) read x (seed x read s) := by
    intro i _
    simp [NativeOrder.observe,NativeOrder.word,NativeOrder.arrange,seed,VeriTile.Bench.Repairs.SoftmaxFlaggems.Native.eval]
  have cells (column : Fin (columns x.metadata)) :
      read (x.tensor.storage.address (x.tensor.storageOffset+row.val*columns x.metadata+column.val))=
        logicalValue (fun w => w ^^^ 0x80000000) read x row column := by
    have pointer := source_load (NativeOrder.observe I S O) (fun w => w ^^^ 0x80000000)
      read x h (seed x read s) binding row column
    simpa [NativeOrder.observe,NativeOrder.word,NativeOrder.arrange,seed,VeriTile.Bench.Repairs.SoftmaxFlaggems.Native.eval] using pointer.1.trans pointer.2
  have memEq : (logicalView x y read row s).mem=(rebase x y (seed x read s)).mem := by
    funext r i
    change (if active : r="x" ∧ row.val*columns x.metadata≤ i ∧ i<row.val*columns x.metadata+columns x.metadata
      then _ else _) = _
    split
    next active =>
      rcases active with ⟨rfl,lo,hi⟩
      have index : row.val*columns x.metadata+(i-row.val*columns x.metadata)=i := by omega
      have word := cells ⟨i-row.val*columns x.metadata,by omega⟩
      simp only [Fin.val_mk,Nat.add_assoc,index] at word
      simp [rebase,seed]
      exact word.symm
    next _ => rfl
  change {rebase x y (seed x read s) with mem := (logicalView x y read row s).mem}=
    rebase x y (seed x read s)
  rw [memEq]

def project (y : Object) (N row : Nat) (s t : State Term) : State Term :=
  {s with mem := fun r i =>
    if r=y.region ∧ y.tensor.storageOffset+row*N≤ i ∧ i<y.tensor.storageOffset+row*N+N then
      t.mem "out" (i-y.tensor.storageOffset)
    else s.mem r i}

def execute (candidate : Bool) (R : Reductions) (I : Instructions) (S : Schedule) (O : NativeOrder.AddOrder)
    (x y : Object) (read : Nat → Word) (s : State Term) : Option (State Term) :=
  if columns x.metadata=0 then some (seed x read s) else
  let k := if candidate then indexCommutedKernel else kernel
  (FP.Structural.exec (VeriTile.Triton.FP.ProfiledRewrite.engine (algebra R I S O))
    (k "out" "x" (rows x.metadata) (columns x.metadata) (Config.tile (columns x.metadata)))
    (rebase x y (seed x read s))).map
      (project y (columns x.metadata) (s.pids 0) (seed x read s))

/-- Construct successful source and candidate runs from the actual Tensor's
logical/physical load binding. The common result includes every raw memory cell. -/
theorem complete (R : Reductions) (I : Instructions) (S : Schedule) (O : NativeOrder.AddOrder)
    (x y : Object) (h : InputContract x.metadata) (read : Nat → Word)
    (row : Fin (rows x.metadata)) (width : 0<columns x.metadata)
    (s : State Term) (pid : s.pids 0=row.val) :
    ∃ raw result,
      FP.Structural.exec (VeriTile.Triton.FP.ProfiledRewrite.engine (algebra R I S O))
        (kernel "out" "x" (rows x.metadata) (columns x.metadata) (Config.tile (columns x.metadata)))
        (rebase x y (seed x read s))=some raw ∧
      FP.Structural.exec (VeriTile.Triton.FP.ProfiledRewrite.engine (algebra R I S O))
        (indexCommutedKernel "out" "x" (rows x.metadata) (columns x.metadata) (Config.tile (columns x.metadata)))
        (rebase x y (seed x read s))=some raw ∧
      FPExecution.RowFrame "out" (columns x.metadata) row.val (rebase x y (seed x read s)) raw ∧
      execute Bool.false R I S O x y read s=some result ∧
      execute Bool.true R I S O x y read s=some result ∧
      (∀ r i,r≠y.region ∨ i<y.tensor.storageOffset+row.val*columns x.metadata ∨
        y.tensor.storageOffset+(row.val+1)*columns x.metadata≤ i →
        result.mem r i=(seed x read s).mem r i) := by
  obtain ⟨raw,source,frame⟩ := FPExecution.complete_row (VeriTile.Bench.Repairs.SoftmaxFlaggems.Native.algebra R) (equal I S O)
    "out" "x" (rows x.metadata) (columns x.metadata) (Config.tile (columns x.metadata))
    width (Config.tile_positive _) (logicalView x y read row s)
  rw [logicalView_eq I S O x y h read row s] at source frame
  change FP.Structural.exec (VeriTile.Triton.FP.ProfiledRewrite.engine (algebra R I S O)) _ _=some raw at source
  have candidate := (KernelRewrite.execution (algebra R I S O) "out" "x" (rows x.metadata)
    (columns x.metadata) (Config.tile (columns x.metadata)) (rebase x y (seed x read s))).symm.trans source
  refine ⟨raw,project y (columns x.metadata) (s.pids 0) (seed x read s) raw,
    source,candidate,?_,?_,?_,?_⟩
  · simpa [rebase,seed,pid] using frame
  · simp [execute,source,Nat.ne_of_gt width]
  · simp [execute,candidate,Nat.ne_of_gt width]
  · intro r i outside
    have no : ¬(r=y.region ∧ y.tensor.storageOffset+s.pids 0*columns x.metadata≤ i ∧
        i<y.tensor.storageOffset+s.pids 0*columns x.metadata+columns x.metadata) := by
      rw [pid]
      rcases outside with ne|lo|hi <;> simp_all [Nat.add_mul] <;> omega
    simp [project,no]


theorem complete_all (R : Reductions) (I : Instructions) (S : Schedule) (O : NativeOrder.AddOrder)
    (x y : Object) (h : InputContract x.metadata) (read : Nat → Word)
    (row : Fin (rows x.metadata)) (s : State Term) (pid : s.pids 0=row.val) :
    ∃ result,execute Bool.false R I S O x y read s=some result ∧
      execute Bool.true R I S O x y read s=some result ∧
      (∀ r i,r≠y.region ∨ i<y.tensor.storageOffset+row.val*columns x.metadata ∨
        y.tensor.storageOffset+(row.val+1)*columns x.metadata≤ i →
        result.mem r i=(seed x read s).mem r i) := by
  by_cases empty : columns x.metadata=0
  · exact ⟨seed x read s,by simp [execute,empty],by simp [execute,empty],fun _ _ _ => rfl⟩
  · obtain ⟨raw,result,_,_,_,source,candidate,frame⟩ :=
      complete R I S O x y h read row (Nat.pos_of_ne_zero empty) s pid
    exact ⟨result,source,candidate,frame⟩

#axiomsClean logicalView_eq
#axiomsClean complete
#axiomsClean complete_all
end
end VeriTile.Bench.Repairs.SoftmaxOptimize.KernelNative
