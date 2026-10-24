import bench.repairs.tritonbench_g.softmax_optimize.Proofs.TensorBinding
import bench.repairs.tritonbench_g.softmax_optimize.Proofs.SourceIO

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.KernelReal
open VeriTile.Triton Binding
open VeriTile.Bench.TritonBenchG.SoftmaxOptimize
open scoped BigOperators
noncomputable section

def rebase (x y : Object) (s : BlockState) : BlockState :=
  {s with mem := fun r i =>
    if r="x" then s.mem x.region (x.tensor.storageOffset+i)
    else if r="out" then s.mem y.region (y.tensor.storageOffset+i)
    else s.mem r i}

def project (y : Object) (N row : Nat) (s t : BlockState) : BlockState :=
  {s with mem := fun r i =>
    if r=y.region ∧ y.tensor.storageOffset+row*N≤ i ∧ i<y.tensor.storageOffset+row*N+N then
      t.mem "out" (i-y.tensor.storageOffset)
    else s.mem r i}

def execute (x y : Object) (s : BlockState) : Option BlockState :=
  if columns x.metadata=0 then some s else
  (exec (kernel "out" "x" (rows x.metadata) (columns x.metadata)
    (Config.tile (columns x.metadata))).eraseDType.toAlgKernel (rebase x y s)).map
      (project y (columns x.metadata) s.pid s)

def reference (input : Fin M → Fin N → ℝ) (row : Fin M) (column : Fin N) : ℝ :=
  Real.exp (input row column)/(∑ j : Fin N,Real.exp (input row j))

theorem complete (x y : Object) (h : InputContract x.metadata)
    (read : Nat → ℝ) (s : BlockState) (binding : MemoryBinding BlockState.readMem read x s)
    (row : Fin (rows x.metadata)) (pid : s.pid=row.val) (width : 0<columns x.metadata) :
    ∃ t,execute x y s=some t ∧
      (∀ column : Fin (columns x.metadata),
        t.readMem y.region (y.tensor.storageOffset+row.val*columns x.metadata+column.val)=
          reference (logicalValue Neg.neg read x) row column) ∧
      (∀ r i,r≠y.region ∨ i<y.tensor.storageOffset+row.val*columns x.metadata ∨
        y.tensor.storageOffset+(row.val+1)*columns x.metadata≤ i → t.mem r i=s.mem r i) := by
  obtain ⟨u,run,values,_,_,_,_⟩ := SourceIO.selected_tile_io "out" "x"
    (rows x.metadata) (columns x.metadata) (by decide) width (rebase x y s)
  refine ⟨project y (columns x.metadata) s.pid s u,by simp [execute,run,Nat.ne_of_gt width],?_,?_⟩
  · intro column
    have active : y.region=y.region ∧
        y.tensor.storageOffset+s.pid*columns x.metadata≤y.tensor.storageOffset+row.val*columns x.metadata+column.val ∧
        y.tensor.storageOffset+row.val*columns x.metadata+column.val<
          y.tensor.storageOffset+s.pid*columns x.metadata+columns x.metadata := by
      rw [pid]
      exact ⟨rfl,by omega,by have hc:=column.isLt;omega⟩
    have index : y.tensor.storageOffset+row.val*columns x.metadata+column.val-y.tensor.storageOffset=
        row.val*columns x.metadata+column.val := by omega
    have value := values column
    rw [softmaxOptimizeFullSpec_eq_div] at value
    have loads (j : Fin (columns x.metadata)) :
        (rebase x y s).readMem "x" ((rebase x y s).pid*columns x.metadata+j.val)=
          logicalValue Neg.neg read x row j := by
      have link := source_load BlockState.readMem Neg.neg read x h s binding row j
      simpa [rebase,BlockState.readMem,BlockState.pid,pid,Nat.add_assoc] using link.1.trans link.2
    rw [←Fin.sum_univ_eq_sum_range (fun j => Real.exp ((rebase x y s).readMem "x"
      ((rebase x y s).pid*columns x.metadata+j)))] at value
    simp_rw [loads] at value
    simpa [project,BlockState.readMem,active,index,rebase,BlockState.pid,pid,reference] using value
  · intro r i outside
    have no : ¬(r=y.region ∧ y.tensor.storageOffset+s.pid*columns x.metadata≤ i ∧
        i<y.tensor.storageOffset+s.pid*columns x.metadata+columns x.metadata) := by
      rw [pid]
      rcases outside with ne|lo|hi <;> simp_all [Nat.add_mul,Nat.one_mul] <;> omega
    simp [project,no]


theorem complete_all (x y : Object) (h : InputContract x.metadata)
    (read : Nat → ℝ) (s : BlockState) (binding : MemoryBinding BlockState.readMem read x s)
    (row : Fin (rows x.metadata)) (pid : s.pid=row.val) :
    ∃ t,execute x y s=some t ∧
      (∀ column : Fin (columns x.metadata),
        t.readMem y.region (y.tensor.storageOffset+row.val*columns x.metadata+column.val)=
          reference (logicalValue Neg.neg read x) row column) ∧
      (∀ r i,r≠y.region ∨ i<y.tensor.storageOffset+row.val*columns x.metadata ∨
        y.tensor.storageOffset+(row.val+1)*columns x.metadata≤ i → t.mem r i=s.mem r i) := by
  by_cases empty : columns x.metadata=0
  · refine ⟨s,by simp [execute,empty],?_,fun _ _ _ => rfl⟩
    intro column
    exact False.elim (by have hc:=column.isLt;omega)
  · exact complete x y h read s binding row pid (Nat.pos_of_ne_zero empty)

#specNonCircular reference avoiding [execute,exec,kernel]
#axiomsClean complete
#axiomsClean complete_all
end
end VeriTile.Bench.Repairs.SoftmaxOptimize.KernelReal
