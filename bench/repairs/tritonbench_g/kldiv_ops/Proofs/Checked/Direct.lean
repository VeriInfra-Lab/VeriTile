import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Checked.Execution

namespace VeriTile.Bench.Repairs.KldivOps.Contiguous
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open KldivTriton.Grid (runRows rowJobs scheduledJobs stores_congr runRows_eq_stores)
noncomputable section
set_option autoImplicit false
set_option maxHeartbeats 8000000

private theorem target_ne_prediction : ("T" : RegionName) ≠ "Y" := by decide

def width (red : Host.Reduction) (cols : Nat) := if red = .none then cols else 1

def forwardRowValue {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (a : Adapter α κ)
    (memory : M) (prediction target output : View) (rows cols : Nat)
    (flag : Bool) (red : Host.Reduction) (eps : α) (r c : Nat) : α :=
  a.project ((Program.rowModel a.algebra (parameters rows cols) flag red.code r (a.embed eps)
    (physicalState ops a memory prediction target output).mem "O" (r*width red cols+c)).read .real)

/-- The value used by a physical row step is computed by a complete execution
of that row's actual typed source, including every chunk and the final store. -/
theorem forward_row_executes {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (a : Adapter α κ)
    (memory : M) (prediction target output : View) (rows cols : Nat)
    (flag : Bool) (red : Host.Reduction) (eps : α) (r : Nat) :
    ∃ final,
      run (engine a.algebra) (forward (parameters rows cols) flag red.code).surfaceBody
        (Tile.bindEpsilon (a.embed eps) (Program.atRow (physicalState ops a memory prediction target output) r)) = some final ∧
      ∀ c, a.project ((final.mem "O" (r*width red cols+c)).read .real) =
        forwardRowValue ops a memory prediction target output rows cols flag red eps r c := by
  obtain ⟨final,exec,values⟩ := Program.row_complete a.algebra (parameters rows cols) flag red.code (a.embed eps)
    (Program.atRow (physicalState ops a memory prediction target output) r) (Host.block_positive cols)
  refine ⟨final,exec,?_⟩
  intro c
  rw [values]
  rfl

theorem forward_row_value {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (neg : Negation α)
    (a : Adapter α κ) (memory : M) (p t : Tensor) (prediction target output : View)
    (rows cols : Nat) (flag : Bool) (red : Host.Reduction) (eps : α)
    (admit : PairContract p t) (pb : Binding p prediction) (tb : Binding t target)
    (shape : prediction.shape = [rows,cols]) (r c : Nat) (hr : r < rows) (hc : c < width red cols) :
    forwardRowValue ops a memory prediction target output rows cols flag red eps r c =
      forwardExpected ops neg a memory prediction target cols flag red eps (r*width red cols+c) := by
  have sameShape : target.shape = prediction.shape := tb.shape.trans (admit.shape.trans pb.shape.symm)
  have psize : prediction.numel = rows*cols := by simp [View.numel,shape]
  have tsize : target.numel = rows*cols := by simp [View.numel,sameShape,shape]
  unfold forwardRowValue
  by_cases mode : red = .none
  · have col : c < cols := by simpa [width,mode] using hc
    have value := Program.row_value a.algebra (parameters rows cols) flag (a.embed eps) r c
      (physicalState ops a memory prediction target output).mem (Host.block_positive cols) col
      (by change ("Y" : RegionName) ≠ "O";decide) (by change ("T" : RegionName) ≠ "O";decide)
    simp only [parameters,Host.params] at value ⊢
    simp only [mode,Host.Reduction.code,width,if_true]
    rw [value,Cell.read_same]
    simp only [forwardExpected,mode,if_true,point,width,physicalState,Cell.read_same,
      target_ne_prediction,String.reduceEq,if_true,if_false]
    rw [pb.raw ops neg admit.left memory (r*cols+c) (by rw [psize];nlinarith),
      tb.raw ops neg admit.right memory (r*cols+c) (by rw [tsize];nlinarith)]
  · have zero : c = 0 := by simpa [width,mode] using hc
    subst c
    have code : red.code ≠ 0 := by cases red <;> simp_all [Host.Reduction.code]
    have value := FPGraph.row_binding a.algebra (parameters rows cols) flag red.code r (a.embed eps)
      (physicalState ops a memory prediction target output).mem code
    simp only [parameters,Host.params] at value ⊢
    simp only [width,mode,if_false,Nat.mul_one,Nat.add_zero]
    rw [value,Cell.read_same]
    simp only [forwardExpected,mode,if_false,row,width,Nat.mul_one,Nat.add_zero,parameters,Host.params]
    congr 1
    apply FPGraph.accumulate_congr
    · intro c hc
      simp only [physicalState,Cell.read_same,target_ne_prediction,String.reduceEq,if_true,if_false]
      rw [pb.raw ops neg admit.left memory (r*cols+c) (by rw [psize];nlinarith)]
    · intro c hc
      simp only [physicalState,Cell.read_same,target_ne_prediction,String.reduceEq,if_true,if_false]
      rw [tb.raw ops neg admit.right memory (r*cols+c) (by rw [tsize];nlinarith)]

theorem forward_physical_rows {M α β κ : Type} [Inhabited κ]
    (ops : Ops M α β) (neg : Negation α) (a : Adapter α κ) (initial : M)
    (p t : Tensor) (prediction target output : View) (rows cols : Nat)
    (flag : Bool) (red : Host.Reduction) (eps : α)
    (admit : PairContract p t) (pb : Binding p prediction) (tb : Binding t target)
    (shape : prediction.shape = [rows,cols]) (outShape : output.shape = red.outputShape rows cols)
    (backed : output.backed) (ypSeparate : prediction.storage.separate output.storage)
    (tpSeparate : target.storage.separate output.storage) :
    runRows ops (fun rc => output.key (rc.1*width red cols+rc.2))
      (fun s r c => forwardRowValue ops a s prediction target output rows cols flag red eps r c)
      (width red cols) (List.range rows) initial =
    stores ops (fun rc => output.key (rc.1*width red cols+rc.2))
      (fun rc => forwardRowValue ops a initial prediction target output rows cols flag red eps rc.1 rc.2)
      (scheduledJobs (width red cols) (List.range rows)) initial := by
  have size : output.numel = rows*width red cols := by
    by_cases mode : red = .none <;> simp [View.numel,outShape,Host.Reduction.outputShape,width,mode]
  have sameShape : target.shape = prediction.shape := tb.shape.trans (admit.shape.trans pb.shape.symm)
  have stable : ∀ (done : List (Nat×Nat)), (∀ rc ∈ done, rc.1 < rows ∧ rc.2 < width red cols) →
      ∀ r c, r < rows → c < width red cols →
      forwardRowValue ops a (stores ops (fun rc => output.key (rc.1*width red cols+rc.2))
        (fun rc => forwardRowValue ops a initial prediction target output rows cols flag red eps rc.1 rc.2) done initial)
        prediction target output rows cols flag red eps r c =
      forwardRowValue ops a initial prediction target output rows cols flag red eps r c := by
    intro done valid r c hr hc
    have active : ∀ rc ∈ done, rc.1*width red cols+rc.2 < output.numel := by
      intro rc hrc
      obtain ⟨row,col⟩ := valid rc hrc
      rw [size];nlinarith
    rw [forward_row_value ops neg a _ p t prediction target output rows cols flag red eps admit pb tb shape r c hr hc,
      forward_row_value ops neg a initial p t prediction target output rows cols flag red eps admit pb tb shape r c hr hc]
    exact forwardExpected_congr ops neg a initial _ prediction target output rows cols flag red eps shape sameShape outShape
      (input_after_prefix ops neg prediction output pb.backed backed ypSeparate _ _ done active initial)
      (input_after_prefix ops neg target output tb.backed backed tpSeparate _ _ done active initial)
      _ (by rw [size];nlinarith)
  simpa only [stores,List.nil_append] using runRows_eq_stores ops _ _ rows (width red cols) initial stable
    (List.range rows) (fun r hr => List.mem_range.mp hr) [] (by simp)

theorem stores_map {M α β ι μ : Type} (ops : Ops M α β) (key : μ → Key)
    (values : μ → α) (index : ι → μ) (jobs : List ι) (s : M) :
    stores ops (fun i => key (index i)) (fun i => values (index i)) jobs s =
      stores ops key values (jobs.map index) s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons i rest ih => exact ih _

theorem scheduled_flat (rows cols : Nat) :
    (scheduledJobs cols (List.range rows)).map (fun rc => rc.1*cols+rc.2) = List.range (rows*cols) := by
  induction rows with
  | zero => simp [scheduledJobs]
  | succ rows ih =>
    simp only [List.range_succ,scheduledJobs,List.flatMap_append,List.flatMap_singleton,List.map_append] at ih ⊢
    rw [ih]
    simp only [rowJobs,List.map_map,Function.comp_apply,Nat.succ_mul,List.range_add]
    rfl

/-- Physical row execution equals the public source-derived store graph.
The full source computes each row against evolving memory. The proof uses
only checked input binding and fresh output separation. -/
theorem forward_complete {M α β κ : Type} [Inhabited κ]
    (ops : Ops M α β) (neg : Negation α) (a : Adapter α κ) (initial : M)
    (p t : Tensor) (prediction target output : View) (rows cols : Nat)
    (flag : Bool) (red : Host.Reduction) (eps : α)
    (admit : PairContract p t) (pb : Binding p prediction) (tb : Binding t target)
    (shape : prediction.shape = [rows,cols]) (outShape : output.shape = red.outputShape rows cols)
    (dense : Dense output) (positive : output.negative = Bool.false)
    (backed : output.backed) (ypSeparate : prediction.storage.separate output.storage)
    (tpSeparate : target.storage.separate output.storage) :
    runRows ops (fun rc => output.key (rc.1*width red cols+rc.2))
      (fun s r c => forwardRowValue ops a s prediction target output rows cols flag red eps r c)
      (width red cols) (List.range rows) initial =
      forwardKernel ops a initial prediction target output rows cols flag red eps := by
  rw [forward_physical_rows ops neg a initial p t prediction target output rows cols flag red eps admit pb tb shape outShape backed ypSeparate tpSeparate]
  let values := forwardExpected ops neg a initial prediction target cols flag red eps
  have rowsEq : stores ops (fun rc => output.key (rc.1*width red cols+rc.2))
      (fun rc => forwardRowValue ops a initial prediction target output rows cols flag red eps rc.1 rc.2)
      (scheduledJobs (width red cols) (List.range rows)) initial =
      stores ops (fun rc => output.key (rc.1*width red cols+rc.2))
        (fun rc => values (rc.1*width red cols+rc.2)) (scheduledJobs (width red cols) (List.range rows)) initial := by
    apply stores_congr
    intro rc member
    obtain ⟨hr,hc⟩ := (KldivTriton.Grid.scheduled_member _ _ _ _).mp member
    exact forward_row_value ops neg a initial p t prediction target output rows cols flag red eps admit pb tb shape
      rc.1 rc.2 (List.mem_range.mp hr) hc
  rw [rowsEq,stores_map,scheduled_flat]
  have size : output.numel = rows*width red cols := by
    by_cases mode : red = .none <;> simp [View.numel,outShape,Host.Reduction.outputShape,width,mode]
  rw [←size]
  unfold forwardKernel writeView
  symm
  apply stores_congr
  intro i hi
  have active := List.mem_range.mp hi
  have h := forward_values ops neg a initial p t prediction target output rows cols flag red eps admit pb tb shape outShape dense positive i active
  unfold forwardKernel at h
  rw [writeView_readback ops neg output dense positive _ initial i active] at h
  exact h

def backwardRowValue {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (a : Adapter α κ)
    (memory : M) (target output : View) (rows cols : Nat) (flag : Bool) (r c : Nat) : α :=
  a.project ((Backward.loopModel a.algebra (parameters rows cols) flag r 0
    (physicalState ops a memory target target output).mem "O" (r*cols+c)).read .real)

theorem backward_row_executes {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (a : Adapter α κ)
    (memory : M) (target output : View) (rows cols : Nat) (flag : Bool) (r : Nat) :
    ∃ final,
      run (engine a.algebra) (backward (parameters rows cols) flag).surfaceBody
        (Program.atRow (physicalState ops a memory target target output) r) = some final ∧
      ∀ c, a.project ((final.mem "O" (r*cols+c)).read .real) =
        backwardRowValue ops a memory target output rows cols flag r c := by
  obtain ⟨final,exec,values⟩ := Backward.row_complete a.algebra (parameters rows cols) flag
    (Program.atRow (physicalState ops a memory target target output) r) (Host.block_positive cols)
  refine ⟨final,exec,?_⟩
  intro c
  rw [values]
  rfl

theorem backward_row_value {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (neg : Negation α)
    (a : Adapter α κ) (memory : M) (d t : Tensor) (target output : View)
    (rows cols : Nat) (flag : Bool) (admit : PairContract d t)
    (db : Binding d output) (tb : Binding t target) (shape : output.shape = [rows,cols])
    (r c : Nat) (hr : r < rows) (hc : c < cols) :
    backwardRowValue ops a memory target output rows cols flag r c =
      derivative a flag (logical ops neg memory target (r*cols+c)) := by
  have size : target.numel = rows*cols := by simp [View.numel,tb.shape,admit.shape,←db.shape,shape]
  have value := Backward.loop_value a.algebra (parameters rows cols) flag r 0 c
    (physicalState ops a memory target target output).mem (Host.block_positive cols) (by omega) hc
    (by change ("T" : RegionName) ≠ "O";decide)
  unfold backwardRowValue
  simp only [parameters,Host.params] at value ⊢
  rw [value,Cell.read_same]
  simp only [derivative,physicalState,Cell.read_same,target_ne_prediction,String.reduceEq,if_true,if_false]
  rw [tb.raw ops neg admit.right memory (r*cols+c) (by rw [size];nlinarith)]

/-- Complete direct backward row execution and physical stores. Separation
comes from the actual checked backing guard, including empty matrices. -/
theorem backward_complete {M α β κ : Type} [Inhabited κ]
    (ops : Ops M α β) (neg : Negation α) (a : Adapter α κ) (initial : M)
    (d t g : Tensor) (target output : View) (rows cols : Nat) (flag : Bool)
    (admit : BackwardContract d t g) (db : Binding d output) (tb : Binding t target)
    (shape : output.shape = [rows,cols]) :
    runRows ops (fun rc => output.key (rc.1*cols+rc.2))
      (fun s r c => backwardRowValue ops a s target output rows cols flag r c)
      cols (List.range rows) initial = backwardKernel ops a initial target output rows cols flag := by
  have size : output.numel = rows*cols := by simp [View.numel,shape]
  have targetSize : target.numel = rows*cols := by simp [View.numel,tb.shape,admit.shape,←db.shape,shape]
  have stable : ∀ (done : List (Nat×Nat)), (∀ rc ∈ done, rc.1 < rows ∧ rc.2 < cols) →
      ∀ r c, r < rows → c < cols →
      backwardRowValue ops a (stores ops (fun rc => output.key (rc.1*cols+rc.2))
        (fun rc => backwardRowValue ops a initial target output rows cols flag rc.1 rc.2) done initial)
        target output rows cols flag r c = backwardRowValue ops a initial target output rows cols flag r c := by
    intro done valid r c hr hc
    rw [backward_row_value ops neg a _ d t target output rows cols flag admit.toPairContract db tb shape r c hr hc,
      backward_row_value ops neg a initial d t target output rows cols flag admit.toPairContract db tb shape r c hr hc]
    congr 1
    have nonempty : d.metadata.numel ≠ 0 := by rw [←db.toViewBinding.numel,size];nlinarith
    apply input_after_prefix ops neg target output tb.backed db.backed
      (KldivTriton.Host.separate_symm (db.toViewBinding.separate tb.toViewBinding (admit.separate nonempty).1))
    · intro rc hrc
      obtain ⟨row,col⟩ := valid rc hrc
      rw [size];nlinarith
    · rw [targetSize];nlinarith
  have direct := runRows_eq_stores ops (fun rc => output.key (rc.1*cols+rc.2))
    (fun s r c => backwardRowValue ops a s target output rows cols flag r c) rows cols initial stable
    (List.range rows) (fun r hr => List.mem_range.mp hr) [] (by simp)
  simp only [stores,List.nil_append] at direct
  rw [direct]
  let values := fun i => derivative a flag (logical ops neg initial target i)
  have rowsEq : stores ops (fun rc => output.key (rc.1*cols+rc.2))
      (fun rc => backwardRowValue ops a initial target output rows cols flag rc.1 rc.2)
      (scheduledJobs cols (List.range rows)) initial =
      stores ops (fun rc => output.key (rc.1*cols+rc.2)) (fun rc => values (rc.1*cols+rc.2))
        (scheduledJobs cols (List.range rows)) initial := by
    apply stores_congr
    intro rc member
    obtain ⟨hr,hc⟩ := (KldivTriton.Grid.scheduled_member _ _ _ _).mp member
    exact backward_row_value ops neg a initial d t target output rows cols flag admit.toPairContract db tb shape
      rc.1 rc.2 (List.mem_range.mp hr) hc
  rw [rowsEq,stores_map,scheduled_flat,←size]
  unfold backwardKernel writeView
  symm
  apply stores_congr
  intro i hi
  have active := List.mem_range.mp hi
  have h := backward_values ops neg a initial d t target output rows cols flag admit.toPairContract db tb shape i active
  unfold backwardKernel at h
  rw [writeView_readback ops neg output (db.contiguous admit.left.contiguous) (db.negative.trans admit.left.noNeg)
    _ initial i active] at h
  exact h

#axiomsClean forward_row_executes
#axiomsClean forward_row_value
#axiomsClean forward_physical_rows
#axiomsClean forward_complete
#axiomsClean backward_row_executes
#axiomsClean backward_row_value
#axiomsClean backward_complete
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Contiguous
