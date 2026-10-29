import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Checked.Binding
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.ForwardValues
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.BackwardValues

namespace VeriTile.Bench.Repairs.KldivOps.Contiguous
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical
set_option autoImplicit false
set_option maxHeartbeats 6000000

private theorem target_ne_prediction : ("T" : RegionName) ≠ "Y" := by decide

structure Adapter (α κ : Type) where
  algebra : Algebra κ
  embed : α → κ
  project : κ → α

def parameters (rows cols : Nat) : Parameters := Host.params rows cols

def physicalState {M α β κ : Type} [Inhabited κ] (ops : Ops M α β)
    (a : Adapter α κ) (memory : M) (prediction target output : View) : State κ where
  mem := fun region i => .mk .real (a.embed (ops.read memory
    ((if region = "Y" then prediction else if region = "T" then target else output).storage.key
      ((if region = "Y" then prediction else if region = "T" then target else output).storageOffset+i))))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d

def forwardRun {κ : Type} [Inhabited κ] (alg : Algebra κ) (rows cols : Nat)
    (flag : Bool) (red : Host.Reduction) (eps : κ) (initial : State κ) :
    {final : State κ // Program.grid alg (parameters rows cols) flag red.code eps initial = some final ∧
      final.mem = Program.gridModel alg (parameters rows cols) flag red.code eps (List.range rows) initial.mem} :=
  ⟨(Program.grid_complete alg (parameters rows cols) flag red.code eps initial (Host.block_positive cols)).choose,
    (Program.grid_complete alg (parameters rows cols) flag red.code eps initial (Host.block_positive cols)).choose_spec⟩

def backwardRun {κ : Type} [Inhabited κ] (alg : Algebra κ) (rows cols : Nat)
    (flag : Bool) (initial : State κ) :
    {final : State κ // Backward.grid alg (parameters rows cols) flag initial = some final ∧
      final.mem = Backward.gridModel alg (parameters rows cols) flag (List.range rows) initial.mem} :=
  ⟨(Backward.grid_complete alg (parameters rows cols) flag initial (Host.block_positive cols)).choose,
    (Backward.grid_complete alg (parameters rows cols) flag initial (Host.block_positive cols)).choose_spec⟩

/-- Store only values computed by the complete typed source execution. -/
def writeView {M α β : Type} (ops : Ops M α β) (output : View) (values : Nat → α) (memory : M) : M :=
  stores ops output.key values (List.range output.numel) memory

def forwardKernel {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (a : Adapter α κ)
    (memory : M) (prediction target output : View) (rows cols : Nat)
    (flag : Bool) (red : Host.Reduction) (eps : α) : M :=
  let run := forwardRun a.algebra rows cols flag red (a.embed eps)
    (physicalState ops a memory prediction target output)
  writeView ops output (fun i => a.project ((run.val.mem "O" i).read .real)) memory

def backwardKernel {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (a : Adapter α κ)
    (memory : M) (target output : View) (rows cols : Nat) (flag : Bool) : M :=
  let run := backwardRun a.algebra rows cols flag (physicalState ops a memory target target output)
  writeView ops output (fun i => a.project ((run.val.mem "O" i).read .real)) memory

theorem writeView_readback {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    (output : View) (dense : Dense output) (positive : output.negative = Bool.false)
    (values : Nat → α) (memory : M) (i : Nat) (hi : i < output.numel) :
    logical ops neg (writeView ops output values memory) output i = values i := by
  simp only [logical,positive,Negation.encode,Bool.false_eq_true,if_false,writeView]
  apply readback ops output.key values _ memory i (List.mem_range.mpr hi)
  · intro j hj same
    rw [dense.unique j i (List.mem_range.mp hj) hi same]
  · intro j hj different
    exact dense.disjoint i j hi (List.mem_range.mp hj) (fun eq => different (congrArg output.key eq.symm))

def point {α κ : Type} (a : Adapter α κ) (flag : Bool) (eps y t : α) : α :=
  a.project (Tile.scalar a.algebra flag (a.embed eps) (a.embed y) (a.embed t))

def row {α κ : Type} (a : Adapter α κ) (cols : Nat) (flag : Bool) (eps : α)
    (ys ts : Nat → α) : α :=
  a.project (FPGraph.row a.algebra cols (Host.blockSize cols) flag (a.embed eps)
    (fun i => a.embed (ys i)) (fun i => a.embed (ts i)))

def forwardExpected {M α β κ : Type} (ops : Ops M α β) (neg : Negation α)
    (a : Adapter α κ) (memory : M) (prediction target : View) (cols : Nat)
    (flag : Bool) (red : Host.Reduction) (eps : α) (i : Nat) : α :=
  if red = .none then point a flag eps (logical ops neg memory prediction i) (logical ops neg memory target i)
  else row a cols flag eps (fun c => logical ops neg memory prediction (i*cols+c))
    (fun c => logical ops neg memory target (i*cols+c))

/-- Complete source readback agrees with entry logical views, including the
actual FP32 block reductions and sequential carried additions. -/
theorem forward_values {M α β κ : Type} [Inhabited κ]
    (ops : Ops M α β) (neg : Negation α) (a : Adapter α κ) (memory : M)
    (p t : Tensor) (prediction target output : View) (rows cols : Nat)
    (flag : Bool) (red : Host.Reduction) (eps : α)
    (admit : PairContract p t) (pb : Binding p prediction) (tb : Binding t target)
    (shape : prediction.shape = [rows,cols]) (outShape : output.shape = red.outputShape rows cols)
    (outDense : Dense output) (positive : output.negative = Bool.false)
    (i : Nat) (hi : i < output.numel) :
    logical ops neg (forwardKernel ops a memory prediction target output rows cols flag red eps) output i =
      forwardExpected ops neg a memory prediction target cols flag red eps i := by
  have sameShape : target.shape = prediction.shape := tb.shape.trans (admit.shape.trans pb.shape.symm)
  have psize : prediction.numel = rows*cols := by simp [View.numel,shape]
  have tsize : target.numel = rows*cols := by simp [View.numel,sameShape,shape]
  unfold forwardKernel
  rw [writeView_readback ops neg output outDense positive _ memory i hi]
  rw [(forwardRun a.algebra rows cols flag red (a.embed eps) (physicalState ops a memory prediction target output)).property.2]
  by_cases mode : red = .none
  · have active : i < rows*cols := by simpa [View.numel,outShape,mode,Host.Reduction.outputShape] using hi
    have width : 0 < cols := by nlinarith
    have hr : i/cols < rows := (Nat.div_lt_iff_lt_mul width).mpr (by simpa [Nat.mul_comm] using active)
    have hc := Nat.mod_lt i width
    have index : i/cols*cols+i%cols = i := by simpa [Nat.mul_comm] using Nat.div_add_mod i cols
    have value := Program.grid_value a.algebra (parameters rows cols) flag (a.embed eps) (List.range rows)
      (physicalState ops a memory prediction target output).mem (i/cols) (i%cols)
      (List.mem_range.mpr hr) (Host.block_positive cols) hc (by change ("Y" : RegionName) ≠ "O"; decide) (by change ("T" : RegionName) ≠ "O"; decide)
    simp only [parameters,Host.params] at value ⊢
    rw [mode]
    simp only [Host.Reduction.code]
    rw [show (i/cols)*cols+i%cols=i from index] at value
    rw [value,Cell.read_same]
    simp only [forwardExpected,mode,if_true,point,physicalState,Cell.read_same,target_ne_prediction,String.reduceEq,if_true,if_false]
    rw [pb.raw ops neg admit.left memory i (by rw [psize];exact active),
      tb.raw ops neg admit.right memory i (by rw [tsize];exact active)]
  · have active : i < rows := by simpa [View.numel,outShape,Host.Reduction.outputShape,mode] using hi
    have code : red.code ≠ 0 := by cases red <;> simp_all [Host.Reduction.code]
    have value := FPGraph.grid_binding a.algebra (parameters rows cols) flag red.code (a.embed eps)
      (List.range rows) (physicalState ops a memory prediction target output).mem i (List.mem_range.mpr active)
      code (by change ("Y" : RegionName) ≠ "O"; decide) (by change ("T" : RegionName) ≠ "O"; decide)
    simp only [parameters,Host.params] at value ⊢
    rw [value,Cell.read_same]
    simp only [forwardExpected,mode,if_false,row,parameters,Host.params]
    congr 1
    apply FPGraph.accumulate_congr
    · intro c hc
      simp only [physicalState,Cell.read_same,target_ne_prediction,String.reduceEq,if_true,if_false]
      rw [pb.raw ops neg admit.left memory (i*cols+c) (by rw [psize];nlinarith)]
    · intro c hc
      simp only [physicalState,Cell.read_same,target_ne_prediction,String.reduceEq,if_true,if_false]
      rw [tb.raw ops neg admit.right memory (i*cols+c) (by rw [tsize];nlinarith)]

theorem forwardExpected_congr {M α β κ : Type}
    (ops : Ops M α β) (neg : Negation α) (a : Adapter α κ) (before after : M)
    (prediction target output : View) (rows cols : Nat) (flag : Bool) (red : Host.Reduction) (eps : α)
    (shape : prediction.shape = [rows,cols]) (sameShape : target.shape = prediction.shape)
    (outShape : output.shape = red.outputShape rows cols)
    (yp : ∀ i, i < prediction.numel → logical ops neg after prediction i = logical ops neg before prediction i)
    (tp : ∀ i, i < target.numel → logical ops neg after target i = logical ops neg before target i)
    (i : Nat) (hi : i < output.numel) :
    forwardExpected ops neg a after prediction target cols flag red eps i =
      forwardExpected ops neg a before prediction target cols flag red eps i := by
  have psize : prediction.numel = rows*cols := by simp [View.numel,shape]
  have tsize : target.numel = rows*cols := by simp [View.numel,sameShape,shape]
  by_cases mode : red = .none
  · have active : i < rows*cols := by simpa [View.numel,outShape,mode,Host.Reduction.outputShape] using hi
    simp only [forwardExpected,mode,if_true]
    rw [yp i (by rw [psize];exact active),tp i (by rw [tsize];exact active)]
  · have active : i < rows := by simpa [View.numel,outShape,Host.Reduction.outputShape,mode] using hi
    simp only [forwardExpected,mode,if_false,row]
    congr 1
    apply FPGraph.accumulate_congr
    · intro c hc;rw [yp (i*cols+c) (by rw [psize];nlinarith)]
    · intro c hc;rw [tp (i*cols+c) (by rw [tsize];nlinarith)]

def derivative {α κ : Type} (a : Adapter α κ) (flag : Bool) (t : α) : α :=
  a.project (Backward.scalar a.algebra flag (a.embed t))

theorem backward_values {M α β κ : Type} [Inhabited κ]
    (ops : Ops M α β) (neg : Negation α) (a : Adapter α κ) (memory : M)
    (d t : Tensor) (target output : View) (rows cols : Nat) (flag : Bool)
    (admit : PairContract d t) (db : Binding d output) (tb : Binding t target)
    (shape : output.shape = [rows,cols]) (i : Nat) (hi : i < output.numel) :
    logical ops neg (backwardKernel ops a memory target output rows cols flag) output i =
      derivative a flag (logical ops neg memory target i) := by
  have sameShape : target.shape = output.shape := tb.shape.trans (admit.shape.trans db.shape.symm)
  have osize : output.numel = rows*cols := by simp [View.numel,shape]
  have tsize : target.numel = output.numel := by simp [View.numel,sameShape]
  have active : i < rows*cols := by simpa only [osize] using hi
  have width : 0 < cols := by nlinarith
  have hr : i/cols < rows := (Nat.div_lt_iff_lt_mul width).mpr (by simpa [Nat.mul_comm] using active)
  have hc := Nat.mod_lt i width
  have index : i/cols*cols+i%cols = i := by simpa [Nat.mul_comm] using Nat.div_add_mod i cols
  unfold backwardKernel
  rw [writeView_readback ops neg output (db.contiguous admit.left.contiguous)
    (db.negative.trans admit.left.noNeg) _ memory i hi]
  rw [(backwardRun a.algebra rows cols flag (physicalState ops a memory target target output)).property.2]
  have value := Backward.grid_value a.algebra (parameters rows cols) flag (List.range rows)
    (physicalState ops a memory target target output).mem (i/cols) (i%cols)
    (List.mem_range.mpr hr) (Host.block_positive cols) hc
    (by change ("T" : RegionName) ≠ "O"; decide)
  simp only [parameters,Host.params,index] at value ⊢
  rw [value,Cell.read_same]
  simp only [physicalState,Cell.read_same,target_ne_prediction,String.reduceEq,if_true,if_false,derivative]
  rw [tb.raw ops neg admit.right memory i (by rw [tsize];exact hi)]

/-- Full bytes outside the directly written Tensor, including partial tails. -/
theorem writeView_bytes {M α β : Type} (ops : Ops M α β) (output : View)
    (values : Nat → α) (memory : M) (owner address : Nat)
    (outside : ∀ i, i < output.numel → ¬(output.key i).covers owner address) :
    ops.observe (writeView ops output values memory) owner address = ops.observe memory owner address := by
  apply bytes_frame
  intro i hi
  exact outside i (List.mem_range.mp hi)

#axiomsClean backward_values
#axiomsClean writeView_bytes
#axiomsClean forward_values
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Contiguous
