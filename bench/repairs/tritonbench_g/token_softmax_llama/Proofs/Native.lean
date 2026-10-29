import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.Grid
import bench.repairs.tritonbench_g.softmax_triton1.Proofs.NativeGraph
namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided.Native
open VeriTile.Triton FP.Structural Grid
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false
abbrev Word := SoftmaxTriton1.NativeGraph.Word
abbrev Node := SoftmaxTriton1.NativeGraph.Node
abbrev Instructions := SoftmaxTriton1.NativeGraph.Instructions

/-- Numerical cells become constant lane functions; discrete metadata stays
unchanged at the actual strided address. No packed metadata snapshot is used. -/
def inject : (d : TileDType)→Value Word d→Value Node d
  | .real,w => SoftmaxTriton1.NativeGraph.atom w
  | .fp32,w => SoftmaxTriton1.NativeGraph.atom w
  | .fp16,w => SoftmaxTriton1.NativeGraph.atom w
  | .bf16,w => SoftmaxTriton1.NativeGraph.atom w
  | .f8e4,w => SoftmaxTriton1.NativeGraph.atom w
  | .f8e5,w => SoftmaxTriton1.NativeGraph.atom w
  | .nat,n => n
  | .int,n => n
  | .bool,n => n
  | .ptr,n => n
  | .blockPtr,n => n
def entry (m : Memory Word) : Memory Node := fun r o =>
  match m r o with | .mk d v => .mk d (inject d v)

@[simp] theorem read_nat (m : Memory Word) (r : RegionName) (o : Nat) :
    ((entry m) r o).read .nat=(m r o).read .nat := by
  unfold entry
  cases m r o with | mk d v => cases d <;> rfl
@[simp] theorem read_fp32 (m : Memory Word) (r : RegionName) (o : Nat) :
    ((entry m) r o).read .fp32=SoftmaxTriton1.NativeGraph.atom ((m r o).read .fp32) := by
  unfold entry
  cases m r o with | mk d v => cases d <;> rfl
@[simp] theorem count_entry (p : Parameters) (m : Memory Word) (pid : PID) :
    count p (state (entry m) pid)=count p (state m pid) := by simp [count,state]
@[simp] theorem start_entry (p : Parameters) (m : Memory Word) (pid : PID) :
    startIndex p (state (entry m) pid)=startIndex p (state m pid) := by simp [startIndex,state]

def rowInput (p : Parameters) (m : Memory Word) (pid : PID) (i : Nat) : Word :=
  (m p.input (pid.2*p.inputHeadStride+(startIndex p (state m pid)+i)*p.inputTokenStride)).read .fp32

/-- Independent ordered register/shuffle/shared-memory graph. The selected
V is the actual compilation's sizePerThread. Default fusion is preserved;
this kernel has no adjacent multiply/add pair to contract. -/
def target (I : Instructions) (V : Nat) (p : Parameters) (m : Memory Word) (pid : PID) (i : Nat) :=
  SoftmaxTriton1.NativeGraph.row I V (count p (state m pid)) p.width (rowInput p m pid) i

theorem value_graph (I : Instructions) (V : Nat) (p : Parameters)
    (m : Memory Word) (pid : PID) (i : Fin p.width) :
    outputValues (SoftmaxTriton1.NativeGraph.algebra I V) .fp32 p (state (entry m) pid) (i,PUnit.unit)
      (SoftmaxTriton1.NativeGraph.owner V p.width i.val)=target I V p m pid i.val := by
  have rows : inputRow (SoftmaxTriton1.NativeGraph.algebra I V) .fp32 p (state (entry m) pid)=
      fun j => if j.1.val < count p (state m pid) then SoftmaxTriton1.NativeGraph.atom (rowInput p m pid j.1.val) else SoftmaxTriton1.NativeGraph.atom SoftmaxTriton1.NativeGraph.negInf := by
    funext j
    simp [inputRow,SoftmaxTriton1.NativeGraph.algebra,rowInput,count,startIndex,state,ofFloat,toFloat]
  have lowered := SoftmaxTriton1.NativeGraph.source_graph_lowering I V (count p (state m pid)) p.width (rowInput p m pid) i.val i.isLt
  unfold target
  rw [← lowered]
  simp only [outputValues,numerators,rows,SoftmaxTriton1.Floating.formula,
    dif_pos i.isLt,SoftmaxTriton1.Floating.rowExpression]
  rfl

#axiomsClean value_graph
#specNonCircular target avoiding [FP.Structural.exec,Grid.run,outputValues]
end
end VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided.Native
