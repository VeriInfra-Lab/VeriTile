import bench.tritonbench_g.llama_ff_triton.Proofs.Logical
import bench.tritonbench_g.int_scaled_matmul.Sources.GridExecution

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.GridExecution (Outside atProgram)
noncomputable section
set_option maxHeartbeats 4000000

theorem byteValues_preserved {α : Type} [Inhabited α] (initial current : State α)
    (h : Outside "OUT" initial current) (bytes : ByteValues initial) : ByteValues current := by
  intro r hr o
  have hn : r≠("OUT" : RegionName) := by
    simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with rfl|rfl|rfl <;> decide
  rw [h r o hn]
  exact bytes r hr o

theorem logical_preserved {α : Type} [Inhabited α] (a : Arithmetic α)
    (g : GateInstructions α) (p : Parameters) (initial current : State α)
    (h : Outside "OUT" initial current) (row column : Nat) :
    logicalSource a p current row column=logicalSource a p initial row column ∧
    logicalNative a g p current row column=logicalNative a g p initial row column := by
  have mem (r : RegionName) (hn : r≠"OUT") : current.mem r=initial.mem r :=
    funext (fun o => h r o hn)
  have hx (t : Nat) (k : Fin 64) : rowX a p current row t k=rowX a p initial row t k := by
    simp only [rowX,mem "A" (by decide)]
  have hr (t k : Nat) : rmsValue a p current t k=rmsValue a p initial t k := by
    simp only [rmsValue,rmsByte,mem "RMS" (by decide)]
  have hs (t : Nat) : rowScaled a p current row t=rowScaled a p initial row t := by
    funext k
    simp only [rowScaled,hx,hr]
  have hw (r : RegionName) (hn : r≠"OUT") (t : Nat) :
      rowW a p current r column t=rowW a p initial r column t := by
    funext k
    simp only [rowW,mem r hn]
  have hwn (r : RegionName) (hn : r≠"OUT") (t : Nat) :
      nativeRowW a p current r column t=nativeRowW a p initial r column t := by
    funext k
    simp only [nativeRowW,mem r hn]
  have ha (native : Bool) (r : RegionName) (hn : r≠"OUT") (t : Nat) :
      rowAcc a native p current r row column t=rowAcc a native p initial r row column t := by
    induction t with
    | zero => rfl
    | succ t ih => simp only [rowAcc,hs,hw r hn,hwn r hn,ih]
  have hm (k : Fin 64) (t : Nat) :
      rowMoment a p current row k t=rowMoment a p initial row k t := by
    induction t with
    | zero => rfl
    | succ t ih => simp only [rowMoment,ih,hx]
  have hn : rowNorm a p current row=rowNorm a p initial row := by
    simp only [rowNorm,hm]
  constructor <;>
    simp only [logicalSource,logicalNative,hn,ha _ "W1" (by decide),ha _ "W3" (by decide)]

#axiomsClean byteValues_preserved
#axiomsClean logical_preserved
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
