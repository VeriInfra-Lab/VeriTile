import bench.repairs.tritonbench_g.softmax_triton1.Proofs.KernelMath
import bench.repairs.tritonbench_g.softmax_triton1.Proofs.Wrapper

namespace VeriTile.Bench.Repairs.SoftmaxTriton1.RealModel
open VeriTile.Triton
noncomputable section

def access : Access BlockState MemCell ℝ where
  cells := BlockState.mem
  decode := fun c => match c.readAs .real with | some (some v) => v | _ => 0

def launchState (x : Tensor) (p : Nat) (s : BlockState) : BlockState :=
  {s with pids := (fun ax => if ax=0 then p else 0)
          numPids := (fun ax => if ax=0 then x.rows else 1)}

def evaluate (x y : Tensor) (B p : Nat) (s : BlockState) : Option BlockState :=
  exec (softmax_kernel y.region x.region x.offset y.offset x.rowStride x.colStride
    y.rowStride y.colStride x.cols B).toAlgKernel (launchState x p s)

theorem source : SourceContract access evaluate formula := by
  constructor
  intro x y B p s t values shape injective hB cover hp input run
  have inj : RowInjective x.cols B y.colStride := by
    intro i j hi hj eq
    apply Fin.ext
    exact (injective p i.val p j.val (by rwa [shape.rows]) (by rwa [shape.cols])
      (by rwa [shape.rows]) (by rwa [shape.cols]) (by simp [Tensor.index,eq])).2
  constructor
  · intro c hc
    have cB : c<B := lt_of_lt_of_le hc cover
    let i : Fin B := ⟨c,cB⟩
    have value := row_value y.region x.region x.offset y.offset x.rowStride x.colStride
      y.rowStride y.colStride x.cols B hB inj (launchState x p s) t run i hc
    have same : ∀ j : Fin B, j.val<x.cols →
        (launchState x p s).readMem x.region
          (x.offset+(launchState x p s).pid*x.rowStride+j.val*x.colStride)=values p j.val := by
      intro j hj
      exact input p j.val hp hj
    rw [VeriTile.Bench.TritonBenchG.SoftmaxTriton1.softmaxSpec_congr x.cols B _
      (fun j => values p j.val) same i] at value
    rw [VeriTile.Bench.TritonBenchG.SoftmaxTriton1.softmaxSpec_eq_formula x.cols B hB _ i hc] at value
    rw [padded_sum_eq x.cols B cover (fun j => Real.exp (values p j))] at value
    simpa [access,Access.read,BlockState.readMem,formula,Tensor.index,launchState,BlockState.pid_eq,i] using value
  · intro r a miss
    exact row_frame y.region x.region x.offset y.offset x.rowStride x.colStride
      y.rowStride y.colStride x.cols B hB (launchState x p s) t run r a (by
        intro i hi eq
        exact miss p (by simp) i.val (by rwa [shape.cols]) eq)

theorem source_terminates (x y : Tensor) (B p : Nat) (hB : 0<B) (s : BlockState) :
    ∃ t, evaluate x y B p s=some t :=
  row_exec y.region x.region x.offset y.offset x.rowStride x.colStride
    y.rowStride y.colStride x.cols B hB (launchState x p s)

theorem source_safe (layout : Layout) (x y : Tensor) (B p : Nat)
    (hx : layout.registered x) (hy : layout.registered y)
    (shape : SameShape x y) (hB : 0<B) (hp : p<x.rows) (s : BlockState) :
    Kernel.TraceSafe (fun r => (layout.buffer r).capacity/4)
      (softmax_kernel y.region x.region x.offset y.offset x.rowStride x.colStride
        y.rowStride y.colStride x.cols B).toAlgKernel (launchState x p s) := by
  apply row_safe
  · exact hB
  · intro i hi
    have bound := (layout.legal x hx).2.2.2.2.2.2.2 p i.val hp hi
    change x.offset+p*x.rowStride+i.val*x.colStride < (layout.buffer x.region).capacity/4
    dsimp [Tensor.index] at bound
    omega
  · intro i hi
    have bound := (layout.legal y hy).2.2.2.2.2.2.2 p i.val (by rwa [shape.rows]) (by rwa [shape.cols])
    change y.offset+p*y.rowStride+i.val*y.colStride < (layout.buffer y.region).capacity/4
    dsimp [Tensor.index] at bound
    omega

#axiomsClean source
#axiomsClean source_safe
#axiomsClean source_terminates
end
end VeriTile.Bench.Repairs.SoftmaxTriton1.RealModel
