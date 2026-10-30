import bench.repairs.tritonbench_g.softmax_triton1.Proofs.FP
import bench.repairs.tritonbench_g.softmax_triton1.Proofs.SourceContract

namespace VeriTile.Bench.Repairs.SoftmaxTriton1.Floating
open VeriTile.Triton FP.Structural
noncomputable section

def access {α} [Inhabited α] : Access (State α) (Cell α) α :=
  ⟨State.mem,fun c => c.read .real⟩

def formula {α} [Inhabited α] (M : Algebra α) (n B : Nat) (xs : Nat → α) (i : Nat) : α :=
  if hi : i<B then rowExpression M n B (fun j => xs j.val) ⟨i,hi⟩ else default

def launchState {α} (x : Tensor) (p : Nat) (s : State α) : State α :=
  {s with pids := (fun ax => if ax=0 then p else 0)
          numPids := (fun ax => if ax=0 then x.rows else 1)}

def evaluate {α} [Inhabited α] (M : Algebra α) (x y : Tensor) (B p : Nat) (s : State α) : Option (State α) :=
  exec (backend M) (softmax_kernel y.region x.region x.offset y.offset x.rowStride x.colStride
    y.rowStride y.colStride x.cols B) (launchState x p s)

theorem source {α} [Inhabited α] (M : Algebra α) : SourceContract access (evaluate M) (formula M) := by
  constructor
  intro x y B p s t values shape injective hB cover hp input run
  have inj : ∀ i j : Fin B, i.val<x.cols → j.val<x.cols → i.val*y.colStride=j.val*y.colStride → i=j := by
    intro i j hi hj eq
    apply Fin.ext
    exact (injective p i.val p j.val (by rwa [shape.rows]) (by rwa [shape.cols])
      (by rwa [shape.rows]) (by rwa [shape.cols]) (by simp [Tensor.index,eq])).2
  constructor
  · intro c hc
    have cB : c<B := lt_of_lt_of_le hc cover
    have value := row_value M y.region x.region x.offset y.offset x.rowStride x.colStride
      y.rowStride y.colStride x.cols B hB inj (launchState x p s) t run ⟨c,cB⟩ hc
    have same : ∀ j : Fin B, j.val<x.cols →
        ((launchState x p s).mem x.region
          (x.offset+(launchState x p s).pids 0*x.rowStride+j.val*x.colStride)).read .real=values p j.val := by
      intro j hj
      exact input p j.val hp hj
    rw [expression_congr M x.cols B _ (fun j => values p j.val) same ⟨c,cB⟩] at value
    have read := congrArg (fun cell : Cell α => cell.read .real) value
    simpa [access,Access.read,formula,Tensor.index,launchState,cB,Cell.read] using read
  · intro r a miss
    exact row_frame M y.region x.region x.offset y.offset x.rowStride x.colStride
      y.rowStride y.colStride x.cols B hB (launchState x p s) t run r a (by
        intro i hi eq
        exact miss p (by simp) i.val (by rwa [shape.cols]) eq)

theorem source_terminates {α} [Inhabited α] (M : Algebra α)
    (x y : Tensor) (B p : Nat) (hB : 0<B) (s : State α) :
    ∃ t, evaluate M x y B p s=some t :=
  row_exec M y.region x.region x.offset y.offset x.rowStride x.colStride
    y.rowStride y.colStride x.cols B hB (launchState x p s)

#axiomsClean source
#axiomsClean source_terminates
end
end VeriTile.Bench.Repairs.SoftmaxTriton1.Floating
