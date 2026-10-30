import bench.repairs.tritonbench_g.iv_dependent_matmul_dot_shape.IO
import bench.repairs.tritonbench_g.iv_dependent_matmul_dot_shape.Proofs.GridSafety

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape.IOBinding
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Whole
open IvDependentMatmul
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Finite IO addresses cover exactly each original contiguous extent. -/
theorem window (rows columns o : Nat) (ho : o<rows*columns) :
    ∃ i : Fin rows, ∃ j : Fin columns, i.val*max columns 1+j.val=o := by
  have pos : 0<columns := by nlinarith
  refine ⟨⟨o/columns,(Nat.div_lt_iff_lt_mul pos).mpr ho⟩,
    ⟨o%columns,Nat.mod_lt o pos⟩,?_⟩
  simp only [max_eq_left (by omega : 1≤columns), Nat.mul_comm (o/columns), Nat.div_add_mod]

theorem windows (M : Algebra α) (graph : Graph) (g : Kernel.Geometry)
    (vals : ChannelValues α (IO.inputs g)) (bounds : RegionName → Nat)
    (hb : (IO.matmulIO M graph g).InBounds bounds vals) : SourceSafety.Windows g.request bounds where
  input := by
    intro o ho
    obtain ⟨i,k,h⟩ := window g.rows g.inner o ho
    have z : i.val*max g.inner 1+k.val<bounds g.A := hb.2.1 (0 : Fin 2) (i,k,()) trivial
    simpa only [h] using z
  weight := by
    intro o ho
    obtain ⟨k,j,h⟩ := window g.inner g.columns o ho
    have z : k.val*max g.columns 1+j.val<bounds g.B := hb.2.1 (1 : Fin 2) (k,j,()) trivial
    simpa only [h] using z
  output := by
    intro o ho
    obtain ⟨i,j,h⟩ := window g.rows g.columns o ho
    have z : i.val*max g.columns 1+j.val<bounds g.C := hb.2.2.1 (0 : Fin 1) (i,j,()) trivial
    simpa only [h] using z

theorem outside (M : Algebra α) (graph : Graph) (g : Kernel.Geometry)
    (vals : ChannelValues α (IO.inputs g)) (r : RegionName) (o : Nat)
    (h : (IO.matmulIO M graph g).Outside vals r o) : ¬Footprint g.request r o := by
  rintro ⟨rfl,ho⟩
  obtain ⟨i,j,eq⟩ := window g.rows g.columns o ho
  have hh : g.C≠g.C ∨ o≠i.val*max g.columns 1+j.val := h.1 (0 : Fin 1) (i,j,()) trivial
  simpa only [eq,ne_eq,not_true_eq_false,or_self] using hh

/-- Actual launch interpreter agrees definitionally with the retained grid. -/
theorem schedule_run (M : Algebra α) (graph : Graph) (g : Kernel.Geometry)
    (ids : List Nat) (s : State α) :
    runSchedule (ids.map (IO.invocation M graph g)) s=Kernel.runPrograms M graph g.request ids s := by
  induction ids generalizing s with
  | nil => rfl
  | cons pid tail ih =>
    change (FP.Structural.exec M (Kernel.quotation g.schedule graph g.request.parameters)
      {s with pids:=fun ax => if ax=0 then pid else 0}).bind
        (runSchedule (tail.map (IO.invocation M graph g))) =
      (FP.Structural.exec M (Kernel.quotation g.schedule graph g.request.parameters)
      {s with pids:=fun ax => if ax=0 then pid else 0}).bind
        (Kernel.runPrograms M graph g.request tail)
    rw [show runSchedule (tail.map (IO.invocation M graph g))=
      Kernel.runPrograms M graph g.request tail from funext ih]

theorem real_schedule_safe (rest : Algebra ℝ) (g : Kernel.Geometry) (graph : Graph)
    (bounds : RegionName → Nat) (W : SourceSafety.Windows g.request bounds)
    (ids : List Nat) (valid : ∀ pid∈ids,pid<programs g.request) (s : State ℝ) :
    ScheduleSafe bounds (ids.map (IO.invocation (RealExecution.backend rest) graph g)) s := by
  induction ids generalizing s with
  | nil => trivial
  | cons pid tail ih =>
    constructor
    · exact SourceSafety.real_safe rest g.request graph bounds W pid (valid pid (by simp)) s
    · intro t _
      exact ih (fun pid hp => valid pid (by simp [hp])) t

theorem native_schedule_safe (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (g : Kernel.Geometry) (graph : Graph) (bounds : RegionName → Nat)
    (W : SourceSafety.Windows g.request bounds)
    (ids : List Nat) (valid : ∀ pid∈ids,pid<programs g.request) (s : State Word) :
    ScheduleSafe bounds (ids.map (IO.invocation (FPValue.backend rest cfg D F) graph g)) s := by
  induction ids generalizing s with
  | nil => trivial
  | cons pid tail ih =>
    constructor
    · exact SourceSafety.native_safe rest cfg D F g.request graph bounds W pid (valid pid (by simp)) s
    · intro t _
      exact ih (fun pid hp => valid pid (by simp [hp])) t

/-- Source-free physical facts reconstruct the historical allocation premise;
input tags come from exact active input binding only on the native path. -/
def allocation (g : Kernel.Geometry) (L : IO.Layout g) (s : State Word)
    (ha : ∀ o<g.rows*g.inner, ∃ v, s.mem g.A o=Cell.mk .fp32 v)
    (hb : ∀ o<g.inner*g.columns, ∃ v, s.mem g.B o=Cell.mk .fp32 v) : Allocation g.request s :=
  ⟨L.freshA,L.freshB,L.baseA,L.baseB,L.baseC,L.physicalFreshA,L.physicalFreshB,
    L.alignA,L.alignB,L.alignC,ha,hb⟩

theorem addressable (g : Kernel.Geometry) (L : IO.Layout g) (s : State Word) (ha hb) :
    Index64.Addressable g.request s (allocation g L s ha hb) :=
  ⟨L.inputAddressable,L.weightAddressable,L.outputAddressable⟩

theorem input_cells (M : Algebra Word) (graph : Graph) (g : Kernel.Geometry)
    (vals : ChannelValues Word (IO.inputs g)) (s : State Word)
    (h : (IO.matmulIO M graph g).InputCells vals s) :
    (∀ o<g.rows*g.inner, ∃ v, s.mem g.A o=Cell.mk .fp32 v) ∧
    (∀ o<g.inner*g.columns, ∃ v, s.mem g.B o=Cell.mk .fp32 v) := by
  constructor
  · intro o ho
    obtain ⟨i,k,eq⟩ := window g.rows g.inner o ho
    have z : s.mem g.A (i.val*max g.inner 1+k.val)=Cell.mk .fp32 (vals 0 (i,k,())) :=
      h (0 : Fin 2) (i,k,()) trivial
    exact ⟨_,eq ▸ z⟩
  · intro o ho
    obtain ⟨k,j,eq⟩ := window g.inner g.columns o ho
    have z : s.mem g.B (k.val*max g.columns 1+j.val)=Cell.mk .fp32 (vals 1 (k,j,())) :=
      h (1 : Fin 2) (k,j,()) trivial
    exact ⟨_,eq ▸ z⟩

theorem inputA (M : Algebra α) (graph : Graph) (g : Kernel.Geometry)
    (vals : ChannelValues α (IO.inputs g)) (s : State α)
    (h : (IO.matmulIO M graph g).InputBinding vals s)
    (i k : Nat) (hi : i<g.rows) (hk : k<g.inner) :
    (s.mem g.A (i*max g.inner 1+k)).read .fp32=IO.matrixA g vals i k := by
  simpa [IO.matmulIO,IO.inputs,IO.matrixA,hi,hk,mapValue] using h 0 (⟨i,hi⟩,⟨k,hk⟩,()) trivial

theorem inputB (M : Algebra α) (graph : Graph) (g : Kernel.Geometry)
    (vals : ChannelValues α (IO.inputs g)) (s : State α)
    (h : (IO.matmulIO M graph g).InputBinding vals s)
    (k j : Nat) (hk : k<g.inner) (hj : j<g.columns) :
    (s.mem g.B (k*max g.columns 1+j)).read .fp32=IO.matrixB g vals k j := by
  simpa [IO.matmulIO,IO.inputs,IO.matrixB,hk,hj,mapValue] using h 1 (⟨k,hk⟩,⟨j,hj⟩,()) trivial

#axiomsClean windows
#axiomsClean schedule_run
#axiomsClean real_schedule_safe
#axiomsClean native_schedule_safe
end
end VeriTile.Bench.Repairs.IvDependentMatmulDotShape.IOBinding
