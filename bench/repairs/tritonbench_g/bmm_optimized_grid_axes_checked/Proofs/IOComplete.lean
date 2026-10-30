import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.IO
import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Proofs.GraphWholeSafety
import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Proofs.ReadGrid

namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Interface
open VeriTile.Triton FP.Structural BmmOptimized SourceGraph
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

theorem output_valid (s : Shape) (i : Fin s.size) :
    s.Active (outputCell s i) ∧
      s.o (outputCell s i).batch (outputCell s i).row (outputCell s i).column=i.val := by
  have bound:=i.isLt
  change i.val<s.batch*s.m*s.n at bound
  have hm : 0<s.m := by
    by_cases hz : s.m=0
    · simp [hz] at bound
    · omega
  have hn : 0<s.n := by
    by_cases hz : s.n=0
    · simp [hz] at bound
    · omega
  have hmn : 0<s.m*s.n := Nat.mul_pos hm hn
  constructor
  · exact ⟨(Nat.div_lt_iff_lt_mul hmn).mpr (by nlinarith),Nat.mod_lt _ hm,Nat.mod_lt _ hn⟩
  · change (i.val/(s.m*s.n)*s.m+(i.val/s.n)%s.m)*s.n+i.val%s.n=i.val
    rw [Nat.mul_comm s.m s.n,←Nat.div_div_eq_div_mul]
    rw [Nat.mul_comm (i.val/s.n/s.m) s.m,Nat.div_add_mod,
      Nat.mul_comm (i.val/s.n) s.n,Nat.div_add_mod]

theorem invocation_initial (s : Shape) (c : Config) (p : Scalar α) (rest : Algebra α)
    (pid : Program) (initial : State (Node α)) :
    (invocation s c p rest pid).initial initial=Bridge.start s c pid initial := rfl

theorem schedule_equal (s : Shape) (c : Config) (p : Scalar α) (rest : Algebra α)
    (ps : List Program) (initial : State (Node α)) :
    runSchedule (ps.map (invocation s c p rest)) initial=
      Bridge.grid s c (scalar p) (base p rest) ps initial := by
  induction ps generalizing initial with
  | nil=>rfl
  | cons pid ps ih=>
    simp only [List.map_cons,runSchedule,Bridge.grid,Invocation.run,invocation_initial]
    change ((FP.Structural.exec (engine p rest) (Source.kernel (Source.parameters s c))
      (Bridge.start s c pid initial)).bind (runSchedule (ps.map (invocation s c p rest))))=_
    rw [←exec_equal]
    cases run : Execution.exec (scalar p) (base p rest) (Source.parameters s c)
        (Bridge.start s c pid initial) with
    | none=>rfl
    | some next=>exact ih next

theorem schedule_safe (s : Shape) (c : Config) (p : Scalar α) (rest : Algebra α)
    (ps : List Program) (legal : ∀ pid∈ps,pid∈programs s c) (bounds : RegionName → Nat)
    (ha : ∀ i,i<s.batch*s.m*s.k → i<bounds "A")
    (hb : ∀ i,i<s.batch*s.k*s.n → i<bounds "B")
    (ho : ∀ i,i<s.size → i<bounds "O") (initial : State (Node α)) :
    ScheduleSafe bounds (ps.map (invocation s c p rest)) initial := by
  induction ps generalizing initial with
  | nil=>trivial
  | cons pid ps ih=>
    refine ⟨?_,fun next _=>ih (fun q hq=>legal q (by simp [hq])) next⟩
    exact selected_safe s c p rest pid (legal pid (by simp)) bounds ha hb ho initial

theorem reference_observe (s : Shape) (p : Scalar α) (m : Memory (Node α))
    (x : BmmOptimized.Cell) (count : Nat) :
    observe p (reference s (scalar p) m x count)=
      reference s p (fun r i=>observe p (m r i)) x count := by
  induction count with
  | zero=>rfl
  | succ count ih=>
    simp only [reference]
    change p.fma _ _ (observe p (reference s (scalar p) m x count))=_
    rw [ih]
    by_cases inside : count<s.k <;> simp only [inside,ite_true,ite_false] <;> rfl

def inputMemory (initial : State (Node α)) : Memory (Node α)
  | .a,i=>(initial.mem "A" i).read .real
  | .b,i=>(initial.mem "B" i).read .real
  | .output,i=>(initial.mem "O" i).read .real
  | _,_=>default

theorem finite_reference (s : Shape) (c : Config) (p : Scalar α) (rest : Algebra α)
    (ps : List Program) (vals : (io s c p rest ps).Inputs) (initial : State (Node α))
    (h : (io s c p rest ps).InputBinding vals initial)
    (x : BmmOptimized.Cell) (hx : s.Active x) (count : Nat) :
    reference s p (fun r i=>observe p (inputMemory initial r i)) x count=
      reference s p (arrays p.zero s vals) x count := by
  induction count with
  | zero=>rfl
  | succ count ih=>
    simp only [reference,ih]
    by_cases inside : count<s.k
    · have ha:=active_a_bound s x.batch x.row count hx.1 hx.2.1 inside
      have hb:=active_b_bound s x.batch count x.column hx.1 inside hx.2.2
      have a:=h (0 : Fin 2) (⟨s.a x.batch x.row count,ha⟩,PUnit.unit) trivial
      have b:=h (1 : Fin 2) (⟨s.b x.batch count x.column,hb⟩,PUnit.unit) trivial
      change (initial.mem "A" (s.a x.batch x.row count)).read .real=
        Node.value (vals (0 : Fin 2) (⟨s.a x.batch x.row count,ha⟩,PUnit.unit)) at a
      change (initial.mem "B" (s.b x.batch count x.column)).read .real=
        Node.value (vals (1 : Fin 2) (⟨s.b x.batch count x.column,hb⟩,PUnit.unit)) at b
      simp only [inside,ite_true,arrays,dif_pos ha,dif_pos hb,inputMemory,a,b,observe]
      rfl
    · simp only [inside,ite_false]

theorem complete (s : Shape) (c : Config) (p : Scalar α) (rest : Algebra α)
    (ps : List Program) (order : ps.Perm (programs s c)) :
    (io s c p rest ps).Implements (scalarFormula s p) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds hb initial _
    apply schedule_safe s c p rest ps (fun pid hp=>order.mem_iff.mp hp) bounds
    · intro i hi
      exact hb.2.1 (0 : Fin 2) (⟨i,hi⟩,PUnit.unit) trivial
    · intro i hi
      exact hb.2.1 (1 : Fin 2) (⟨i,hi⟩,PUnit.unit) trivial
    · intro i hi
      exact hb.2.2.1 (0 : Fin 1) (⟨i,hi⟩,PUnit.unit) trivial
  · intro vals initial hin
    obtain ⟨final,run,values,frame⟩:=ReadGrid.ordered_execution s c (scalar p) (base p rest)
      ps order (inputMemory initial) initial ⟨fun _=>rfl,fun _=>rfl⟩
    refine ⟨final,?_,?_,?_⟩
    · exact (schedule_equal s c p rest ps initial).trans run
    · intro o j _
      have ho : o=(0 : Fin 1) := Fin.eq_zero o
      subst o
      have valid:=output_valid s j.1
      have result:=values (outputCell s j.1) valid.1
      rw [valid.2] at result
      change mapCell (observe p) (final.mem "O" j.1.val)=
        .mk .real (ordered s p (arrays p.zero s vals) (outputCell s j.1))
      rw [result]
      change FP.Structural.Cell.mk .real
        (observe p (reference s (scalar p) (inputMemory initial) (outputCell s j.1) _))=_
      rw [reference_observe,finite_reference s c p rest ps vals initial hin _ valid.1]
      rfl
    · intro r off outside
      apply frame
      by_cases eq : r="O"
      · right
        by_contra small
        have hi : off<s.size := by omega
        have no:=outside.1 (0 : Fin 1) (⟨off,hi⟩,PUnit.unit) trivial
        exact no.elim (fun h=>h eq) (fun h=>h rfl)
      · exact Or.inl eq

#axiomsClean complete
end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Interface
