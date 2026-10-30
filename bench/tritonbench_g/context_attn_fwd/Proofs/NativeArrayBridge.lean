import bench.tritonbench_g.context_attn_fwd.InvocationIO
import bench.tritonbench_g.context_attn_fwd.Proofs.KernelRawGrid
import VeriTile.Triton.Float.ArrayProgramIO

namespace VeriTile.Bench.Repairs.ContextAttnFwd.NativeArrayBridge
open VeriTile.Triton FP.Structural NativeFormula
noncomputable section
open Classical
variable {α : Type} [Inhabited α]

def arrays (r : Regions) (s : State α) : Arrays α where
  query := fun off => (s.mem r.query off).read .real
  key := fun off => (s.mem r.key off).read .real
  value := fun off => (s.mem r.value off).read .real
  start := fun off => (s.mem r.start off).read .int
  sequence := fun off => (s.mem r.sequence off).read .int
  prompt := fun off => (s.mem r.prompt off).read .int

theorem arrays_congr (r : Regions) (s t : State α) (memory : s.mem=t.mem) :
    arrays r s=arrays r t := by simp only [arrays,memory]

theorem operands_eq (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (job : Launch) (s : State α) :
    operands p job (arrays r s)=FPReference.inputs p r ((invocation p r scale M job).initial s) := rfl

theorem arrays_write (r : Regions) (s : State α) (off : Nat) (v : α) :
    arrays r (s.write r.out off (.mk .real v))=writeArrays r (arrays r s) off v := by
  have floating (rg : RegionName) :
      (fun k => ((s.write r.out off (.mk .real v)).mem rg k).read .real) =
        (if rg=r.out then Function.update (fun k => (s.mem rg k).read .real) off v
          else fun k => (s.mem rg k).read .real) := by
    funext k
    by_cases hr : rg=r.out <;> by_cases hk : k=off <;>
      simp [State.write,hr,hk,Function.update]
  have integer (rg : RegionName) :
      (fun k => ((s.write r.out off (.mk .real v)).mem rg k).read .int) =
        (if rg=r.out then Function.update (fun k => (s.mem rg k).read .int) off 0
          else fun k => (s.mem rg k).read .int) := by
    funext k
    by_cases hr : rg=r.out <;> by_cases hk : k=off <;>
      simp [State.write,hr,hk,Function.update]
    rfl
  unfold arrays writeArrays
  rw [floating,floating,floating,integer,integer,integer]

def applyWrites (r : Regions) (changes : List (Nat×α)) (s : State α) : State α :=
  changes.foldl (fun current w => current.write r.out w.1 (.mk .real w.2)) s

theorem arrays_applyWrites (r : Regions) (changes : List (Nat×α)) (s : State α) :
    arrays r (applyWrites r changes s)=update r changes (arrays r s) := by
  induction changes generalizing s with
  | nil => rfl
  | cons w rest ih =>
    change arrays r (applyWrites r rest (s.write r.out w.1 (.mk .real w.2))) = _
    rw [ih,arrays_write]
    rfl

theorem applyWrites_congr (r : Regions) (changes : List (Nat×α)) (s t : State α)
    (memory : s.mem=t.mem) : (applyWrites r changes s).mem=(applyWrites r changes t).mem := by
  induction changes generalizing s t with
  | nil => exact memory
  | cons w rest ih =>
    apply ih
    simp only [State.write,memory]

theorem applyWrites_append (r : Regions) (xs ys : List (Nat×α)) (s : State α) :
    applyWrites r (xs++ys) s=applyWrites r ys (applyWrites r xs s) := by
  simp only [applyWrites,List.foldl_append]

theorem scatter_applyWrites (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (dots : FPSuccess.DotSupport p M) (job : Launch) (s : State α) :
    (FPSuccess.scatter p r ((invocation p r scale M job).initial s)
      ((invocation p r scale M job).initial s)
      (FPReference.output p M dots scale
        (FPReference.inputs p r ((invocation p r scale M job).initial s)))).mem =
      (applyWrites r (writes p M dots scale job (arrays r s)) s).mem := by
  rw [←operands_eq]
  unfold FPSuccess.scatter writes
  generalize TileShape.allIndices [p.BM,p.D]=indices
  have fold : ∀ (indices : List (TileIndex [p.BM,p.D])) (t u : State α),
      t.mem=u.mem →
      (indices.foldl (fun acc i => if FPSuccess.active p r ((invocation p r scale M job).initial s) i
        then acc.write r.out (FPSuccess.offset p r ((invocation p r scale M job).initial s) i)
          (.mk .real (FPReference.output p M dots scale (operands p job (arrays r s)) i))
        else acc) t).mem =
      (applyWrites r (indices.filterMap (fun i => if active p (operands p job (arrays r s)) i
        then some (address p (operands p job (arrays r s)) i,
          FPReference.output p M dots scale (operands p job (arrays r s)) i) else none)) u).mem := by
    intro indices
    induction indices with
    | nil => intro t u memory; exact memory
    | cons i rest ih =>
      intro t u memory
      have ha : FPSuccess.active p r ((invocation p r scale M job).initial s) i =
          active p (operands p job (arrays r s)) i := rfl
      have ho : FPSuccess.offset p r ((invocation p r scale M job).initial s) i =
          address p (operands p job (arrays r s)) i := rfl
      by_cases hi : active p (operands p job (arrays r s)) i
      · simp only [List.foldl_cons,List.filterMap_cons,ha,ho,if_pos hi,applyWrites]
        apply ih
        simp only [State.write,memory]
      · simpa only [List.foldl_cons,List.filterMap_cons,ha,if_neg hi] using ih t u memory
  exact fold indices _ _ rfl

theorem program (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (dots : FPSuccess.DotSupport p M) (positive : 0 < p.BN) (job : Launch) (s : State α) :
    ∃ t,(invocation p r scale M job).run s=some t ∧
      t.mem=(applyWrites r (writes p M dots scale job (arrays r s)) s).mem := by
  obtain ⟨t,ran,memory⟩ := FPReference.program p r scale M dots positive
    ((invocation p r scale M job).initial s)
  exact ⟨t,ran,memory.trans (scatter_applyWrites p r scale M dots job s)⟩

theorem schedule (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (dots : FPSuccess.DotSupport p M) (positive : 0 < p.BN) (jobs : List Launch) (s : State α) :
    ∃ t,runSchedule (jobs.map (invocation p r scale M)) s=some t ∧
      t.mem=(applyWrites r (allWrites p r M dots scale jobs (arrays r s)) s).mem := by
  induction jobs generalizing s with
  | nil => exact ⟨s,rfl,rfl⟩
  | cons job rest ih =>
    obtain ⟨mid,ran,memory⟩ := program p r scale M dots positive job s
    obtain ⟨t,done,final⟩ := ih mid
    have view := (arrays_congr r mid _ memory).trans (arrays_applyWrites r _ s)
    refine ⟨t,?_,?_⟩
    · simp only [List.map_cons,runSchedule,ran,Option.bind_some]
      exact done
    · rw [final,view,allWrites,applyWrites_append]
      exact applyWrites_congr r _ _ _ memory

#axiomsClean schedule
end
end VeriTile.Bench.Repairs.ContextAttnFwd.NativeArrayBridge
