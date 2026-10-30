import bench.tritonbench_g.context_attn_fwd.Proofs.NativeWholeSafety
import bench.tritonbench_g.context_attn_fwd.Proofs.NativeArrayResults

namespace VeriTile.Bench.Repairs.ContextAttnFwd.NativeIOComplete
open VeriTile.Triton FP.Structural NativeFormula NativeArrayBridge NativeArrayResults
noncomputable section
variable {α : Type} [Inhabited α]

structure Backed (p : Layout) (r : Regions) (M : Algebra α)
    (dots : FPSuccess.DotSupport p M) (scale : ℝ) (jobs : List Launch) (a : Arrays α)
    (caps : RegionName → Nat) : Prop where
  reads : ∀ role : Fin 6,∀ off,off ∈ readFootprint p r M dots scale role jobs a →
    off < caps (NativeIO.inputs r)[role].buf
  writes : ∀ w,w ∈ allWrites p r M dots scale jobs a → w.1 < caps r.out

theorem Backed.head (p : Layout) (r : Regions) (M : Algebra α)
    (dots : FPSuccess.DotSupport p M) (scale : ℝ) (job : Launch) (jobs : List Launch)
    (a : Arrays α) (caps : RegionName → Nat) (bound : Backed p r M dots scale (job::jobs) a caps) :
    NativeSourceSafety.Bounds p r (operands p job a) caps := by
  have headRead (role : Fin 6) (off : Nat)
      (member : off ∈ (if role.val=0 then queryReads p (operands p job a)
        else if role.val=1 then keyReads p (operands p job a)
        else if role.val=2 then valueReads p (operands p job a) else [(operands p job a).batch])) :
      off < caps (NativeIO.inputs r)[role].buf :=
    bound.reads role off (List.mem_append_left _ member)
  refine ⟨headRead 3 _ (by simp),headRead 4 _ (by simp),headRead 5 _ (by simp),
    fun off h => headRead 0 off h,fun off h => headRead 1 off h,fun off h => headRead 2 off h,?_⟩
  intro i hi
  apply bound.writes (_,FPReference.output p M dots scale (operands p job a) i)
  apply List.mem_append_left
  exact (mem_writes p M dots scale job a _ _).mpr ⟨i,hi,rfl,rfl⟩

theorem Backed.tail (p : Layout) (r : Regions) (M : Algebra α)
    (dots : FPSuccess.DotSupport p M) (scale : ℝ) (job : Launch) (jobs : List Launch)
    (a : Arrays α) (caps : RegionName → Nat) (bound : Backed p r M dots scale (job::jobs) a caps) :
    Backed p r M dots scale jobs (update r (NativeFormula.writes p M dots scale job a) a) caps := by
  constructor
  · intro role off member
    exact bound.reads role off (List.mem_append_right _ member)
  · intro w member
    exact bound.writes w (List.mem_append_right _ member)

theorem schedule_safe (p : Layout) (r : Regions) (M : Algebra α)
    (dots : FPSuccess.DotSupport p M) (scale : ℝ) (jobs : List Launch) (a : Arrays α)
    (positive : 0 < p.BN) (caps : RegionName → Nat) (bound : Backed p r M dots scale jobs a caps)
    (s : State α) (view : arrays r s=a) :
    ScheduleSafe caps (jobs.map (invocation p r scale M)) s := by
  induction jobs generalizing a s with
  | nil => trivial
  | cons job rest ih =>
    have head := bound.head p r M dots scale job rest a caps
    have head' : NativeSourceSafety.Bounds p r
        (FPReference.inputs p r ((invocation p r scale M job).initial s)) caps := by
      rw [←operands_eq,view]
      exact head
    constructor
    · exact NativeSourceSafety.source p r scale M dots positive _ caps head'
    · intro t ran
      obtain ⟨mid,actual,memory⟩ := program p r scale M dots positive job s
      have same : t=mid := Option.some.inj (ran.symm.trans actual)
      subst t
      apply ih _ (bound.tail p r M dots scale job rest a caps) mid
      rw [←view]
      exact (arrays_congr r mid _ memory).trans (arrays_applyWrites r _ s)

theorem backed (p : Layout) (r : Regions) (M : Algebra α)
    (dots : FPSuccess.DotSupport p M) (scale : ℝ) (jobs : List Launch)
    (v : (NativeIO.io p r M dots scale jobs).Inputs) (caps : RegionName → Nat)
    (bound : (NativeIO.io p r M dots scale jobs).InBounds caps v) :
    Backed p r M dots scale jobs (NativeIO.arrays r v) caps := by
  constructor
  · exact bound.2.1
  · intro w member
    obtain ⟨k,hk⟩ := (mem_allWrites p r M dots scale jobs _ w).mp member
    obtain ⟨i,hi,ho,hv⟩ := (mem_writes p M dots scale _ _ _ _).mp hk
    have hout := bound.2.2.1 (0 : Fin 1) (k,i) hi
    change address p (NativeIO.eventOperands p r M dots scale jobs v k) i < caps r.out at hout
    simpa only [NativeIO.eventOperands,ho] using hout

/-- Both execution/output/frame and actual source safety are derived; none is
an input assumption. Capacities do not restrict the admitted raw-array domain. -/
theorem complete (p : Layout) (r : Regions) (M : Algebra α)
    (dots : FPSuccess.DotSupport p M) (scale : ℝ) (jobs : List Launch) (positive : 0 < p.BN) :
    (NativeIO.io p r M dots scale jobs).Implements (NativeIO.formula p r M dots scale jobs) := by
  apply ArrayProgramIO.ImplementsOn.intro
  · intro v _ caps bounds s bound
    exact schedule_safe p r M dots scale jobs _ positive caps (backed p r M dots scale jobs v caps bounds)
      s (bound_arrays p r M dots scale jobs v s bound)
  · intro v _ s bound
    exact NativeArrayResults.correct p r M dots scale jobs positive v s bound

#axiomsClean complete
end
end VeriTile.Bench.Repairs.ContextAttnFwd.NativeIOComplete
