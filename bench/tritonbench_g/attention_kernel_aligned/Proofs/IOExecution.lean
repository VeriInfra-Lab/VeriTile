import bench.tritonbench_g.attention_kernel_aligned.IO
import bench.tritonbench_g.attention_kernel_aligned.Proofs.GraphLocality

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Interface
open VeriTile Triton FP.Structural Full Whole FPValue
noncomputable section
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

theorem job_initial {α : Type} (M : Algebra α) (P : Primitives α) (p : Parameters)
    (i : Program) (s : State α) : (job M P p i).initial s=start (grid p) i s := rfl

theorem jobs_run {α : Type} [Inhabited α] (M : Algebra α) (P : Primitives α)
    (p : Parameters) (work : List Program) (s : State α) :
    runSchedule (jobs M P p work) s=runPrograms (fun _=>M) (fun _=>P) p (grid p) work s := by
  induction work generalizing s with
  | nil => rfl
  | cons i work ih =>
    simp only [jobs,List.map_cons,runSchedule,Invocation.run,job_initial,job,runPrograms]
    exact congrArg (Option.bind _) (funext ih)

theorem algorithm_at {α : Type} [Inhabited α] (P : Primitives α) (p : Parameters) (before : State α)
    (x b h : Nat) (hh : h<p.heads) :
    Whole.answer P p (start (grid p) (x,b*p.heads+h,0) before)=
      KernelOnly.Specification.outputValue P p (KernelOnly.operandReader before) x b h := by
  rw [KernelOnly.source_algorithm]
  have hp : 0<p.heads := by omega
  have hd : (b*p.heads+h)/p.heads=b := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hp,Nat.div_eq_of_lt hh]
    omega
  have hm : (b*p.heads+h)%p.heads=h := by simp [Nat.add_mod,Nat.mod_eq_of_lt hh]
  simp [start,ids,hd,hm] <;> rfl

theorem real_finite (p : Parameters) (g : Guard p) (work : List Program)
    (vals : (realIO p work).Inputs) (before : State Real.R)
    (bound : (realIO p work).InputBinding vals before)
    (b h n d : Nat) (hb : b<p.batches) (hh : h<p.heads) (hn : n<p.length) (hd : d<p.head) :
    answer Real.model p (start (grid p) (n/p.blockM,b*p.heads+h,0) before)
      (lane p g.selected n d hd)=some (Real.expected p (readInputs .fp16 p vals) b h n d) := by
  let μ := readInputs .fp16 p vals
  let reference : State Real.R := {before with mem := fun r off => .mk .fp16 (some (μ r off))}
  have reads : Real.Reads p reference μ := by intro r hr off;rfl
  have points := point_binding .fp16 p (some : ℝ → Real.R) vals before (fun i j => bound i j trivial)
  have agree : Agree p (KernelOnly.operandReader before) (KernelOnly.operandReader reference) := by
    intro i
    exact readInputs_binding .fp16 p (some : ℝ → Real.R) vals before points i
  have same := Locality.output_agree Real.model p g _ _ agree (n/p.blockM) b h hb hh
  rw [algorithm_at _ _ _ _ _ _ hh,same,←algorithm_at _ _ reference _ _ _ hh]
  have active : Source.row p (start (grid p) (n/p.blockM,b*p.heads+h,0) reference)
      (lane p g.selected n d hd).1<p.length := by
    change n/p.blockM*p.blockM+n%p.blockM<p.length
    simpa only [Nat.mul_comm,Nat.div_add_mod] using hn
  change Source.outputValue Real.model p _ _=_
  have launchedReads : Real.Reads p (start (grid p) (n/p.blockM,b*p.heads+h,0) reference) μ := reads
  rw [Real.answer_real p _ μ launchedReads _ active (g.selected.blockN_pos p),
    Real.expected_at p g.selected reference μ b h n d hh hd]

theorem native_finite (p : Parameters) (g : Guard p) (work : List Program)
    (vals : (nativeIO p work).Inputs) (before : State Native.Expr)
    (bound : (nativeIO p work).InputBinding vals before)
    (b h n : Nat) (hb : b<p.batches) (hh : h<p.heads) :
    answer (Native.modelFor p) p (start (grid p) (n/p.blockM,b*p.heads+h,0) before)=
      KernelOnly.Specification.outputValue (Native.modelFor p) p (readInputs .fp16 p vals)
        (n/p.blockM) b h := by
  have points := point_binding .fp16 p (id : Native.Expr → Native.Expr) vals before (fun i j => bound i j trivial)
  have agree : Agree p (KernelOnly.operandReader before) (readInputs .fp16 p vals) := by
    intro i
    exact readInputs_binding .fp16 p (id : Native.Expr → Native.Expr) vals before points i
  rw [algorithm_at _ _ _ _ _ _ hh]
  exact Locality.output_agree (Native.modelFor p) p g _ _ agree (n/p.blockM) b h hb hh

#axiomsClean real_finite
#axiomsClean native_finite
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.Interface
