import bench.tritonbench_g.rowwise_quantization_triton.IO
import bench.tritonbench_g.rowwise_quantization_triton.Proofs.SourceSafety
import bench.tritonbench_g.rowwise_quantization_triton.Proofs.KernelWhole

namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.IOImplements
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
set_option linter.unusedSectionVars false
variable {α : Type} [Inhabited α]

theorem invocation_initial (M : Algebra α) (round : α → BitVec 64) (equal : α → α → Bool)
    (p : Launch) (r : Nat) (s : State α) :
    (IO.invocation M round equal p r).initial s=KernelGrid.atRow s r := by
  unfold IntegerInvocation.initial Invocation.initial IO.invocation bindRegisters KernelGrid.atRow
  simp only [List.foldl_nil,Bool.false_eq_true,if_false,Option.getD_none]
  congr 1
  funext ax
  by_cases h : ax=0 <;> simp [h,Function.update]

theorem invocation_run (M : Algebra α) (round : α → BitVec 64) (equal : α → α → Bool)
    (p : Launch) (r : Nat) (s : State α) :
    (IO.invocation M round equal p r).run s=
      KernelExecution.exec (KernelExecution.backend M equal) round
        (KernelExecution.kernel (KernelGrid.parameters p)) (KernelGrid.atRow s r) := by
  unfold IntegerInvocation.run
  rw [invocation_initial]
  exact IntegerBinding.execution_eq M round equal (KernelGrid.parameters p) _

theorem run_schedule (M : Algebra α) (round : α → BitVec 64) (equal : α → α → Bool)
    (p : Launch) (order : List Nat) (s : State α) :
    runIntegerSchedule (order.map (IO.invocation M round equal p)) s=
      KernelGrid.run M round equal p order s := by
  induction order generalizing s with
  | nil => rfl
  | cons row rest ih =>
    simp only [List.map_cons,runIntegerSchedule,invocation_run,KernelGrid.run]
    cases KernelExecution.exec (KernelExecution.backend M equal) round
      (KernelExecution.kernel (KernelGrid.parameters p)) (KernelGrid.atRow s row) <;> simp [ih]

theorem schedule_safe (M : Algebra α) (round : α → BitVec 64) (equal : α → α → Bool)
    (p : Launch) (schedule : Schedule p) (vals : ChannelValues α (IO.inputs p))
    (bounds : RegionName → Nat) (h : (IO.quantizerIO M round equal p schedule).InBounds bounds vals)
    (s : State α) : IntegerScheduleSafe bounds ((IO.quantizerIO M round equal p schedule).schedule vals) s := by
  have one : ∀ row, row<p.rows → ∀ s : State α,
      (IO.invocation M round equal p row).Safe bounds s := by
    intro row hr s
    unfold IntegerInvocation.Safe
    rw [invocation_initial]
    apply SourceSafety.source_safe M round equal (KernelGrid.parameters p)
      (KernelGrid.atRow s row) (lt_of_lt_of_le p.positive p.covers) bounds
    · intro j hj
      exact h.2.1 (0:Fin 1) (⟨row,hr⟩,⟨j.val,hj⟩,()) trivial
    · intro j hj
      exact h.2.2.1 (0:Fin 2) (⟨row,hr⟩,⟨j.val,hj⟩,()) trivial
    · exact h.2.2.1 (1:Fin 2) (⟨row,hr⟩,()) trivial
  have go : ∀ order : List Nat, (∀ row∈order,row<p.rows) → ∀ s : State α,
      IntegerScheduleSafe bounds (order.map (IO.invocation M round equal p)) s := by
    intro order
    induction order with
    | nil => intros;trivial
    | cons row rest ih =>
      intro valid s
      exact ⟨one row (valid _ List.mem_cons_self) s,
        fun t _ => ih (fun row hr => valid row (List.mem_cons_of_mem _ hr)) t⟩
  exact go schedule.order (fun row hr => List.mem_range.mp (schedule.complete.mem_iff.mp hr)) s

theorem reads (M : Algebra α) (round : α → BitVec 64) (equal : α → α → Bool)
    (p : Launch) (schedule : Schedule p) (vals : ChannelValues α (IO.inputs p))
    (s : State α) (h : (IO.quantizerIO M round equal p schedule).InputBinding vals s) :
    KernelWhole.Reads p (IO.values p vals) s := by
  intro r c hr hc
  simpa [IO.quantizerIO,IO.values,hr,hc,mapValue] using h (0:Fin 1) (⟨r,hr⟩,⟨c,hc⟩,()) trivial

theorem outside_bounds (M : Algebra α) (round : α → BitVec 64) (equal : α → α → Bool)
    (p : Launch) (schedule : Schedule p) (vals : ChannelValues α (IO.inputs p))
    (r : RegionName) (a : Nat) (h : (IO.quantizerIO M round equal p schedule).Outside vals r a) :
    (r≠"q" ∨ p.rows*p.columns≤a) ∧ (r≠"m" ∨ p.rows≤a) := by
  constructor
  · by_cases rq : r="q"
    · right
      by_contra large
      have small : a<p.rows*p.columns := by omega
      have hr : a/p.columns<p.rows := (Nat.div_lt_iff_lt_mul p.positive).mpr small
      have hc : a%p.columns<p.columns := Nat.mod_lt _ p.positive
      have missWindow := h.1 (0:Fin 2) (⟨a/p.columns,hr⟩,⟨a%p.columns,hc⟩,()) trivial
      rcases missWindow with miss|miss
      · exact miss rq
      · exact miss (by simpa [Nat.mul_comm] using (Nat.div_add_mod a p.columns).symm)
    · exact Or.inl rq
  · by_cases rm : r="m"
    · right
      by_contra large
      have hr : a<p.rows := by omega
      exact (h.1 (1:Fin 2) (⟨a,hr⟩,()) trivial).elim (fun miss => miss rm) (fun miss => miss rfl)
    · exact Or.inl rm

def target (ops : Operations α) (half : α → α) (round : α → BitVec 64) (p : Launch) :=
  IO.result p
    (fun values row col => (KernelExecution.signedByte (round (rowScaled ops (values row col) (rowMaximum ops p (values row))))))
    (fun values row => half (rowMaximum ops p (values row)))

/-- Reuse the exact complete-grid result after identifying each source
invocation with the shared executor; add actual source safety and IO binding. -/
theorem implements (ops : Operations α) (half : α → α) (round : α → BitVec 64)
    (rest : Algebra α) (p : Launch) (schedule : Schedule p) :
    (IO.quantizerIO (KernelProfile.algebra ops half p.tree rest) round (KernelProfile.equal ops) p schedule).Implements
      (target ops half round p) := by
  apply IntegerProgramIO.Implements.intro
  · intro vals bounds hb s _
    exact schedule_safe _ round _ p schedule vals bounds hb s
  · intro vals s hin
    obtain ⟨final,ran,hq,hm,hf⟩ := KernelWhole.source_correct ops half round rest p schedule
      (IO.values p vals) s (reads _ round _ p schedule vals s hin)
    refine ⟨final,?_,?_,?_⟩
    · exact (run_schedule _ round _ p schedule.order s).trans ran
    · intro o j _
      fin_cases o
      · change mapCell id (final.mem "q" (j.1.val*p.columns+j.2.1.val))=
          Cell.mk .int ((KernelExecution.signedByte (round (rowScaled ops (IO.values p vals j.1.val j.2.1.val)
            (rowMaximum ops p (IO.values p vals j.1.val))))))
        rw [hq _ _ j.1.isLt j.2.1.isLt]
        rfl
      · change mapCell id (final.mem "m" j.1.val)=
          Cell.mk .fp16 (half (rowMaximum ops p (IO.values p vals j.1.val)))
        rw [hm _ j.1.isLt]
        rfl
    · intro r a outside
      obtain ⟨hq,hm⟩ := outside_bounds _ round _ p schedule vals r a outside
      exact hf r a hq hm

theorem unsigned8 (z : Int) : IntegerOutputView.apply (.unsignedWidth 8) z=(narrow8 z).val := by
  have lo : 0≤z%256 := Int.emod_nonneg _ (by norm_num)
  have hi : z%256<256 := Int.emod_lt_of_pos _ (by norm_num)
  simp [IntegerOutputView.apply,narrow8,BitVec.toNat_ofInt,lo,Int.emod_eq_of_lt lo hi]

def observedTarget (ops : Operations α) (half : α → α) (round : α → BitVec 64) (p : Launch) :=
  IO.result p
    (fun values row col => ((narrow8 (round
      (rowScaled ops (values row col) (rowMaximum ops p (values row)))).toInt).val : Int))
    (fun values row => half (rowMaximum ops p (values row)))

theorem observed (ops : Operations α) (half : α → α) (round : α → BitVec 64)
    (rest : Algebra α) (p : Launch) (schedule : Schedule p) :
    (IO.quantizerIO (KernelProfile.algebra ops half p.tree rest) round (KernelProfile.equal ops) p schedule).ObservesInteger
      (IO.byteView p) (observedTarget ops half round p) := by
  have base := IntegerProgramIO.ObservesInteger.of_implements
    (implements ops half round rest p schedule) (IO.byteView p)
  have mapped : (fun vals => (IO.quantizerIO (KernelProfile.algebra ops half p.tree rest)
      round (KernelProfile.equal ops) p schedule).toMemoryIO.mapIntegerOutputs
      (IO.byteView p) (target ops half round p vals))=observedTarget ops half round p := by
    funext vals o j
    fin_cases o
    · exact (unsigned8 _).trans
        (congrArg (fun b : Byte => (b.val : Int)) (KernelExecution.narrow_signedByte _))
    · rfl
  rw [mapped] at base
  exact base

#axiomsClean implements
#axiomsClean observed
end
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.IOImplements
