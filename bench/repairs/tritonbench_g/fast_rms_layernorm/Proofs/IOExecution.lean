import bench.repairs.tritonbench_g.fast_rms_layernorm.IO
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.RawInputs

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.IOExecution
open VeriTile Triton FP.Structural FP.ProfiledRewrite NativeGraph
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

def realHeap (s : BlockState) : RealGridForward.Heap := fun a=>s.mem a.1 a.2

theorem real_initial (h : Call) (backward weight : Bool) (pid count : Nat) (s : BlockState) :
    (RealIO.job h backward weight pid count).initial s=RealGridForward.entry (realHeap s) pid count := rfl

theorem real_map {ι : Type} (jobs : List ι) (job : ι → PreparedAlgorithmInvocation)
    (execute : ι → RealGridForward.Heap → Option RealGridForward.Heap)
    (head : ∀ i s,((job i).run s).map realHeap=execute i (realHeap s)) (s : BlockState) :
    (runPreparedAlgorithmSchedule (jobs.map job) s).map realHeap=GridRun.run execute jobs (realHeap s) := by
  induction jobs generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.map_cons,runPreparedAlgorithmSchedule,GridRun.run]
    rw [←head i s]
    cases ran : (job i).run s <;> simp [ran,ih]

theorem real_forward (h : Call) (s : BlockState) :
    (runPreparedAlgorithmSchedule (RealIO.schedule h Bool.false) s).map realHeap=
      RealGridForward.run h (realHeap s) := by
  apply real_map
  intro i s
  simp only [PreparedAlgorithmInvocation.run,real_initial]
  rfl

theorem real_rows (h : Call) (s : BlockState) :
    (runPreparedAlgorithmSchedule ((List.finRange h.rows).map
      (fun i=>RealIO.job h Bool.true Bool.false i.val h.rows)) s).map realHeap=
      RealGridBackward.run h (realHeap s) := by
  apply real_map
  intro i s
  simp only [PreparedAlgorithmInvocation.run,real_initial]
  rfl

theorem real_weights (h : Call) (s : BlockState) :
    (runPreparedAlgorithmSchedule ((List.finRange (Geometry.featureCount h)).map
      (fun i=>RealIO.job h Bool.true Bool.true i.val (Geometry.featureCount h))) s).map realHeap=
      RealGridWeight.run h (realHeap s) := by
  apply real_map
  intro i s
  simp only [PreparedAlgorithmInvocation.run,real_initial]
  rfl

theorem real_append (left right : List PreparedAlgorithmInvocation) (s : BlockState) :
    runPreparedAlgorithmSchedule (left++right) s=
      (runPreparedAlgorithmSchedule left s).bind (runPreparedAlgorithmSchedule right) := by
  induction left generalizing s with
  | nil=>rfl
  | cons job rest ih=>
    simp only [List.cons_append,runPreparedAlgorithmSchedule]
    cases ran : job.run s <;> simp [ran,ih]

theorem real_backward (h : Call) (nonempty : h.rows≠0) (s : BlockState) :
    (runPreparedAlgorithmSchedule (RealIO.schedule h Bool.true) s).map realHeap=
      RealTensor.backwardRun h (realHeap s) := by
  simp only [RealIO.schedule,if_true,if_neg nonempty,RealTensor.backwardRun,RealSequence.run]
  rw [real_append,←real_weights h s]
  cases ran : runPreparedAlgorithmSchedule
    ((List.finRange (Geometry.featureCount h)).map
      (fun i=>RealIO.job h Bool.true Bool.true i.val (Geometry.featureCount h))) s
  · simp
  · simpa only [Option.map_some,Option.bind_some] using real_rows h _

def nativeHeap (s : State Term) : FPGridForward.Heap Term := fun a=>s.mem a.1 a.2

def nativeBase (pid count : Nat) (s : State Term) : State Term :=
  {s with regs:=fun _ _ _=>none,pids:=fun _=>pid,numPids:=fun _=>count}

theorem native_forward_step (M : Algebra Term) (h : Call) (epsilon : Term)
    (i : Fin h.rows) (s : State Term) :
    ((NativeIO.forwardJob M h epsilon i.val).run s).map nativeHeap=
      RuntimeForward.Grid.execute M h epsilon i (nativeHeap s) := by
  obtain ⟨v,ran,mem⟩ := RuntimeForward.Source.forward_memory M h epsilon (nativeBase i.val h.rows s)
  obtain ⟨w,other,omem⟩ := RuntimeForward.Source.forward_memory M h epsilon
    (FPGridForward.entry (nativeHeap s) i.val h.rows)
  change (FP.Structural.exec (engine M) (Kernels.forwardRuntime h)
    (RuntimeForward.Source.bind (nativeBase i.val h.rows s) epsilon)).map nativeHeap=_
  rw [ran]
  simp only [RuntimeForward.Grid.execute,other,Option.map_some]
  apply congrArg some
  funext a
  change v.mem a.1 a.2=w.mem a.1 a.2
  rw [mem,omem]
  rfl

theorem native_backward_step (M : Algebra Term) (h : Call) (i : Fin h.rows) (s : State Term) :
    ((NativeIO.backwardJob M h Bool.false i.val h.rows).run s).map nativeHeap=
      FPGridBackward.execute M h i (nativeHeap s) := by
  obtain ⟨v,ran,mem⟩ := FPBackward.source_memory M h (nativeBase i.val h.rows s)
  obtain ⟨w,other,omem⟩ := FPBackward.source_memory M h (FPGridForward.entry (nativeHeap s) i.val h.rows)
  change (FP.Structural.exec (engine M) (Kernels.backward h) (nativeBase i.val h.rows s)).map nativeHeap=_
  rw [ran]
  simp only [FPGridBackward.execute,other,Option.map_some]
  apply congrArg some
  funext a
  change v.mem a.1 a.2=w.mem a.1 a.2
  rw [mem,omem]
  rfl

theorem native_weight_step (M : Algebra Term) (h : Call)
    (i : Fin (Geometry.featureCount h)) (s : State Term) :
    ((NativeIO.backwardJob M h Bool.true i.val (Geometry.featureCount h)).run s).map nativeHeap=
      FPGridWeight.execute M h i (nativeHeap s) := by
  obtain ⟨v,ran,mem⟩ := FPWeightMemory.source_memory M h 256 (nativeBase i.val (Geometry.featureCount h) s)
  obtain ⟨w,other,omem⟩ := FPWeightMemory.source_memory M h 256
    (FPGridForward.entry (nativeHeap s) i.val (Geometry.featureCount h))
  change (FP.Structural.exec (engine M) (Kernels.weightBackward h 256)
    (nativeBase i.val (Geometry.featureCount h) s)).map nativeHeap=_
  rw [ran]
  simp only [FPGridWeight.execute,other,Option.map_some]
  apply congrArg some
  funext a
  change v.mem a.1 a.2=w.mem a.1 a.2
  rw [mem,omem]
  rfl

theorem native_map {ι : Type} (jobs : List ι) (job : ι → Invocation Term)
    (execute : ι → FPGridForward.Heap Term → Option (FPGridForward.Heap Term))
    (head : ∀ i s,((job i).run s).map nativeHeap=execute i (nativeHeap s)) (s : State Term) :
    (runSchedule (jobs.map job) s).map nativeHeap=GridRun.run execute jobs (nativeHeap s) := by
  induction jobs generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.map_cons,runSchedule,GridRun.run]
    rw [←head i s]
    cases ran : (job i).run s <;> simp [ran,ih]

theorem native_forward (R : Reductions) (h : Call) (epsilon : Term) (s : State Term) :
    (runSchedule (NativeIO.schedule R h Bool.false epsilon) s).map nativeHeap=
      RuntimeForward.Grid.run (algebra R .forward) h epsilon (nativeHeap s) :=
  native_map _ _ _ (native_forward_step (algebra R .forward) h epsilon) s

theorem native_rows (R : Reductions) (h : Call) (s : State Term) :
    (runSchedule ((List.finRange h.rows).map
      (fun i=>NativeIO.backwardJob (algebra R .input) h Bool.false i.val h.rows)) s).map nativeHeap=
      FPGridBackward.run (algebra R .input) h (nativeHeap s) :=
  native_map _ _ _ (native_backward_step (algebra R .input) h) s

theorem native_weights (R : Reductions) (h : Call) (s : State Term) :
    (runSchedule ((List.finRange (Geometry.featureCount h)).map
      (fun i=>NativeIO.backwardJob (algebra R .weight) h Bool.true i.val (Geometry.featureCount h))) s).map nativeHeap=
      FPGridWeight.run (algebra R .weight) h (nativeHeap s) :=
  native_map _ _ _ (native_weight_step (algebra R .weight) h) s

theorem native_backward (R : Reductions) (h : Call) (epsilon : Term) (nonempty : h.rows≠0) (s : State Term) :
    (runSchedule (NativeIO.schedule R h Bool.true epsilon) s).map nativeHeap=
      FPTensor.backwardRun (algebra R .weight) (algebra R .input) h FPEquiv.zero (nativeHeap s) := by
  simp only [NativeIO.schedule,if_true,if_neg nonempty,FPTensor.backwardRun,FPTensor.sequence]
  rw [runSchedule_append,←native_weights R h s]
  cases ran : runSchedule
    ((List.finRange (Geometry.featureCount h)).map
      (fun i=>NativeIO.backwardJob (algebra R .weight) h Bool.true i.val (Geometry.featureCount h))) s
  · simp
  · simpa only [Option.map_some,Option.bind_some] using native_rows R h _

end
end VeriTile.Bench.Repairs.FastRmsLayernorm.IOExecution
