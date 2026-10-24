import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.NativePointers
import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.NativeLoop
import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.Execution

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Native
open VeriTile.Triton FP.Structural Whole
open ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

structure Bases where
  (A B C : Nat)
def bases {α : Type} {r : Request} {s : State α} (t : TensorAPI r s) : Bases :=
  ⟨t.baseA,t.baseB,t.baseC⟩

/-- Source loads use the computed finite physical byte pointer. Ordinary tensor
layout maps its address back to a cell of the named allocation. Every byte and
this exact decoding are proved in NativePointers, including inactive lanes. -/
def aValue {α : Type} [Inhabited α] (M : Algebra α) (r : Request) (b : Bases)
    (pid : Nat) (s : State α) (k : BitVec 64) (i : FPValue.AIndex r.parameters) : α :=
  if kMask r k i.2.1.val then
    (s.mem r.A (((aPtr r b.A pid k i.1.val i.2.1.val).toNat-b.A)/4)).read .fp32
  else M.cast none .real .fp32 (M.literal none .real 0.0)
def bValue {α : Type} [Inhabited α] (M : Algebra α) (r : Request) (b : Bases)
    (pid : Nat) (s : State α) (k : BitVec 64) (i : FPValue.BIndex r.parameters) : α :=
  if kMask r k i.1.val then
    (s.mem r.B (((bPtr r b.B pid k i.1.val i.2.1.val).toNat-b.B)/4)).read .fp32
  else M.cast none .real .fp32 (M.literal none .real 0.0)
def update {α : Type} [Inhabited α] (M : Algebra α) (D : FPValue.DotAcc α)
    (r : Request) (b : Bases) (pid : Nat) (s : State α) (k : BitVec 64)
    (acc : FPValue.CIndex r.parameters → α) : FPValue.CIndex r.parameters → α :=
  D.apply (some .fp32) .fp32 .fp32 .real {} (batch := [])
    (aValue M r b pid s k) (bValue M r b pid s k) acc

/-- Independent finite integer execution of the actual source projection.
The live FP32 dot, activation and FP16 cast are the unchanged opaque primitives;
no numerical laws or successful-execution assumptions occur in this evaluator. -/
def execute {α : Type} [Inhabited α] (M : Algebra α) (D : FPValue.DotAcc α)
    (r : Request) (b : Bases) (pid : Nat) (s : State α) : Option (State α) := do
  let stop := (control r pid).stop
  let (_,acc) ← loop (update M D r b pid s) stop.toNat (word 0) stop
    (fun _ => M.literal (some .fp32) .real 0)
  pure ((TileShape.allIndices [FPValue.BM r.parameters,FPValue.BN r.parameters]).foldl
    (fun current i => if cMask r pid i.1.val i.2.1.val then
      current.write r.C (((cPtr r b.C pid i.1.val i.2.1.val).toNat-b.C)/2)
        (Cell.mk .fp16 (M.cast none .real .fp16
          (FPValue.activation M D r.parameters.ACTIVATION (acc i)))) else current)
    (instanceState r pid s))

theorem aValue_eq {α : Type} [Inhabited α] (M : Algebra α)
    (r : Request) (prepared : State α) (t : TensorAPI r prepared)
    (pid k : Nat) (hp : pid<programs r) (hpid : pid<2^31) (hk : k<r.parameters.numKBlocks)
    (s : State α) : aValue M r (bases t) pid s (word k)=
      FPValue.aValue M r.parameters .single (instanceState r pid s) k := by
  funext i
  rw [aValue,kMask_eq r prepared t pid k hp hk i.2.1]
  have ha := (a_access r prepared t pid k hp hpid hk i.1 i.2.1).2.2
  change ((aPtr r (bases t).A pid (word k) i.1.val i.2.1.val).toNat-(bases t).A)/4=_ at ha
  rw [ha]
  have hm : (k*r.config.bk+i.2.1.val<r.inner) ↔ i.2.1.val<r.inner-k*r.config.bk := by omega
  by_cases h : k*r.config.bk+i.2.1.val<r.inner
  · have h' : i.2.1.val<r.parameters.K-k*FPValue.BK r.parameters := hm.mp h
    simp only [h,decide_true,if_true]
    rw [FPValue.aValue,if_pos h']
    have hoff : safeAOffset r (FPValue.aRow r.parameters (instanceState r pid prepared) i.1) (k*r.config.bk+i.2.1.val)=FPValue.aOffset r.parameters (instanceState r pid s) k i := by
      simp only [safeAOffset,safeCoordinate,h,if_true,FPValue.aOffset]
      dsimp [Request.parameters,FPValue.BK,FPValue.aRow,FPValue.row,FPValue.pidM,
        FPValue.firstM,FPValue.groupSize,instanceState]
      ring
    rw [hoff]
    rfl
  · have h' : ¬i.2.1.val<r.parameters.K-k*FPValue.BK r.parameters := fun hh => h (hm.mpr hh)
    simp only [h,decide_false,if_false]
    rw [FPValue.aValue,if_neg h']
    rfl

theorem bValue_eq {α : Type} [Inhabited α] (M : Algebra α)
    (r : Request) (prepared : State α) (t : TensorAPI r prepared)
    (pid k : Nat) (hp : pid<programs r) (hpid : pid<2^31) (hk : k<r.parameters.numKBlocks)
    (s : State α) : bValue M r (bases t) pid s (word k)=
      FPValue.bValue M r.parameters .single (instanceState r pid s) k := by
  funext i
  rw [bValue,kMask_eq r prepared t pid k hp hk i.1]
  have ha := (b_access r prepared t pid k hp hpid hk i.1 i.2.1).2.2
  change ((bPtr r (bases t).B pid (word k) i.1.val i.2.1.val).toNat-(bases t).B)/4=_ at ha
  rw [ha]
  have hm : (k*r.config.bk+i.1.val<r.inner) ↔ i.1.val<r.inner-k*r.config.bk := by omega
  by_cases h : k*r.config.bk+i.1.val<r.inner
  · have h' : i.1.val<r.parameters.K-k*FPValue.BK r.parameters := hm.mp h
    simp only [h,decide_true,if_true]
    rw [FPValue.bValue,if_pos h']
    have hoff : safeBOffset r (k*r.config.bk+i.1.val) (FPValue.bCol r.parameters (instanceState r pid prepared) i.2.1)=FPValue.bOffset r.parameters (instanceState r pid s) k i := by
      simp only [safeBOffset,safeCoordinate,h,if_true,FPValue.bOffset]
      dsimp [Request.parameters,FPValue.BK,FPValue.bCol,FPValue.col,FPValue.pidN,
        FPValue.firstM,FPValue.groupSize,instanceState]
      ring
    rw [hoff]
    rfl
  · have h' : ¬i.1.val<r.parameters.K-k*FPValue.BK r.parameters := fun hh => h (hm.mpr hh)
    simp only [h,decide_false,if_false]
    rw [FPValue.bValue,if_neg h']
    rfl

theorem update_eq {α : Type} [Inhabited α] (M : Algebra α) (D : FPValue.DotAcc α)
    (r : Request) (prepared : State α) (t : TensorAPI r prepared)
    (pid k : Nat) (hp : pid<programs r) (hpid : pid<2^31) (hk : k<r.parameters.numKBlocks)
    (s : State α) : update M D r (bases t) pid s (word k)=
      FPValue.update M D r.parameters .single (instanceState r pid s) k := by
  funext acc
  rw [update,aValue_eq M r prepared t pid k hp hpid hk s,bValue_eq M r prepared t pid k hp hpid hk s]
  rfl

theorem execute_eq {α : Type} [Inhabited α] (M : Algebra α) (D : FPValue.DotAcc α)
    (r : Request) (prepared : State α) (t : TensorAPI r prepared)
    (pid : Nat) (hp : pid<programs r) (hpid : pid<2^31) (s : State α) :
    execute M D r (bases t) pid s=
      some (FPValue.written M D r.parameters .single (instanceState r pid s)) := by
  have hstop := stop_bound r prepared t pid hp
  have hcontrol : (control r pid).stop=word r.parameters.numKBlocks := by
    rw [control_eq r prepared t pid hp hpid]; rfl
  rw [execute,hcontrol,unsigned _ (by omega),
    loop_correspondence _ _ _ hstop (fun k hk => update_eq M D r prepared t pid k hp hpid hk s)
      r.parameters.numKBlocks 0 (by omega) (by omega)]
  simp only [Option.bind_some]
  rw [← Source.writtenSafe_eq]
  have hstep (current : State α) (i : FPValue.CIndex r.parameters) :
      (if cMask r pid i.1.val i.2.1.val then
        current.write r.C (((cPtr r (bases t).C pid i.1.val i.2.1.val).toNat-(bases t).C)/2)
          (Cell.mk .fp16 (M.cast none .real .fp16 (FPValue.activation M D r.parameters.ACTIVATION
            (FPValue.accumulation M D r.parameters .single (instanceState r pid s) i)))) else current) =
      (if FPValue.active r.parameters (instanceState r pid s) i then
        current.write r.C (Source.cSafeOffset r.parameters (instanceState r pid s) i)
          (FPValue.stored .single (FPValue.outputValue M D r.parameters .single (instanceState r pid s) i)) else current) := by
    have hmask : cMask r pid i.1.val i.2.1.val=decide (FPValue.active r.parameters (instanceState r pid s) i) :=
      cMask_eq r prepared t pid hp hpid i
    have hoff : ((cPtr r (bases t).C pid i.1.val i.2.1.val).toNat-(bases t).C)/2=
        Source.cSafeOffset r.parameters (instanceState r pid s) i := by
      have hc := (c_access r prepared t pid hp hpid i.1 i.2.1).2.2
      simpa only [Source.cSafeOffset,safeCOffset,safeCoordinate,Request.parameters,Nat.one_mul] using hc
    rw [hmask,hoff]
    simp only [decide_eq_true_eq,FPValue.stored,FPValue.outputValue]
  change some ((TileShape.allIndices _).foldl _ _)=some ((TileShape.allIndices _).foldl _ _)
  convert congrArg some (congrArg (fun f => (TileShape.allIndices _).foldl f (instanceState r pid s)) (funext (fun current => funext (hstep current)))) using 1

/-- The demanded composed theorem: actual finite ABI/control/loop/pointer
execution and the candidate Kernels.Source.kernel have equal complete memory. -/
theorem source_correspondence {α : Type} [Inhabited α] (M : Algebra α) (D : FPValue.DotAcc α)
    (r : Request) (prepared : State α) (t : TensorAPI r prepared)
    (pid : Nat) (hp : pid<programs r) (hpid : pid<2^31) (s : State α) :
    ∃ native typed, execute M D r (bases t) pid s=some native ∧
      FP.Structural.exec (FPValue.backend M D) (Source.kernel r.parameters)
        (instanceState r pid s)=some typed ∧ native.mem=typed.mem := by
  obtain ⟨typed,he,hm⟩ := Source.execution_memory M D r.parameters (instanceState r pid s)
  exact ⟨_,typed,execute_eq M D r prepared t pid hp hpid s,he,hm.symm⟩

#axiomsClean execute_eq
#axiomsClean source_correspondence
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Native
