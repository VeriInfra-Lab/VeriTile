import bench.repairs.tritonbench_g.cross_entropy1.Kernels
import VeriTile.Triton.Float.ExecutionProfile

namespace VeriTile.Bench.Repairs.CrossEntropy1.KernelNative
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

@[simp] theorem int_min_branch (a b : Int) : (if a<b then a else b)=min a b := by
  by_cases h : a<b
  · simp [h,min_eq_left (le_of_lt h)]
  · simp [h,min_eq_right (le_of_not_gt h)]

def engine (M : Algebra α) : Algebra α := M.withDefaultPrecision .fp32

def front (p : Parameters) := (forwardSyntax p).surfaceBody.take 10
def tailCode (p : Parameters) := (forwardSyntax p).surfaceBody.drop 10

theorem body_split (p : Parameters) :
    (forwardSyntax p).surfaceBody=front p++tailCode p := by
  exact (List.take_append_drop 10 _).symm

def logits (M : Algebra α) (p : Parameters) (initial : State α) : Values α .real [p.block] :=
  fun (lane,_) => M.cast (some .fp32) .real .real
    (if initial.pids 1*p.block+lane.val<p.cols then
      (initial.mem p.logits (p.logitsBase+initial.pids 0*p.logitsStride+
        (initial.pids 1*p.block+lane.val))).read .real else M.negInf)

def maxTile (M : Algebra α) (p : Parameters) (initial : State α) : Values α .real [] :=
  M.reduceMax (some .fp32) ⟨0,by simp⟩ Bool.false (logits M p initial)

def maxValue (M : Algebra α) (p : Parameters) (initial : State α) : α :=
  maxTile M p initial PUnit.unit

def sumTile (M : Algebra α) (p : Parameters) (initial : State α) : Values α .real [] :=
  M.reduceSum (some .fp32) (shape:=[p.block]) ⟨0,by simp⟩ Bool.false
    (fun (lane,_) => if initial.pids 1*p.block+lane.val<p.cols then
      logits M p initial (lane,PUnit.unit) else M.literal (some .fp32) .real 0.0)

def sumValue (M : Algebra α) (p : Parameters) (initial : State α) : α :=
  sumTile M p initial PUnit.unit

def lseTile (M : Algebra α) (p : Parameters) (initial : State α) : Values α .real [] :=
  fun _ => M.binary (some .fp32) .real .add
    (M.unary (some .fp32) .log
      (M.reduceSum (some .fp32) ⟨0,by simp⟩ Bool.false
        (fun j => M.unary (some .fp32) .exp
          (M.binary (some .fp32) .real .sub (logits M p initial j)
            (maxValue M p initial))) PUnit.unit))
    (maxTile M p initial PUnit.unit)

def lseValue (M : Algebra α) (p : Parameters) (initial : State α) : α :=
  lseTile M p initial PUnit.unit

@[simp] theorem nat_address (a b c : Nat) : ((a : Int)+(b : Int)*c).toNat=a+b*c := by
  norm_cast

@[simp] theorem nat_column (b c d e : Nat) :
    ((b : Int)*(c : Int)+(d : Int)<(e : Int)) ↔ b*c+d<e := by
  norm_cast

@[simp] theorem nat_column_offset (a b c : Nat) : ((a : Int)*b+c).toNat=a*b+c := by
  norm_cast

@[simp] theorem nat_output_offset (a b c d : Nat) : ((a : Int)+(b : Int)*c+d).toNat=a+b*c+d := by
  norm_cast

def ready (M : Algebra α) (p : Parameters) (initial : State α) : State α :=
  let row := initial.pids 0
  let block := initial.pids 1
  let s := initial.setReg "row_idx" .int [] (fun _ => (row : Int))
  let s := s.setReg "col_block_idx" .int [] (fun _ => (block : Int))
  let s := s.setReg "logits_ptr" .ptr [] (fun _ => (p.logits,p.logitsBase+row*p.logitsStride))
  let s := s.setReg "col_offsets" .int [p.block] (fun (lane,_) => (block : Int)*p.block+lane.val)
  let s := s.setReg "label_idx" .int [] (fun _ =>
    (initial.mem (Region.cast p.labels) (p.labelsBase+row)).read .int)
  let s := s.setReg "logits" .real [p.block] (logits M p initial)
  let s := s.setReg "max_logits" .real [] (maxTile M p initial)
  let s := if p.hasSmoothing then s.setReg "sum_logits" .real [] (sumTile M p initial) else s
  let s := s.setReg "lse" .real [] (lseTile M p initial)
  s.write p.lse (p.lseBase+block*p.rows+row) (.mk .real (lseValue M p initial))

/-- Actual source execution through its first LSE store. -/
theorem forwardPrefix (M : Algebra α) (p : Parameters)
    (initial : State α) (positive : 0<p.block) :
    run (engine M) (front p) initial=some (ready M p initial) := by
  cases hs : p.hasSmoothing
  all_goals
    simp [front,forwardSyntax,hs,ComputeKernel.surfaceBody,ready,logits,maxValue,sumValue,lseValue,maxTile,sumTile,lseTile,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,natLt,numericLt,numericLe,bop,ofFloat,toFloat,Region.cast,
      State.setReg,FP.Structural.store,TileShape.axisDim,positive,Function.comp_def]
    congr 1
    all_goals
      try simp only [bop,Broadcast.leftIndex,Broadcast.rightIndex]
      first | rfl | congr 1
      all_goals
        funext d sh n
        by_cases scalar : sh=[]
        · subst sh
          cases d <;>
            simp (config := {maxSteps := 1000000}) [bop,logits,lseTile,maxTile,sumTile,maxValue,sumValue,
              Broadcast.leftIndex,Broadcast.rightIndex]
          all_goals
            congr <;> first | rfl | simp (config := {failIfUnchanged := Bool.false}) [bop,logits,lseTile,maxTile,sumTile,maxValue,sumValue,
              Broadcast.leftIndex,Broadcast.rightIndex]
          all_goals
            funext i
            simp [bop]
        · by_cases vector : sh=[p.block]
          · subst sh
            cases d <;>
              simp (config := {maxSteps := 1000000}) [bop,logits,lseTile,maxTile,sumTile,maxValue,sumValue,
                Broadcast.leftIndex,Broadcast.rightIndex]
            all_goals
              congr <;> first | rfl | simp (config := {failIfUnchanged := Bool.false}) [bop,logits,lseTile,maxTile,sumTile,maxValue,sumValue,
                Broadcast.leftIndex,Broadcast.rightIndex]
          · have reversed : [p.block]≠sh := Ne.symm vector
            simp [scalar,reversed]

theorem ready_memory (M : Algebra α) (p : Parameters) (initial : State α) :
    (ready M p initial).mem=
      (initial.write p.lse (p.lseBase+initial.pids 1*p.rows+initial.pids 0)
        (.mk .real (lseValue M p initial))).mem := by
  cases hs : p.hasSmoothing <;> simp [ready,hs,State.write]

theorem ready_lse (M : Algebra α) (p : Parameters) (initial : State α) :
    (ready M p initial).regs .real [] "lse"=some (fun _ => lseValue M p initial) := by
  simp only [ready,State.write,State.setReg_same]
  apply congrArg some
  funext i
  cases i
  rfl

def target (p : Parameters) (initial : State α) : Int :=
  (initial.mem (Region.cast p.labels) (p.labelsBase+initial.pids 0)).read .int

def owns (p : Parameters) (initial : State α) : Prop :=
  (initial.pids 1 : Int)*p.block≤target p initial-p.classStart ∧
    target p initial-p.classStart<min (p.cols : Int) (((initial.pids 1 : Int)+1)*p.block)

def gathered (M : Algebra α) (p : Parameters) (initial : State α) : α :=
  M.fp32Load ((initial.mem p.logits (p.logitsBase+initial.pids 0*p.logitsStride+
    (target p initial-p.classStart).toNat)).read .real)

def coreLoss (M : Algebra α) (p : Parameters) (initial : State α) : α := by
  classical
  let L := if p.split then M.literal (some .fp32) .real 0.0 else lseValue M p initial
  let s := RuntimeArgs.fp initial "smoothing"
  let C := M.fromInt (some .fp32) (RuntimeArgs.classes initial)
  let sub := M.binary (some .fp32) .real .sub
  let mul := M.binary (some .fp32) .real .mul
  let div := M.binary (some .fp32) .real .div
  exact if owns p initial then
    if p.hasSmoothing then
      sub (sub L (div (mul s (sumValue M p initial)) C))
        (mul (sub (M.literal (some .fp32) .real 1) s) (gathered M p initial))
    else sub L (gathered M p initial)
  else if p.hasSmoothing then mul s (sub L (div (sumValue M p initial) C))
  else M.literal (some .fp32) .real 0.0

def lossValue (M : Algebra α) (p : Parameters) (initial : State α) : α :=
  M.cast (some .fp32) .real .real (if target p initial=p.ignored then M.literal (some .fp32) .real 0.0 else
    if p.split then coreLoss M p initial else
      M.binary (some .fp32) .real .add (coreLoss M p initial)
        (M.binary (some .fp32) .real .mul
          (M.binary (some .fp32) .real .mul (RuntimeArgs.fp initial "lse_square_scale")
            (lseValue M p initial)) (lseValue M p initial)))

@[simp] theorem ready_row (M : Algebra α) (p : Parameters) (initial : State α) :
    (ready M p initial).regs .int [] "row_idx"=some (fun _ => (initial.pids 0 : Int)) := by
  cases hs : p.hasSmoothing <;> simp [ready,hs,State.write,State.setReg]

@[simp] theorem ready_block (M : Algebra α) (p : Parameters) (initial : State α) :
    (ready M p initial).regs .int [] "col_block_idx"=some (fun _ => (initial.pids 1 : Int)) := by
  cases hs : p.hasSmoothing <;> simp [ready,hs,State.write,State.setReg]

@[simp] theorem ready_label (M : Algebra α) (p : Parameters) (initial : State α) :
    (ready M p initial).regs .int [] "label_idx"=some (fun _ => target p initial) := by
  cases hs : p.hasSmoothing <;> simp [ready,hs,State.write,State.setReg,target]

@[simp] theorem ready_pointer (M : Algebra α) (p : Parameters) (initial : State α) :
    (ready M p initial).regs .ptr [] "logits_ptr"=
      some (fun _ => (p.logits,p.logitsBase+initial.pids 0*p.logitsStride)) := by
  cases hs : p.hasSmoothing <;> simp [ready,hs,State.write,State.setReg]

@[simp] theorem ready_smoothing (M : Algebra α) (p : Parameters) (initial : State α) :
    (ready M p initial).regs .real [] "smoothing"=initial.regs .real [] "smoothing" := by
  cases hs : p.hasSmoothing <;> simp [ready,hs,State.write,State.setReg]

@[simp] theorem ready_scale (M : Algebra α) (p : Parameters) (initial : State α) :
    (ready M p initial).regs .real [] "lse_square_scale"=initial.regs .real [] "lse_square_scale" := by
  cases hs : p.hasSmoothing <;> simp [ready,hs,State.write,State.setReg]

@[simp] theorem ready_classes (M : Algebra α) (p : Parameters) (initial : State α) :
    (ready M p initial).regs .int [] "total_classes"=initial.regs .int [] "total_classes" := by
  cases hs : p.hasSmoothing <;> simp [ready,hs,State.write,State.setReg]

theorem ready_sum (M : Algebra α) (p : Parameters) (initial : State α)
    (enabled : p.hasSmoothing=Bool.true) :
    (ready M p initial).regs .real [] "sum_logits"=some (sumTile M p initial) := by
  simp [ready,enabled,State.write,State.setReg]

@[simp] theorem ready_sum_cases (M : Algebra α) (p : Parameters) (initial : State α) :
    (ready M p initial).regs .real [] "sum_logits"=
      if p.hasSmoothing then some (sumTile M p initial) else initial.regs .real [] "sum_logits" := by
  cases hs : p.hasSmoothing <;> simp [ready,hs,State.write,State.setReg]

theorem ready_logits (M : Algebra α) (p : Parameters) (initial : State α)
    (separate : p.lse≠p.logits) (offset : Nat) :
    (ready M p initial).mem p.logits offset=initial.mem p.logits offset := by
  rw [ready_memory]
  simp [State.write,separate.symm]

/-- Remaining source control flow and final loss store. The graph reads the
target through initial storage; LSE/output separation justifies that read. -/
theorem forwardTail (M : Algebra α) (p : Parameters) (initial : State α)
    (separate : p.lse≠p.logits) (values : RuntimeArgs.Values α) (abi : RuntimeArgs.Bound p initial values) :
    ∃ final, run (engine M) (tailCode p) (ready M p initial)=some final ∧
      final.mem=((ready M p initial).write p.loss
        (p.lossBase+initial.pids 1*p.rows+initial.pids 0) (.mk .real (lossValue M p initial))).mem := by
  classical
  cases hs : p.hasSmoothing <;> cases sp : p.split
  all_goals
    by_cases ignored : target p initial=p.ignored
    · refine ⟨?_,?_,?_⟩
      rotate_left
      · simp [tailCode,forwardSyntax,hs,sp,ComputeKernel.surfaceBody,run,step,
          evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
          Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
          numeric,numericEq,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
          State.setReg,FP.Structural.store,Region.cast,ignored,ready_lse,abi.smoothing,abi.squareScale,abi.totalClasses]
        rfl
      · simp [lossValue,ignored,State.write] <;> rfl
    · by_cases hit : owns p initial
      all_goals
        refine ⟨?_,?_,?_⟩
        rotate_left
        · have condition := hit
          simp only [owns,lt_min_iff] at condition
          simp [tailCode,forwardSyntax,hs,sp,ComputeKernel.surfaceBody,run,step,
            evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
            Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
            numeric,numericEq,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
            State.setReg,FP.Structural.store,Region.cast,ignored,ready_lse,
            condition,ready_logits M p initial separate,abi.smoothing,abi.squareScale,abi.totalClasses]
          rfl
        · simp [lossValue,coreLoss,ignored,hit,hs,sp,gathered,sumValue,State.write,
            RuntimeArgs.fp,RuntimeArgs.classes,abi.smoothing,abi.squareScale,abi.totalClasses] <;> rfl

/-- Complete successful execution of the actual typed forward DSL, together
with its exact tagged memory. The arithmetic graph is independent of execution. -/
theorem forward_executes (M : Algebra α) (p : Parameters) (initial : State α)
    (positive : 0<p.block) (separate : p.lse≠p.logits)
    (values : RuntimeArgs.Values α) (abi : RuntimeArgs.Bound p initial values) :
    ∃ final, exec (engine M) (forwardSyntax p) initial=some final ∧
      final.mem=((initial.write p.lse (p.lseBase+initial.pids 1*p.rows+initial.pids 0)
        (.mk .real (lseValue M p initial))).write p.loss
          (p.lossBase+initial.pids 1*p.rows+initial.pids 0) (.mk .real (lossValue M p initial))).mem := by
  obtain ⟨final,tailRun,memory⟩ := forwardTail M p initial separate values abi
  refine ⟨final,?_,?_⟩
  · change run (engine M) (forwardSyntax p).surfaceBody initial=some final
    rw [body_split,run_append,forwardPrefix M p initial positive,Option.bind_some]
    exact tailRun
  · rw [memory]
    funext r i
    simp [State.write,ready_memory]

/-- Both output cells and the entire untouched-memory frame follow from the
successful source run; no final-state premise is supplied by the caller. -/
theorem forward_outputs (M : Algebra α) (p : Parameters) (initial : State α)
    (positive : 0<p.block) (separate : p.lse≠p.logits) (outputs : p.lse≠p.loss)
    (values : RuntimeArgs.Values α) (abi : RuntimeArgs.Bound p initial values) :
    ∃ final, exec (engine M) (forwardSyntax p) initial=some final ∧
      (final.mem p.lse (p.lseBase+initial.pids 1*p.rows+initial.pids 0)).read .real=
        lseValue M p initial ∧
      (final.mem p.loss (p.lossBase+initial.pids 1*p.rows+initial.pids 0)).read .real=
        lossValue M p initial ∧
      (∀ region offset,
        (region≠p.loss ∨ offset≠p.lossBase+initial.pids 1*p.rows+initial.pids 0) →
        (region≠p.lse ∨ offset≠p.lseBase+initial.pids 1*p.rows+initial.pids 0) →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,execution,memory⟩ := forward_executes M p initial positive separate values abi
  refine ⟨final,execution,?_,?_,?_⟩
  · rw [memory]
    simp [State.write,outputs]
  · rw [memory]
    simp [State.write]
  · intro region offset lossMiss lseMiss
    rw [memory]
    simp only [State.write]
    rw [if_neg (by tauto),if_neg (by tauto)]

#axiomsClean forward_outputs

#axiomsClean forward_executes
#specNonCircular lseValue avoiding [forwardSyntax,backwardSyntax,Source.forward,Source.backward,exec,run]
#specNonCircular lossValue avoiding [forwardSyntax,backwardSyntax,Source.forward,Source.backward,exec,run]

#axiomsClean forwardTail

#axiomsClean forwardPrefix
#axiomsClean ready_memory
#axiomsClean ready_lse
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.KernelNative
