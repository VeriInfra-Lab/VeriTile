import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Backward
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Library

namespace VeriTile.Bench.Repairs.KldivOps.Host
open Storage VeriTile.Bench.Repairs.DiagSsmTriton.ByteMemory
open VeriTile.Triton.FP.Structural (Algebra)
noncomputable section
set_option maxHeartbeats 6000000

def blockSize (cols : Nat) : Nat :=
  min 16384 (if cols≤1 then 1 else 2^(Nat.log2 (cols-1)+1))

def numWarps (block : Nat) : Nat :=
  if block<2048 then 4 else if block<8192 then 8 else if block<32768 then 16 else 32

theorem block_positive (cols : Nat) : 0<blockSize cols := by
  unfold blockSize
  apply Nat.lt_min.mpr
  constructor
  · decide
  · split <;> positivity

theorem block_capped (cols : Nat) : blockSize cols≤16384 := Nat.min_le_left _ _

structure ForwardPlan (rows cols : Nat) (prediction target output : Tensor) (red : Reduction) : Type where
  predictionDense : Dense prediction
  targetDense : Dense target
  outputDense : Dense output
  predictionShape : prediction.shape=[rows,cols]
  outputShape : output.shape=red.outputShape rows cols
  predApart : output.backing.disjoint prediction.backing
  targetApart : output.backing.disjoint target.backing

def ForwardPlan.arena {rows cols : Nat} {prediction target output : Tensor} {red : Reduction}
    (plan : ForwardPlan rows cols prediction target output red) : Arena :=
  ⟨denseView prediction plan.predictionDense,denseView target plan.targetDense,
    denseView output plan.outputDense,plan.predApart,plan.targetApart⟩

structure BackwardPlan (rows cols : Nat) (target output : Tensor) : Type where
  targetDense : Dense target
  outputDense : Dense output
  targetShape : target.shape=[rows,cols]
  targetApart : output.backing.disjoint target.backing

def BackwardPlan.arena {rows cols : Nat} {target output : Tensor}
    (plan : BackwardPlan rows cols target output) : Arena :=
  ⟨denseView target plan.targetDense,denseView target plan.targetDense,
    denseView output plan.outputDense,plan.targetApart,plan.targetApart⟩

/-- Local compiler/dispatch metadata only: actual dense descriptors, original
block/warp policy and fresh output separation. No arithmetic/output law. -/
structure Compiler where
  forward : (rows cols : Nat)→(prediction target output : Tensor)→(red : Reduction)→
    Action (ForwardPlan rows cols prediction target output red)
  backward : (rows cols : Nat)→(target output : Tensor)→Action (BackwardPlan rows cols target output)
  forward_state : ∀ rows cols p t o red s plan after,forward rows cols p t o red s=.ok plan after → after=s
  backward_state : ∀ rows cols t o s plan after,backward rows cols t o s=.ok plan after → after=s

def params (rows cols : Nat) : Parameters := ⟨"Y","T","O",rows,cols,blockSize cols⟩

def launchForward {Word : Type} [Inhabited Word] (codec : Codec Word) (M : Algebra Word)
    (compiler : Compiler) (rows cols : Nat) (p t o : Tensor) (flag : Bool) (red : Reduction)
    (eps : Word) : Action Unit :=
  bind (compiler.forward rows cols p t o red) fun plan ready =>
    match Program.grid M (params rows cols) flag red.code eps (represented codec plan.arena ready.heap) with
    | none => .error "typed-source failure" ready
    | some final => .ok () {ready with heap:=commit codec plan.arena final ready.heap}

def launchBackward {Word : Type} [Inhabited Word] (codec : Codec Word) (M : Algebra Word)
    (compiler : Compiler) (rows cols : Nat) (target output : Tensor) (flag : Bool) : Action Unit :=
  bind (compiler.backward rows cols target output) fun plan ready =>
    match Backward.grid M (params rows cols) flag (represented codec plan.arena ready.heap) with
    | none => .error "typed-source failure" ready
    | some final => .ok () {ready with heap:=commit codec plan.arena final ready.heap}

def sourceActions {Word : Type} [Inhabited Word] (codec : Codec Word) (M : Algebra Word)
    (compiler : Compiler) : SourceActions Word :=
  ⟨launchForward codec M compiler,launchBackward codec M compiler⟩

structure ForwardData {Word : Type} [Inhabited Word] (codec : Codec Word) (M : Algebra Word)
    (compiler : Compiler) (rows cols : Nat) (p t o : Tensor) (flag : Bool) (red : Reduction) (eps : Word)
    (before after : State) where
  plan : ForwardPlan rows cols p t o red
  final : VeriTile.Triton.FP.Structural.State Word
  setup : compiler.forward rows cols p t o red before=.ok plan before
  source : Program.grid M (params rows cols) flag red.code eps (represented codec plan.arena before.heap)=some final
  model : final.mem=Program.gridModel M (params rows cols) flag red.code eps (List.range rows)
    (represented codec plan.arena before.heap).mem
  heap : after.heap=commit codec plan.arena final before.heap
  live : after.live=before.live
  readback : ∀ i,i<o.numel → value codec (fun x=>x) after.heap o i=(final.mem "O" i).read .real
  frame : ∀ caller,o.backing.disjoint caller → Frame before.heap after.heap caller

/-- Every successful setup continues through the full actual typed grid.
Output bytes are reconstructed from that computed state, not a desired result. -/
theorem forward_complete {Word : Type} [Inhabited Word] (codec : Codec Word) (M : Algebra Word)
    (compiler : Compiler) (rows cols : Nat) (p t o : Tensor) (flag : Bool) (red : Reduction)
    (eps : Word) (before : State) :
    Ensures (fun _ after=>Nonempty (ForwardData codec M compiler rows cols p t o flag red eps before after))
      (launchForward codec M compiler rows cols p t o flag red eps before) := by
  unfold launchForward bind
  cases hs : compiler.forward rows cols p t o red before with
  | error => trivial
  | ok plan ready =>
    have hr := compiler.forward_state rows cols p t o red before plan ready hs
    subst ready
    obtain ⟨final,he,hm⟩ := Program.grid_complete M (params rows cols) flag red.code eps
      (represented codec plan.arena before.heap) (block_positive cols)
    simp only [he]
    change Nonempty _
    refine ⟨⟨plan,final,hs,he,hm,rfl,rfl,?_,?_⟩⟩
    · intro i hi
      rw [←dense_value codec (fun x=>x) _ o plan.outputDense i hi]
      exact commit_read codec plan.arena final before.heap i hi
    · intro caller apart
      exact commit_caller_frame codec plan.arena final before.heap caller apart

theorem forward_failure_is_setup {Word : Type} [Inhabited Word] (codec : Codec Word) (M : Algebra Word)
    (compiler : Compiler) (rows cols : Nat) (p t o : Tensor) (flag : Bool) (red : Reduction)
    (eps : Word) (before failed : State) (e : String)
    (h : launchForward codec M compiler rows cols p t o flag red eps before=.error e failed) :
    compiler.forward rows cols p t o red before=.error e failed := by
  cases hs : compiler.forward rows cols p t o red before with
  | error => simpa [launchForward,bind,hs] using h
  | ok plan ready =>
    obtain ⟨final,he,hm⟩ := Program.grid_complete M (params rows cols) flag red.code eps
      (represented codec plan.arena ready.heap) (block_positive cols)
    simp [launchForward,bind,hs,he] at h

structure BackwardData {Word : Type} [Inhabited Word] (codec : Codec Word) (M : Algebra Word)
    (compiler : Compiler) (rows cols : Nat) (target output : Tensor) (flag : Bool)
    (before after : State) where
  plan : BackwardPlan rows cols target output
  final : VeriTile.Triton.FP.Structural.State Word
  setup : compiler.backward rows cols target output before=.ok plan before
  source : Backward.grid M (params rows cols) flag (represented codec plan.arena before.heap)=some final
  model : final.mem=Backward.gridModel M (params rows cols) flag (List.range rows)
    (represented codec plan.arena before.heap).mem
  heap : after.heap=commit codec plan.arena final before.heap
  live : after.live=before.live
  readback : ∀ i,i<output.numel → value codec (fun x=>x) after.heap output i=(final.mem "O" i).read .real
  frame : ∀ caller,output.backing.disjoint caller → Frame before.heap after.heap caller

theorem backward_complete {Word : Type} [Inhabited Word] (codec : Codec Word) (M : Algebra Word)
    (compiler : Compiler) (rows cols : Nat) (target output : Tensor) (flag : Bool) (before : State) :
    Ensures (fun _ after=>Nonempty (BackwardData codec M compiler rows cols target output flag before after))
      (launchBackward codec M compiler rows cols target output flag before) := by
  unfold launchBackward bind
  cases hs : compiler.backward rows cols target output before with
  | error => trivial
  | ok plan ready =>
    have hr := compiler.backward_state rows cols target output before plan ready hs
    subst ready
    obtain ⟨final,he,hm⟩ := Backward.grid_complete M (params rows cols) flag
      (represented codec plan.arena before.heap) (block_positive cols)
    simp only [he]
    change Nonempty _
    refine ⟨⟨plan,final,hs,he,hm,rfl,rfl,?_,?_⟩⟩
    · intro i hi
      rw [←dense_value codec (fun x=>x) _ output plan.outputDense i hi]
      exact commit_read codec plan.arena final before.heap i hi
    · intro caller apart
      exact commit_caller_frame codec plan.arena final before.heap caller apart

theorem backward_failure_is_setup {Word : Type} [Inhabited Word] (codec : Codec Word) (M : Algebra Word)
    (compiler : Compiler) (rows cols : Nat) (target output : Tensor) (flag : Bool)
    (before failed : State) (e : String)
    (h : launchBackward codec M compiler rows cols target output flag before=.error e failed) :
    compiler.backward rows cols target output before=.error e failed := by
  cases hs : compiler.backward rows cols target output before with
  | error => simpa [launchBackward,bind,hs] using h
  | ok plan ready =>
    obtain ⟨final,he,hm⟩ := Backward.grid_complete M (params rows cols) flag
      (represented codec plan.arena ready.heap) (block_positive cols)
    simp [launchBackward,bind,hs,he] at h

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Host
