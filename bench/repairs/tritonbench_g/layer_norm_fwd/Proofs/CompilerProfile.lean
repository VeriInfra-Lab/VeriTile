import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.CompilerRecipes
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeMachineHistory
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeScheduleReturn

namespace VeriTile.Bench.Repairs.LayerNormFwd.Compiler
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open NativeRow
noncomputable section
set_option maxHeartbeats 3000000

/-- The source pin is part of the compiler request, not a caller-chosen label. -/
def sourceSHA256 : String := "e23770ea53c68c79871d768e055478d8f762291d2c7e727eea50544198526eda"

inductive Root where
  | y (col : Nat) | y1 (col : Nat) | residual (col : Nat)
  | mean | rstd
  | random (second : Bool) (col : Nat)
  | dropoutP
  deriving DecidableEq

def Root.active (f : Flags) (N : Nat) : Root → Prop
  | .y col => col<N
  | .y1 col => f.w1=Bool.true ∧ col<N
  | .residual col => f.residualOut=Bool.true ∧ col<N
  | .mean => f.rms=Bool.false
  | .rstd => True
  | .random second col => f.dropout=Bool.true ∧ (second=Bool.true → f.x1=Bool.true) ∧ col<N
  | .dropoutP => f.dropout=Bool.true

def recipe (schedule : NativeMachine.Schedule block) (f : Flags) (N : Nat) : Root → Term
  | .y col => y schedule.reductions schedule.plan f N Bool.false col
  | .y1 col => y schedule.reductions schedule.plan1 f N Bool.true col
  | .residual col => residual f N col
  | .mean => storedMean schedule.reductions f N
  | .rstd => storedRstd schedule.reductions f N
  | .random second col => random second col
  | .dropoutP => .input .dropoutP

/-- Row-relative targets. The already checked address machine supplies the
actual row, stride and element width; mask true addresses use (M+row)*N+col. -/
inductive Destination where
  | y (col : Nat) | y1 (col : Nat) | residualOut (col : Nat)
  | mean | rstd | mask (second : Bool) (col : Nat)
  deriving DecidableEq

inductive StoreOperand where
  | real32 (value : Code)
  | boolGreater (left right : Code)

def sourceStore (f : Flags) (N : Nat) (code : Root → Code) : Destination → Option StoreOperand
  | .y col => if col<N then some (.real32 (code (.y col))) else none
  | .y1 col => if f.w1 && decide (col<N) then some (.real32 (code (.y1 col))) else none
  | .residualOut col => if f.residualOut && decide (col<N) then some (.real32 (code (.residual col))) else none
  | .mean => if f.rms then none else some (.real32 (code .mean))
  | .rstd => some (.real32 (code .rstd))
  | .mask second col => if f.dropout && f.dropoutMask && (!second || f.x1) && decide (col<N) then
      some (.boolGreater (code (.random second col)) (code .dropoutP)) else none

/-- Ordinary selected-compiler contract. Trees/affine plans come from the
artifact. Each active operand is recursively bound to individual emitted
opcodes and source identities. Each store has its source mask, target and
register operand. No memory state or computed tensor occurs in these fields.
This interface is not a per-task proof of the compiler implementation. -/
structure Artifact (f : Flags) (N block : Nat) where
  warps : Nat
  supportedWarps : warps ∈ [1,2,4,8,16,32]
  schedule : NativeMachine.Schedule block
  code : Root → Code
  operands : ∀ root, root.active f N → Binds (recipe schedule f N root) (code root)
  stores : Destination → Option StoreOperand
  storeOperands : ∀ destination, stores destination=sourceStore f N code destination

theorem Artifact.expression (artifact : Artifact f N block) (root : Root) (h : root.active f N)
    (x : Inputs α) (M row : Nat) (eps p : α) :
    (artifact.code root).expression x M row eps p=(recipe artifact.schedule f N root).instantiate x M row eps p :=
  (artifact.operands root h).expression x M row eps p

/-- Input-only reading of the actual emitted dependency trees. Returned seed
and fallback fields retain the original snapshot; masks evaluate the actual
comparison operands instead of treating the mask as an FP value. -/
def Artifact.observe {α : Type} [Inhabited α] (artifact : Artifact f N block) (ops : Ops α)
    (r : Regions) (stride : Strides) (M : Nat) (eps p : α) (s : State α) :
    TypedSource.Result α M N (HostReturn.seedCount f M) :=
  let x := NativeSchedule.snapshot r stride s
  let value := fun row root => ((artifact.code root).expression x M row eps p).eval ops
  let keep := fun row second col => ops.greater (value row (.random second col)) (value row .dropoutP)
  { y := fun i j => .mk .real (value i.val (.y j.val))
    y1 := if f.w1 then some (fun i j => .mk .real (value i.val (.y1 j.val))) else none
    mean := if f.rms then none else some (fun i => .mk .real (value i.val .mean))
    rstd := fun i => .mk .real (value i.val .rstd)
    residual := if f.residualOut then (fun i j => .mk .real (value i.val (.residual j.val)))
      else (fun i j => s.mem r.x (i.val*stride.x+j.val))
    seeds := if f.dropout then some (fun i => s.mem r.seeds i.val) else none
    mask := if f.dropout && f.dropoutMask then some (fun i j => .mk .bool (keep i.val Bool.false j.val)) else none
    mask1 := if f.dropout && f.dropoutMask && f.x1 then some (fun i j => .mk .bool (keep i.val Bool.true j.val)) else none }

/-- Complete graph-level agreement follows from the local instruction
bindings, for arbitrary opaque primitive implementations. In particular no
FP-to-Real law or complete-kernel/output equation is a compiler premise. -/
theorem Artifact.observe_eq {α : Type} [Inhabited α] (artifact : Artifact f N block) (ops : Ops α)
    (r : Regions) (stride : Strides) (M : Nat) (eps p : α) (s : State α) :
    artifact.observe ops r stride M eps p s=
      NativeSchedule.oracle ops artifact.schedule.reductions artifact.schedule.plan artifact.schedule.plan1
        f r stride M N eps p s := by
  let x := NativeSchedule.snapshot r stride s
  have hy := fun (row : Nat) (j : Fin N) => artifact.expression (.y j.val) j.isLt x M row eps p
  have hy1 := fun (hf : f.w1=Bool.true) (row : Nat) (j : Fin N) =>
    artifact.expression (.y1 j.val) ⟨hf,j.isLt⟩ x M row eps p
  have hm := fun (hf : f.rms=Bool.false) (row : Nat) => artifact.expression .mean hf x M row eps p
  have hs := fun (row : Nat) => artifact.expression .rstd trivial x M row eps p
  have hr := fun (hf : f.residualOut=Bool.true) (row : Nat) (j : Fin N) =>
    artifact.expression (.residual j.val) ⟨hf,j.isLt⟩ x M row eps p
  have hp := fun (hf : f.dropout=Bool.true) (row : Nat) => artifact.expression .dropoutP hf x M row eps p
  have hk := fun (hf : f.dropout=Bool.true) (row : Nat) (j : Fin N) =>
    artifact.expression (.random Bool.false j.val) ⟨hf,by simp,j.isLt⟩ x M row eps p
  have hk1 := fun (hf : f.dropout=Bool.true) (hx : f.x1=Bool.true) (row : Nat) (j : Fin N) =>
    artifact.expression (.random Bool.true j.val) ⟨hf,fun _ => hx,j.isLt⟩ x M row eps p
  apply TypedSource.Result.ext
  all_goals dsimp only [Artifact.observe,NativeSchedule.oracle]
  · funext i j
    rw [hy i.val j]
    simp only [recipe,y_instantiate,x]
  · cases hf : f.w1
    · simp [hf]
    · simp only [hf,ite_true]
      congr 1
      funext i j
      rw [hy1 hf i.val j]
      simp only [recipe,y1_instantiate,x]
  · cases hf : f.rms
    · simp only [hf,Bool.false_eq_true,ite_false]
      congr 1
      funext i
      rw [hm hf i.val]
      simp only [recipe,storedMean,Term.instantiate,mean_instantiate,Expr.eval,x]
    · simp [hf]
  · funext i
    rw [hs i.val]
    simp only [recipe,storedRstd,Term.instantiate,rstd_instantiate,Expr.eval,x]
  · cases hf : f.residualOut
    · simp [hf]
    · simp only [hf,ite_true]
      funext i j
      rw [hr hf i.val j]
      simp only [recipe,residual,Term.instantiate,combined_instantiate,Expr.eval,x]
  · cases hd : f.dropout <;> cases hm : f.dropoutMask
    all_goals simp only [hd,hm,Bool.and_true,Bool.and_false,Bool.false_and,Bool.true_and,Bool.false_eq_true,ite_false,ite_true]
    congr 1
    funext i j
    rw [hk hd i.val j,hp hd i.val]
    rfl
  · cases hd : f.dropout <;> cases hm : f.dropoutMask <;> cases hx : f.x1
    all_goals simp only [hd,hm,hx,Bool.and_true,Bool.and_false,Bool.false_and,Bool.true_and,Bool.false_eq_true,ite_false,ite_true]
    congr 1
    funext i j
    rw [hk1 hd hx i.val j,hp hd i.val]
    rfl

#axiomsClean Artifact.expression
#axiomsClean Artifact.observe_eq
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.Compiler
