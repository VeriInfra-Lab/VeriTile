import bench.repairs.tritonbench_g.softmax_triton3.Proofs.Model
import bench.repairs.tritonbench_g.softmax_triton1.Proofs.FPKernel
namespace VeriTile.Bench.Repairs.SoftmaxTriton3.Floating
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding blockSize numWarps
noncomputable section
set_option maxHeartbeats 3000000
abbrev access {α} [Inhabited α] := @SoftmaxTriton1.Floating.access α _

def inputRow {α} (M : Algebra α) (n B : Nat) (xs : Fin B → α) : Values α .real [B] :=
  fun j => M.cast (some .fp32) .real .real (if j.1.val<n then xs j.1 else M.negInf)

def maskRow {α} (M : Algebra α) (n B : Nat) (ms : Fin B → α) : Values α .real [B] :=
  fun j => M.cast (some .fp32) .real .real
    (if j.1.val<n then ms j.1 else M.literal (some .fp32) .real 0)

def expression {α} (M : Algebra α) (n B : Nat) (xs : Fin B → α)
    (ms : Option (Fin B → α)) (i : Fin B) : α :=
  let row := inputRow M n B xs
  let maximum := M.reduceMax (some .fp32) ⟨0,by simp⟩ Bool.false row PUnit.unit
  let shifted := fun j => M.binary (some .fp32) .real .sub (row j) maximum
  let shifted := match ms with
    | none => shifted
    | some ms => fun j => M.binary (some .fp32) .real .add (shifted j) (maskRow M n B ms j)
  let numerator : Values α .real [B] := fun j => M.unary (some .fp32) .exp (shifted j)
  let denominator := M.reduceSum (some .fp32) ⟨0,by simp⟩ Bool.false numerator PUnit.unit
  M.binary (some .fp32) .real .div (numerator (i,PUnit.unit)) denominator

def formula {α} [Inhabited α] (M : Algebra α) : Formula α := fun n B xs ms c =>
  if h : c<B then expression M n B (fun j => xs j.val) (ms.map (fun f j => f j.val)) ⟨c,h⟩ else default

theorem expression_congr {α} (M : Algebra α) (n B : Nat)
    (xs ys : Fin B → α) (mx my : Option (Fin B → α))
    (same : ∀ j,j.val<n → xs j=ys j)
    (matchMask : mx.isSome=my.isSome)
    (sameMask : ∀ f g,mx=some f → my=some g → ∀ j,j.val<n → f j=g j) (i : Fin B) :
    expression M n B xs mx i=expression M n B ys my i := by
  have row : inputRow M n B xs=inputRow M n B ys := by
    funext j
    by_cases h : j.1.val<n <;> simp [inputRow,h,same j.1]
  cases mx with
  | none =>
    cases my with
    | some g => simp at matchMask
    | none => unfold expression;rw [row]
  | some f =>
    cases my with
    | none => simp at matchMask
    | some g =>
      have mask : maskRow M n B f=maskRow M n B g := by
        funext j
        by_cases h : j.1.val<n <;> simp [maskRow,h,sameMask f g rfl rfl j.1]
      unfold expression
      rw [row]
      dsimp only
      rw [mask]

def launchState {α} (rows p : Nat) (s : State α) : State α :=
  {s with pids := fun ax => if ax=0 then p else 0
          numPids := fun ax => if ax=0 then rows else 1}

def evaluate {α} [Inhabited α] (M : Algebra α) : Evaluator (State α) := fun out inp mask rows n B p s =>
  exec M (softmax_kernel out inp (mask.getD inp) n n B mask.isSome) (launchState rows p s)

theorem row_value {α} [Inhabited α] (M : Algebra α) (out inp mask : RegionName)
    (n B : Nat) (hasMask : Bool) (hB : 0<B) (s t : State α)
    (runEq : exec M (softmax_kernel out inp mask n n B hasMask) s=some t)
    (i : Fin B) (hi : i.val<n) :
    t.mem out (s.pids 0*n+i.val)=.mk .real (expression M n B
      (fun j => (s.mem inp (s.pids 0*n+j.val)).read .real)
      (if hasMask then some (fun j => (s.mem mask (s.pids 0*n+j.val)).read .real) else none) i) := by
  cases hasMask <;>
    simp [exec,softmax_kernel,VeriTile.Bench.TritonBenchG.SoftmaxTriton3.softmax_kernel,
      run,step,evalExpr,evalComputeOp,evalOp,store,State.setReg,numeric,natLt,bop,
      TileShape.axisDim,hB,Option.bind,Function.comp_def,ComputeDType.eraseDType,toFloat,ofFloat] at runEq
  all_goals subst t
  all_goals rw [SoftmaxTriton1.Floating.masked_readback_active (shape := [B]) out
      (fun j => s.pids 0*n+j.1.val) _ (fun j => j.1.val<n) _ (i,PUnit.unit) hi
      (by
        intro j hj eq
        have e : j.1=i := Fin.ext (by simpa using eq)
        cases j with | mk k u => cases u;simp_all)]
  all_goals rfl

theorem row_frame {α} [Inhabited α] (M : Algebra α) (out inp mask : RegionName)
    (n B : Nat) (hasMask : Bool) (hB : 0<B) (s t : State α)
    (runEq : exec M (softmax_kernel out inp mask n n B hasMask) s=some t)
    (r : RegionName) (a : Nat) (miss : r≠out ∨ ∀ c,c<n → a≠s.pids 0*n+c) :
    t.mem r a=s.mem r a := by
  cases hasMask <;>
    simp [exec,softmax_kernel,VeriTile.Bench.TritonBenchG.SoftmaxTriton3.softmax_kernel,
      run,step,evalExpr,evalComputeOp,evalOp,store,State.setReg,numeric,natLt,bop,
      TileShape.axisDim,hB,Option.bind,Function.comp_def,ComputeDType.eraseDType,toFloat,ofFloat] at runEq
  all_goals subst t
  all_goals apply State.masked_scatter_frame
  all_goals rcases miss with ne|off
  all_goals first | exact Or.inl ne | exact Or.inr (fun i _ hi eq => off i.1.val hi eq)

theorem terminates {α} [Inhabited α] (M : Algebra α) (out inp : RegionName) (mask : Option RegionName)
    (rows n B p : Nat) (hB : 0<B) (s : State α) : ∃ t,evaluate M out inp mask rows n B p s=some t := by
  cases mask <;>
    simp [evaluate,exec,softmax_kernel,VeriTile.Bench.TritonBenchG.SoftmaxTriton3.softmax_kernel,
      run,step,evalExpr,evalComputeOp,evalOp,store,State.setReg,numeric,natLt,bop,
      TileShape.axisDim,hB,Option.bind,Function.comp_def,ComputeDType.eraseDType,toFloat,ofFloat]

theorem source {α} [Inhabited α] (M : Algebra α) : Source access (evaluate M) (formula M) := by
  constructor
  intro out inp mask rows n B p s t xs ms hB cover hp input masks matching run
  constructor
  · intro c hc
    have cb : c<B := lt_of_lt_of_le hc cover
    have val := row_value M out inp (mask.getD inp) n B mask.isSome hB
      (launchState rows p s) t run ⟨c,cb⟩ hc
    have read := congrArg (fun cell : Cell α => cell.read .real) val
    have same : ∀ j : Fin B,j.val<n →
        ((launchState rows p s).mem inp ((launchState rows p s).pids 0*n+j.val)).read .real=xs j.val := by
      intro j hj
      exact input j.val hj
    have mat : (if mask.isSome then some (fun j : Fin B =>
        ((launchState rows p s).mem (mask.getD inp) ((launchState rows p s).pids 0*n+j.val)).read .real)
        else none).isSome=(ms.map (fun f (j : Fin B) => f j.val)).isSome := by
      cases mask <;> cases ms <;> simp_all
    have sm : ∀ f g,
        (if mask.isSome then some (fun j : Fin B =>
          ((launchState rows p s).mem (mask.getD inp) ((launchState rows p s).pids 0*n+j.val)).read .real)
          else none)=some f → ms.map (fun f (j : Fin B) => f j.val)=some g →
        ∀ j : Fin B,j.val<n → f j=g j := by
      intro f g hf hg j hj
      cases mask with
      | none => simp at hf
      | some r =>
        cases ms with
        | none => simp at hg
        | some z =>
          have ff : (fun j : Fin B => (s.mem r (p*n+j.val)).read .real)=f := by simpa [launchState] using hf
          have gg : (fun j : Fin B => z j.val)=g := by simpa using hg
          rw [← ff,← gg]
          exact masks r rfl j.val hj
    have congruence := expression_congr M n B _ (fun j => xs j.val) _ _ same mat sm ⟨c,cb⟩
    simpa [formula,access,SoftmaxTriton1.Floating.access,Access.read,Cell.read,cb,launchState]
      using read.trans congruence
  · intro r a miss
    exact row_frame M out inp (mask.getD inp) n B mask.isSome hB
      (launchState rows p s) t run r a miss

#axiomsClean source
#axiomsClean row_value
#axiomsClean row_frame
#axiomsClean terminates
end
end VeriTile.Bench.Repairs.SoftmaxTriton3.Floating
