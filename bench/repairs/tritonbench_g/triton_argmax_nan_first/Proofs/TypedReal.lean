import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.TypedWhole
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.Real

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedReal
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source Reduction
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

abbrev Value := Math.Value

def best {n : Nat} (positive : 0<n) (input : Fin n → Value) : Fin n :=
  (Math.enumeration n positive).pick orderedGt orderedEq input

def indexed (dtype : FloatDType) {shape : TileShape} (ax : Fin shape.length)
    (values : Values Value dtype.toTileDType shape) :
    Option (Values Value dtype.toTileDType (TileShape.eraseAxis shape ax) ×
      Values Value .nat (TileShape.eraseAxis shape ax)) :=
  if positive : 0<TileShape.axisDim shape ax then
    let pick := fun out => best positive (fun lane => toFloat dtype (values (TileShape.insertAxisIndex shape ax out lane)))
    some ((fun out => values (TileShape.insertAxisIndex shape ax out (pick out))),fun out => (pick out).val)
  else none

/-- Same quotation interpreted over mathematical values. Storage casts erase
precision locally; no assertion identifies this with all native dtype results. -/
def backend (rest : Algebra Value) : Algebra Value :=
  { rest with
    negInf := ⊥
    fp32Load := id
    compareEq := fun _ _ => some orderedEq
    compareLt := fun _ _ => some (fun x y => decide (x<y))
    cast := fun _ _ _ x => x
    indexedMax := fun _ dtype => indexed dtype }

def Ready (BM row inner : Nat) (current : Fin BM → Value × Nat) (s : State Value) : Prop :=
  s.regs .int [] "pid_k"=some (fun _ => (inner : Int)) ∧
  s.regs .int [BM] "m_offset"=some (fun i => ((row+i.1.val : Nat) : Int)) ∧
  s.regs .real [BM] "max_values"=some (fun i => (current i.1).1) ∧
  s.regs .int [BM] "argmax_values"=some (fun i => Int.ofNat (current i.1).2)

def loaded (s : State Value) (X : RegionName) (M N K BM BN row inner cur : Nat)
    (i : Fin BM) (j : Fin BN) : Value :=
  if row+i.val<M ∧ cur+j.val<N then (s.mem X ((row+i.val)*N*K+(cur+j.val)*K+inner)).read .real else ⊥

def next (s : State Value) (X : RegionName) (M N K BM BN row inner cur : Nat)
    (positive : 0<BN) (current : Fin BM → Value × Nat) (i : Fin BM) : Value × Nat :=
  let column := cur+(best positive (loaded s X M N K BM BN row inner cur i)).val
  let value : Value := if row+i.val<M ∧ column<N then
    (s.mem X ((row+i.val)*N*K+column*K+inner)).read .real else ⊥
  if (current i).1<value then (value,column) else current i

theorem body_run (rest : Algebra Value) (s : State Value) (X : RegionName) (O : Region .int)
    (M N K BM BN base row inner cur : Nat) (index64 : Bool) (positive : 0<BN)
    (current : Fin BM → Value × Nat) (ready : Ready BM row inner current s) :
    ∃ t, run (backend rest) (Typed.body X O M N K BM BN base index64)
      (s.setReg "start_n" .nat [] (fun _ => cur))=some t ∧
      t.mem=s.mem ∧ Ready BM row inner (next s X M N K BM BN row inner cur positive current) t := by
  obtain ⟨hk,hm,hv,hi⟩ := ready
  simp [Typed.body,Typed.rangeBody,TypedSource.dimension,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,ComputeDType.eraseDType,backend,State.setReg,hk,hm,hv,hi,indexed,
    TileShape.axisDim,positive,Nat.ne_of_gt positive,TileShape.insertAxisIndex,TileShape.eraseAxis,
    ofFloat,toFloat,bop,numeric,numericLt,numericEq,natLt,TileShape.discreteMaximum,
    TileShape.discreteIntArgMax,orderedEq,loaded,next,Ready,Option.bind,-Nat.cast_add,-Nat.cast_mul,← Int.natCast_add,← Int.natCast_mul]
  constructor <;> funext i
  all_goals dsimp only [loaded]
  all_goals simp only [apply_ite Prod.fst,apply_ite Prod.snd,apply_ite (fun n : Nat => (n : Int)),Nat.cast_add]
  all_goals rfl

/-- Prefix values are derived from actual program IDs, including segmentation. -/
theorem prefix_run (rest : Algebra Value) (s : State Value) (X : RegionName) (O : Region .int)
    (M N K BM BN base : Nat) (index64 : Bool) :
    ∃ t, run (backend rest) ((TypedSource.dimension X O M N K BM BN base index64).surfaceBody.take 7) s=some t ∧
      t.mem=s.mem ∧ Ready BM (((s.pids 0+base)/K)*BM) ((s.pids 0+base)%K)
        (fun _ => (⊥,0)) t := by
  have nonnegative : ¬ ((s.pids 0 + base : Nat) : Int) < 0 := by omega
  have remainder := Nat.mod_add_div (s.pids 0 + base) K
  cases index64 <;>
    simp [TypedSource.dimension,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      backend,State.setReg,Ready,ofFloat,toFloat,bop,numeric,numericLt,integralFloorDiv,integralMod,ComputeDType.eraseDType,Option.bind, IntegerExpr.sameShape, -Nat.cast_add, -Nat.cast_mul, ← Int.natCast_add, ← Int.natCast_mul, -Int.natCast_mod, ← Int.natCast_ediv, ← Int.natCast_emod]
  all_goals constructor <;> funext i <;> simp [bop, Broadcast.leftIndex, Broadcast.rightIndex, -Nat.cast_add, -Nat.cast_mul, ← Int.natCast_add, ← Int.natCast_mul, -Int.natCast_mod, ← Int.natCast_ediv, ← Int.natCast_emod, Int.emod_def, nonnegative]
  all_goals
    have equation := congrArg (fun n : Nat => (n : Int)) remainder
    simp only [Nat.cast_add] at equation
    exact sub_eq_iff_eq_add.mpr equation.symm

def vectorLoop (initial : State Value) (X : RegionName) (M N K BM BN row inner : Nat)
    (positive : 0<BN) (cur : Nat) (current : Fin BM → Value × Nat) : Fin BM → Value × Nat :=
  if h : cur<N then vectorLoop initial X M N K BM BN row inner positive (cur+BN)
    (next initial X M N K BM BN row inner cur positive current) else current
termination_by N-cur
decreasing_by omega

theorem next_memory (initial s : State Value) (same : s.mem=initial.mem) (X : RegionName)
    (M N K BM BN row inner cur : Nat) (positive : 0<BN) (current : Fin BM → Value × Nat) :
    next s X M N K BM BN row inner cur positive current=
      next initial X M N K BM BN row inner cur positive current := by
  funext i
  have loads : loaded s X M N K BM BN row inner cur i=loaded initial X M N K BM BN row inner cur i := by
    funext j
    simp only [loaded,same]
  simp only [next,loads,same]

theorem range_run (rest : Algebra Value) (initial s : State Value) (same : s.mem=initial.mem)
    (X : RegionName) (O : Region .int) (M N K BM BN base row inner cur : Nat)
    (index64 : Bool) (positive : 0<BN) (current : Fin BM → Value × Nat)
    (ready : Ready BM row inner current s) :
    ∃ t, range (backend rest) "start_n" cur N BN (Typed.body X O M N K BM BN base index64) s=some t ∧
      t.mem=initial.mem ∧ Ready BM row inner
        (vectorLoop initial X M N K BM BN row inner positive cur current) t := by
  by_cases active : cur<N
  · obtain ⟨middle,runStep,memStep,readyStep⟩ :=
      body_run rest s X O M N K BM BN base row inner cur index64 positive current ready
    rw [next_memory initial s same] at readyStep
    obtain ⟨t,runRest,memRest,readyRest⟩ := range_run rest initial middle (memStep.trans same)
      X O M N K BM BN base row inner (cur+BN) index64 positive _ readyStep
    refine ⟨t,?_,memRest,?_⟩
    · rw [range_step _ _ _ _ _ _ _ (Nat.ne_of_gt positive) active,runStep]
      exact runRest
    · rw [vectorLoop,dif_pos active]
      exact readyRest
  · refine ⟨s,range_done _ _ _ _ _ _ _ (by omega),same,?_⟩
    rw [vectorLoop,dif_neg active]
    exact ready
termination_by N-cur
decreasing_by omega

def written (s : State Value) (O : Region .int) (M K BM row inner : Nat)
    (current : Fin BM → Value × Nat) : State Value :=
  (TileShape.allIndices [BM]).foldl (fun t i =>
    if row+i.1.val<M then t.write (Region.cast O) ((row+i.1.val)*K+inner)
      (.mk .int (Int.ofNat (current i.1).2)) else t) s

theorem tail_run (rest : Algebra Value) (s : State Value) (X : RegionName) (O : Region .int)
    (M N K BM BN base row inner : Nat) (index64 : Bool) (current : Fin BM → Value × Nat)
    (ready : Ready BM row inner current s) :
    ∃ t, run (backend rest) ((TypedSource.dimension X O M N K BM BN base index64).surfaceBody.drop 8) s=some t ∧
      t.mem=(written s O M K BM row inner current).mem := by
  obtain ⟨hk,hm,hv,hi⟩ := ready
  simp [TypedSource.dimension,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    backend,State.setReg,hk,hm,hv,hi,ofFloat,toFloat,bop,numeric,numericLt,natLt,ComputeDType.eraseDType,
    Option.bind,store,Region.cast,written,-Nat.cast_add,-Nat.cast_mul,← Int.natCast_add,← Int.natCast_mul]
  all_goals simp only [Typed.fold_memory]
  all_goals rfl

theorem written_memory (s initial : State Value) (same : s.mem=initial.mem) (O : Region .int)
    (M K BM row inner : Nat) (current : Fin BM → Value × Nat) :
    (written s O M K BM row inner current).mem=(written initial O M K BM row inner current).mem := by
  simp only [written,Typed.fold_memory,same]

/-- Successful execution of the entire shared readable quotation. Every
loop tile executes; only final active rows store int64 indices. -/
theorem source_run (rest : Algebra Value) (s : State Value) (X : RegionName) (O : Region .int)
    (M N K BM BN base : Nat) (index64 : Bool) (positive : 0<BN) :
    ∃ t, FP.Structural.exec (backend rest) (TypedSource.dimension X O M N K BM BN base index64) s=some t ∧
      t.mem=(written s O M K BM (((s.pids 0+base)/K)*BM) ((s.pids 0+base)%K)
        (vectorLoop s X M N K BM BN (((s.pids 0+base)/K)*BM) ((s.pids 0+base)%K) positive 0
          (fun _ => (⊥,0)))).mem := by
  obtain ⟨a,ha,ma,ra⟩ := prefix_run rest s X O M N K BM BN base index64
  obtain ⟨b,hb,mb,rb⟩ := range_run rest s a ma X O M N K BM BN base
    (((s.pids 0+base)/K)*BM) ((s.pids 0+base)%K) 0 index64 positive _ ra
  obtain ⟨t,ht,mt⟩ := tail_run rest b X O M N K BM BN base
    (((s.pids 0+base)/K)*BM) ((s.pids 0+base)%K) index64 _ rb
  refine ⟨t,?_,mt.trans (written_memory b s mb _ _ _ _ _ _ _)⟩
  let code := (TypedSource.dimension X O M N K BM BN base index64).surfaceBody
  change run (backend rest) code s=some t
  rw [←List.take_append_drop 7 code,run_append,ha,Option.bind_some]
  have splitCode : code.drop 7=[.forRange "start_n" 0 N BN (Typed.body X O M N K BM BN base index64)] ++ code.drop 8 := rfl
  rw [splitCode,run_append]
  simp only [run,step,hb,Option.bind_some]
  exact ht


#axiomsClean body_run
#axiomsClean source_run
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedReal
