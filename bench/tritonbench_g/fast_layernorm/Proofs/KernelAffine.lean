import bench.tritonbench_g.fast_layernorm.Proofs.KernelRows
import VeriTile.Triton.Float.ScalarInt64While

namespace VeriTile.Bench.Repairs.FastLayernorm.KernelAffine
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open KernelMemory
noncomputable section
open scoped Classical
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def kernel (rows N wc bc : Nat) (bias : Bool) := affineKernel "dw" "db" "dy" "x" "r" "mu" rows N wc bc bias
def prefixCode (rows N wc bc : Nat) (bias : Bool) := (kernel rows N wc bc bias).surfaceBody.take 13
def body (rows N wc bc : Nat) (bias : Bool) : List ComputeStmt :=
  match ((kernel rows N wc bc bias).surfaceBody.drop 13).head? with
  | some (.forRangeInt _ _ _ _ code) => code
  | _ => []
def tail (rows N wc bc : Nat) (bias : Bool) := (kernel rows N wc bc bias).surfaceBody.drop 14
theorem source_body (rows N wc bc : Nat) (bias : Bool) :
    (kernel rows N wc bc bias).surfaceBody=prefixCode rows N wc bc bias ++
      [.forRangeInt "row" (.ref .int [] "row") (.ref .int [] "n_rows")
        (.castIntToNat (.castNatToInt (.constNat 1))) (body rows N wc bc bias)] ++
      tail rows N wc bc bias := by rfl
def columns (pid : Nat) : Values α .int [128] := fun i => (pid : Int)*128+i.1.val
def mask (N pid : Nat) : Values α .bool [128] := fun i => decide ((pid : Int)*128+(i.1.val : Int)<(N : Int))
@[simp] theorem column_lt (pid lane N : Nat) :
    ((pid : Int)*128+(lane : Int)<(N : Int)) ↔ pid*128+lane<N := by norm_cast
@[simp] theorem row_address (row N col : Nat) :
    ((row : Int)*N+(col : Int)).toNat=row*N+col := by norm_cast
@[simp] theorem lane_address (row N pid lane : Nat) :
    ((row : Int)*N+((pid : Int)*128+lane)).toNat=row*N+(pid*128+lane) := by norm_cast

/-- Actual caller geometry; source casts retain these same integer values. -/
def arguments (rows N wc bc : Nat) (u : State α) : State α :=
  let u:=u.setReg "n_rows" .int [] (fun _=>(rows : Int))
  let u:=u.setReg "n_cols" .int [] (fun _=>(N : Int))
  let u:=u.setReg "weight_count" .int [] (fun _=>(wc : Int))
  u.setReg "bias_count" .int [] (fun _=>(bc : Int))

def metadataFrame (u v : State α) : Prop :=
  ∀ name, name∈["n_rows","n_cols","weight_count","bias_count"] →
    v.regs .int [] name=u.regs .int [] name

def raw (N pid : Nat) (u : State α) (r : RegionName) (row : Nat) (i : TileIndex [128]) : α :=
  (u.mem r (row*N+(if pid*128+i.1.val<N then pid*128+i.1.val else 0))).read .real
def saved (u : State α) (r : RegionName) (row : Nat) : α := (u.mem r row).read .real
def carry (M : Algebra α) (N pid : Nat) (u : State α) (n : Nat) (i : TileIndex [128]) : α × α :=
  KernelReference.affineCarry M (decide (pid*128+i.1.val<N))
    (fun row => raw N pid u "x" row i) (fun row => raw N pid u "dy" row i)
    (saved u "mu") (saved u "r") n

def next (M : Algebra α) (N pid row : Nat) (u : State α) (a b : Values α .real [128])
    (i : TileIndex [128]) : α × α :=
  (KernelReference.add M (a i) (KernelReference.affineTerm M (decide (pid*128+i.1.val<N))
    (raw N pid u "x" row i) (raw N pid u "dy" row i) (saved u "mu" row) (saved u "r" row)),
   KernelReference.add M (b i) (KernelReference.loaded M
    (if pid*128+i.1.val<N then raw N pid u "dy" row i else KernelReference.literal M 0)))

theorem iteration (M : Algebra α) (rows N wc bc pid row : Nat) (bias : Bool) (u : State α)
    (a b : Values α .real [128])
    (hc : u.regs .int [128] "columns"=some (columns pid))
    (hm : u.regs .bool [128] "active"=some (mask N pid))
    (ha : u.regs .real [128] "dw"=some a) (hb : u.regs .real [128] "dbias"=some b)
    (nc : u.regs .int [] "n_cols"=some (fun _=>(N : Int))) :
    ∃ v, run (engine M) (body rows N wc bc bias) (u.setReg "row" .int [] (fun _ => (row : Int)))=some v ∧
      v.mem=u.mem ∧ v.regs .int [128] "columns"=some (columns pid) ∧
      v.regs .bool [128] "active"=some (mask N pid) ∧
      v.regs .real [128] "dw"=some (fun i => (next M N pid row u a b i).1) ∧
      v.regs .real [128] "dbias"=some (fun i => (next M N pid row u a b i).2) ∧
      metadataFrame u v ∧
      v.setReg "row" .int [] (fun _=>((row+1 : Nat) : Int))=v := by
  simp [body,kernel,affineKernel,affineBodyPastedKernel,ComputeKernel.withScalarArguments,ScalarArgument.statement,ScalarArgument.integer,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,natLt,numericLt,numericLe,numericEq,bop,Region.cast,hc,hm,ha,hb,nc,
    next,columns,mask,raw,saved,KernelReference.affineTerm,
    KernelReference.literal,KernelReference.cast,KernelReference.loaded,KernelReference.count,KernelReference.add,KernelReference.sub,KernelReference.mul]
  constructor
  · funext i
    rcases i with ⟨j,⟨⟩⟩
    by_cases hi : pid*128+j.val<N <;>
      simp [bop,Broadcast.leftIndex,Broadcast.rightIndex,ofFloat,toFloat,hi] <;> norm_num
  constructor
  · funext i
    rcases i with ⟨j,⟨⟩⟩
    by_cases hi : pid*128+j.val<N <;>
      simp [bop,Broadcast.leftIndex,Broadcast.rightIndex,ofFloat,toFloat,hi] <;> norm_num
  constructor
  · intro name member
    simp only [List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl|rfl|rfl <;> rfl
  · simpa only [Int.natCast_add,Int.natCast_one] using
      ScalarInt64While.counter_idempotent _ "row" ((row : Int)+1)

structure Holds (M : Algebra α) (rows N pid : Nat) (initial : State α) (n : Nat) (u : State α) : Prop where
  bound : n≤rows
  col : u.regs .int [128] "columns"=some (columns pid)
  mask : u.regs .bool [128] "active"=some (mask N pid)
  dw : u.regs .real [128] "dw"=some (fun i => (carry M N pid initial n i).1)
  db : u.regs .real [128] "dbias"=some (fun i => (carry M N pid initial n i).2)
  mem : u.mem=initial.mem
  args : metadataFrame initial u
  counter : u.setReg "row" .int [] (fun _=>((n : Nat) : Int))=u

/-- Construct the actual complete source prefix and its initialized loop
invariant, without comparing unrelated register-dispatch functions. -/
theorem prefix_executes (M : Algebra α) (rows N wc bc : Nat) (bias : Bool) (u : State α) :
    ∃ v, run (engine M) (prefixCode rows N wc bc bias) u=some v ∧
      Holds M rows N (u.pids 0) (arguments rows N wc bc u) 0 v := by
  refine ⟨?_,?_,?_⟩
  rotate_left
  · simp [prefixCode,kernel,affineKernel,affineBodyPastedKernel,ComputeKernel.withScalarArguments,
      ScalarArgument.statement,ScalarArgument.integer,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
      Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,natLt,numericLt,numericLe,numericEq,bop,Region.cast,
      columns,mask,KernelReference.literal,Broadcast.leftIndex,Broadcast.rightIndex]
    rfl
  · constructor
    · exact Nat.zero_le _
    · rfl
    · rfl
    · simp [carry,KernelReference.affineCarry,KernelReference.literal]
    · simp [carry,KernelReference.affineCarry,KernelReference.literal]
    · rfl
    · intro name member
      simp only [List.mem_cons,List.mem_singleton,List.not_mem_nil,or_false] at member
      rcases member with rfl|rfl|rfl|rfl <;> rfl
    · exact ScalarInt64While.counter_idempotent _ _ _

/-- Construct both the actual signed range execution and the corresponding
source while derivation, preserving each accumulator and caller register. -/
theorem loop_executes (M : Algebra α) (rows N wc bc pid : Nat) (bias : Bool) (initial : State α)
    (initialN : initial.regs .int [] "n_cols"=some (fun _=>(N : Int)))
    (n : Nat) (u : State α) (inv : Holds M rows N pid initial n u) :
    ∃ v, rangeInt (engine M) "row" (n : Int) (rows : Int) 1 (body rows N wc bc bias) u=some v ∧
      Holds M rows N pid initial rows v ∧
      ScalarInt64While.Executes (engine M) "row" (rows : Int) (body rows N wc bc bias) u v := by
  have read : u.regs .int [] "row"=some (fun _=>(n : Int)) := by
    conv_lhs => rw [← inv.counter]
    exact State.setReg_same _ _ _ _ _
  by_cases hn : n<rows
  · have nc := (inv.args "n_cols" (by simp)).trans initialN
    obtain ⟨v,he,hmem,hcol,hmask,ha,hb,hargs,hcounter⟩ := iteration M rows N wc bc pid n bias u
      (fun i => (carry M N pid initial n i).1) (fun i => (carry M N pid initial n i).2)
      inv.col inv.mask inv.dw inv.db nc
    rw [inv.counter] at he
    have changeCarry : next M N pid n u
        (fun i => (carry M N pid initial n i).1) (fun i => (carry M N pid initial n i).2)=
        carry M N pid initial (n+1) := by
      funext i
      simp [next,carry,KernelReference.affineCarry,raw,saved,inv.mem]
    have nextInv : Holds M rows N pid initial (n+1) v :=
      ⟨Nat.succ_le_of_lt hn,hcol,hmask,by rw [changeCarry] at ha; exact ha,
        by rw [changeCarry] at hb; exact hb,hmem.trans inv.mem,
        fun name member=>(hargs name member).trans (inv.args name member),hcounter⟩
    obtain ⟨w,hw,hf,hwhile⟩ := loop_executes M rows N wc bc pid bias initial initialN (n+1) v nextInv
    refine ⟨w,?_,hf,.next read (by exact_mod_cast hn) he hwhile⟩
    rw [rangeInt]
    simp only [show (1 : Nat)≠0 by decide,if_false,
      show (n : Int)<(rows : Int) by exact_mod_cast hn,if_true,inv.counter,he,Option.bind_some]
    simpa only [Int.natCast_add,Int.natCast_one] using hw
  · have eq : n=rows := Nat.le_antisymm inv.bound (Nat.le_of_not_gt hn)
    refine ⟨u,?_,by simpa [eq] using inv,.stop read (by omega)⟩
    rw [rangeInt]
    simp only [show (1 : Nat)≠0 by decide,if_false,
      show ¬(n : Int)<(rows : Int) by exact_mod_cast hn,if_false]
termination_by rows-n

def heap (M : Algebra α) (rows N wc bc pid : Nat) (bias : Bool) (u : State α) :=
  let w := commit "dw" (fun i : TileIndex [128] => pid*128+i.1.val)
    (fun i => Cell.mk .real ((carry M N pid u rows i).1))
    (fun i => pid*128+i.1.val<wc) (TileShape.allIndices [128]) u.mem
  if bias then commit "db" (fun i : TileIndex [128] => pid*128+i.1.val)
    (fun i => Cell.mk .real ((carry M N pid u rows i).2))
    (fun i => pid*128+i.1.val<bc) (TileShape.allIndices [128]) w else w

theorem tail_executes (M : Algebra α) (rows N wc bc pid : Nat) (bias : Bool) (initial u : State α)
    (inv : Holds M rows N pid initial rows u)
    (weight : initial.regs .int [] "weight_count"=some (fun _=>(wc : Int)))
    (biasCount : initial.regs .int [] "bias_count"=some (fun _=>(bc : Int))) :
    ∃ v, run (engine M) (tail rows N wc bc bias) u=some v ∧
      v.mem=heap M rows N wc bc pid bias initial := by
  have hw := (inv.args "weight_count" (by simp)).trans weight
  have hb := (inv.args "bias_count" (by simp)).trans biasCount
  cases bias
  all_goals
    simp [tail,kernel,affineKernel,affineBodyPastedKernel,ComputeKernel.withScalarArguments,ScalarArgument.statement,ScalarArgument.integer,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,natLt,numericLt,numericLe,numericEq,bop,Region.cast,State.setReg,FP.Structural.store,
      inv.col,inv.mask,inv.dw,inv.db,inv.mem,hw,hb,heap,columns,mask,KernelReference.cast,fold_mem,
      Broadcast.leftIndex,Broadcast.rightIndex,ofFloat,toFloat]
    <;> rfl

theorem source_memory (M : Algebra α) (rows N wc bc : Nat) (bias : Bool) (u : State α) :
    ∃ v, FP.Structural.exec (engine M) (kernel rows N wc bc bias) u=some v ∧
      v.mem=heap M rows N wc bc (u.pids 0) bias u := by
  obtain ⟨first,hfirst,initialized⟩ := prefix_executes M rows N wc bc bias u
  have nc : (arguments rows N wc bc u).regs .int [] "n_cols"=some (fun _=>(N : Int)) := by
    simp [arguments,State.setReg]
  have weight : (arguments rows N wc bc u).regs .int [] "weight_count"=some (fun _=>(wc : Int)) := by
    simp [arguments,State.setReg]
  have biasCount : (arguments rows N wc bc u).regs .int [] "bias_count"=some (fun _=>(bc : Int)) := by
    simp [arguments,State.setReg]
  obtain ⟨v,he,inv,_⟩ := loop_executes M rows N wc bc (u.pids 0) bias
    (arguments rows N wc bc u) nc 0 first initialized
  obtain ⟨w,hw,hmem⟩ := tail_executes M rows N wc bc (u.pids 0) bias
    (arguments rows N wc bc u) v inv weight biasCount
  refine ⟨w,?_,by simpa only [heap,carry,raw,saved,arguments,State.setReg_mem] using hmem⟩
  have hr := initialized.counter
  have hrows : first.regs .int [] "n_rows"=some (fun _=>(rows : Int)) := by
    rw [initialized.args "n_rows" (by simp)]
    simp [arguments,State.setReg]
  have hrow : first.regs .int [] "row"=some (fun _=>0) := by
    conv_lhs => rw [← hr]
    exact State.setReg_same _ _ _ _ _
  change run (engine M) (kernel rows N wc bc bias).surfaceBody u=some w
  rw [source_body,run_append,run_append,hfirst]
  simp only [run,step,FP.Structural.evalOp,hrows,hrow,Option.bind_some]
  simp only [bind,pure,Option.bind,Option.pure_def]
  simp only [Int.ofNat_eq_natCast,Int.toNat_natCast]
  norm_num only at he ⊢
  rw [he]
  exact hw

/-- The original source while really terminates on the same initialized state
and reaches the same complete accumulator invariant as the range lowering. -/
theorem source_while (M : Algebra α) (rows N wc bc : Nat) (bias : Bool) (u : State α) :
    ∃ first last, run (engine M) (prefixCode rows N wc bc bias) u=some first ∧
      ScalarInt64While.Executes (engine M) "row" (rows : Int) (body rows N wc bc bias) first last ∧
      Holds M rows N (u.pids 0) (arguments rows N wc bc u) rows last := by
  obtain ⟨first,hfirst,initialized⟩ := prefix_executes M rows N wc bc bias u
  have nc : (arguments rows N wc bc u).regs .int [] "n_cols"=some (fun _=>(N : Int)) := by
    simp [arguments,State.setReg]
  obtain ⟨last,_,inv,executed⟩ := loop_executes M rows N wc bc (u.pids 0) bias
    (arguments rows N wc bc u) nc 0 first initialized
  exact ⟨first,last,hfirst,executed,inv⟩

#axiomsClean source_while
#axiomsClean source_memory
end
end VeriTile.Bench.Repairs.FastLayernorm.KernelAffine
