import bench.repairs.tritonbench_g.fast_layernorm.Proofs.KernelRows

namespace VeriTile.Bench.Repairs.FastLayernorm.AffineDirect
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open KernelMemory
noncomputable section
open scoped Classical
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def kernel (rows N wc bc : Nat) (bias : Bool) := affineDirectOffsets "dw" "db" "dy" "x" "r" "mu" rows N wc bc bias
def prefixCode (rows N wc bc : Nat) (bias : Bool) := (kernel rows N wc bc bias).surfaceBody.take 4
def body (rows N wc bc : Nat) (bias : Bool) : List ComputeStmt :=
  match ((kernel rows N wc bc bias).surfaceBody.drop 4).head? with
  | some (.forRangeDyn _ _ _ _ code) => code
  | _ => []
def tail (rows N wc bc : Nat) (bias : Bool) := (kernel rows N wc bc bias).surfaceBody.drop 5
theorem source_body (rows N wc bc : Nat) (bias : Bool) :
    (kernel rows N wc bc bias).surfaceBody=prefixCode rows N wc bc bias ++
      [.forRangeDyn "row" (.constNat 0) (.constNat rows) (.constNat 1) (body rows N wc bc bias)] ++
      tail rows N wc bc bias := by rfl
def columns (pid : Nat) : Values α .nat [128] := fun i => pid*128+i.1.val
def mask (N pid : Nat) : Values α .bool [128] := fun i => decide (pid*128+i.1.val<N)
def start (M : Algebra α) (N : Nat) (u : State α) : State α :=
  (((u.setReg "columns" .nat [128] (columns (u.pids 0))).setReg "active" .bool [128] (mask N (u.pids 0))).setReg
    "dw" .real [128] (fun _ => KernelReference.literal M 0)).setReg "dbias" .real [128] (fun _ => KernelReference.literal M 0)
theorem prefix_executes (M : Algebra α) (rows N wc bc : Nat) (bias : Bool) (u : State α) :
    run (engine M) (prefixCode rows N wc bc bias) u=some (start M N u) := by
  simp [prefixCode,kernel,affineDirectOffsets,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,natLt,numericLt,numericLe,numericEq,bop,Region.cast,State.setReg,
    columns,mask,start,KernelReference.literal]
  rfl

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
   KernelReference.add M (b i) (KernelReference.cast M
    (if pid*128+i.1.val<N then raw N pid u "dy" row i else KernelReference.literal M 0)))

theorem iteration (M : Algebra α) (rows N wc bc pid row : Nat) (bias : Bool) (u : State α)
    (a b : Values α .real [128])
    (hc : u.regs .nat [128] "columns"=some (columns pid))
    (hm : u.regs .bool [128] "active"=some (mask N pid))
    (ha : u.regs .real [128] "dw"=some a) (hb : u.regs .real [128] "dbias"=some b) :
    ∃ v, run (engine M) (body rows N wc bc bias) (u.setReg "row" .nat [] (fun _ => row))=some v ∧
      v.mem=u.mem ∧ v.regs .nat [128] "columns"=some (columns pid) ∧
      v.regs .bool [128] "active"=some (mask N pid) ∧
      v.regs .real [128] "dw"=some (fun i => (next M N pid row u a b i).1) ∧
      v.regs .real [128] "dbias"=some (fun i => (next M N pid row u a b i).2) := by
  simp [body,kernel,affineDirectOffsets,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,natLt,numericLt,numericLe,numericEq,bop,Region.cast,State.setReg,hc,hm,ha,hb,
    next,columns,mask,raw,saved,KernelReference.affineTerm,
    KernelReference.literal,KernelReference.cast,KernelReference.add,KernelReference.sub,KernelReference.mul]
  constructor
  all_goals
    funext i
    simp only [bop,Broadcast.leftIndex,Broadcast.rightIndex,ofFloat,toFloat]
    by_cases hi : pid*128+i.1.val<N <;> simp [hi] <;> norm_num <;> rfl

structure Holds (M : Algebra α) (rows N pid : Nat) (initial : State α) (n : Nat) (u : State α) : Prop where
  bound : n≤rows
  col : u.regs .nat [128] "columns"=some (columns pid)
  mask : u.regs .bool [128] "active"=some (mask N pid)
  dw : u.regs .real [128] "dw"=some (fun i => (carry M N pid initial n i).1)
  db : u.regs .real [128] "dbias"=some (fun i => (carry M N pid initial n i).2)
  mem : u.mem=initial.mem

theorem start_holds (M : Algebra α) (rows N : Nat) (u : State α) :
    Holds M rows N (u.pids 0) u 0 (start M N u) := by
  refine ⟨Nat.zero_le _,?_,?_,?_,?_,rfl⟩
  all_goals simp [start,carry,KernelReference.affineCarry]

theorem loop_executes (M : Algebra α) (rows N wc bc pid : Nat) (bias : Bool) (initial : State α)
    (n : Nat) (u : State α) (inv : Holds M rows N pid initial n u) :
    ∃ v, FP.Structural.range (engine M) "row" n rows 1 (body rows N wc bc bias) u=some v ∧
      Holds M rows N pid initial rows v := by
  by_cases hn : n<rows
  · obtain ⟨v,he,hmem,hcol,hmask,ha,hb⟩ := iteration M rows N wc bc pid n bias u
      (fun i => (carry M N pid initial n i).1) (fun i => (carry M N pid initial n i).2)
      inv.col inv.mask inv.dw inv.db
    have changeCarry : next M N pid n u
        (fun i => (carry M N pid initial n i).1) (fun i => (carry M N pid initial n i).2)=
        carry M N pid initial (n+1) := by
      funext i
      simp [next,carry,KernelReference.affineCarry,raw,saved,inv.mem]
    have nextInv : Holds M rows N pid initial (n+1) v :=
      ⟨Nat.succ_le_of_lt hn,hcol,hmask,by rw [changeCarry] at ha; exact ha,
        by rw [changeCarry] at hb; exact hb,hmem.trans inv.mem⟩
    obtain ⟨w,hw,hf⟩ := loop_executes M rows N wc bc pid bias initial (n+1) v nextInv
    refine ⟨w,?_,hf⟩
    rw [FP.Structural.range]
    simpa [hn,he] using hw
  · have eq : n=rows := Nat.le_antisymm inv.bound (Nat.le_of_not_gt hn)
    refine ⟨u,?_,by simpa [eq] using inv⟩
    rw [FP.Structural.range]
    simp [hn]
termination_by rows-n

def heap (M : Algebra α) (rows N wc bc pid : Nat) (bias : Bool) (u : State α) :=
  let w := commit "dw" (fun i : TileIndex [128] => pid*128+i.1.val)
    (fun i => Cell.mk .real (KernelReference.cast M (carry M N pid u rows i).1))
    (fun i => pid*128+i.1.val<wc) (TileShape.allIndices [128]) u.mem
  if bias then commit "db" (fun i : TileIndex [128] => pid*128+i.1.val)
    (fun i => Cell.mk .real (KernelReference.cast M (carry M N pid u rows i).2))
    (fun i => pid*128+i.1.val<bc) (TileShape.allIndices [128]) w else w

theorem tail_executes (M : Algebra α) (rows N wc bc pid : Nat) (bias : Bool) (initial u : State α)
    (inv : Holds M rows N pid initial rows u) :
    ∃ v, run (engine M) (tail rows N wc bc bias) u=some v ∧
      v.mem=heap M rows N wc bc pid bias initial := by
  cases bias
  all_goals
    simp [tail,kernel,affineDirectOffsets,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,natLt,numericLt,numericLe,numericEq,bop,Region.cast,State.setReg,FP.Structural.store,
      inv.col,inv.mask,inv.dw,inv.db,inv.mem,heap,columns,mask,KernelReference.cast,fold_mem,
      Broadcast.leftIndex,Broadcast.rightIndex,ofFloat,toFloat]
    <;> rfl

theorem source_memory (M : Algebra α) (rows N wc bc : Nat) (bias : Bool) (u : State α) :
    ∃ v, FP.Structural.exec (engine M) (kernel rows N wc bc bias) u=some v ∧
      v.mem=heap M rows N wc bc (u.pids 0) bias u := by
  obtain ⟨v,he,inv⟩ := loop_executes M rows N wc bc (u.pids 0) bias u 0 (start M N u)
    (start_holds M rows N u)
  obtain ⟨w,hw,hmem⟩ := tail_executes M rows N wc bc (u.pids 0) bias u v inv
  refine ⟨w,?_,hmem⟩
  change run (engine M) (kernel rows N wc bc bias).surfaceBody u=some w
  rw [source_body,run_append,run_append,prefix_executes]
  simpa [run,step,FP.Structural.evalOp,he] using hw

#axiomsClean source_memory
end
end VeriTile.Bench.Repairs.FastLayernorm.AffineDirect
