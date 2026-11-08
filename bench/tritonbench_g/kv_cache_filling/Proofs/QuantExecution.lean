import Mathlib.Tactic.NormNum
import Mathlib.Tactic.SplitIfs
import bench.tritonbench_g.kv_cache_filling.Proofs.QuantProfile
import bench.tritonbench_g.kv_cache_filling.Proofs.CopyExecution

namespace VeriTile.Bench.Repairs.KvCacheFilling.QuantExecution
open VeriTile.Triton FP.Structural QuantProfile CopySource CopyExecution
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def inputRow {α : Type} [Inhabited α] (p : Parameters)
    (s : State α) (src : Nat) (key : Bool) (i : Nat) : α :=
  (s.mem (input key) (src * width p key + i)).read .real

def metaRegion (key : Bool) : RegionName := if key then "keyMeta" else "valueMeta"

def preparedInputs {α : Type} (p : Parameters) (s : State α) (src dst : Nat) : State α :=
  let z := s.setReg "job" .int [] (fun _ => s.pids 0)
  let z := z.setReg "src" .int [] (fun _ => src)
  let z := z.setReg "dst" .int [] (fun _ => dst)
  let z := z.setReg "ik" .int [p.bk] (fun i => i.1.val)
  let z := z.setReg "iv" .int [p.bv] (fun i => i.1.val)
  let z := z.setReg "safe_ik" .int [p.bk] (fun i => safe p.ck i.1.val)
  z.setReg "safe_iv" .int [p.bv] (fun i => safe p.cv i.1.val)

def row8Data {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src : Nat) (key : Bool) : Row α :=
  row8 a (capacity p key) (lanes p key) (inputRow p s src key)

def scatter {α : Type} (p : Parameters) (dst : Nat) (key : Bool)
    (values : Nat → BitVec 8) (s : State α) : State α :=
  (TileShape.allIndices [lanes p key]).foldl (fun z i =>
    if i.1.val < capacity p key then
      z.write (output key) (dst * capacity p key + i.1.val) (.mk .nat (values i.1.val).toNat)
    else z) s

def writeMetadata {α : Type} (dst : Nat) (k v : Row α) (s : State α) : State α :=
  let z := s.write "keyMeta" (dst*2) (.mk .real k.scale)
  let z := z.write "keyMeta" (dst*2+1) (.mk .real k.zero)
  let z := z.write "valueMeta" (dst*2) (.mk .real v.scale)
  z.write "valueMeta" (dst*2+1) (.mk .real v.zero)

def scalarLoaded8 {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src dst : Nat) : State α :=
  let k := inputRow p s src Bool.true
  let v := inputRow p s src Bool.false
  let kp := pad p.ck p.bk k
  let vp := pad p.cv p.bv v
  let z := preparedInputs p s src dst
  let z := z.setReg "k0" .real [] (fun _ => k 0)
  let z := z.setReg "v0" .real [] (fun _ => v 0)
  z

def raw8 {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src dst : Nat) : State α :=
  let k := inputRow p s src Bool.true
  let v := inputRow p s src Bool.false
  let z := scalarLoaded8 a p s src dst
  let z := z.setReg "k" .real [p.bk] (fun i =>
    if i.1.val < p.dk then k (safe p.ck i.1.val) else a.literal 0)
  let z := z.setReg "v" .real [p.bv] (fun i =>
    if i.1.val < p.dv then v (safe p.cv i.1.val) else a.literal 0)
  z

def loaded8 {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src dst : Nat) : State α :=
  let kp := pad p.ck p.bk (inputRow p s src Bool.true)
  let vp := pad p.cv p.bv (inputRow p s src Bool.false)
  let z := raw8 a p s src dst
  let z := z.setReg "k" .real [p.bk] (fun i => kp i.1)
  let z := z.setReg "v" .real [p.bv] (fun i => vp i.1)
  z

/-- Pure mathematical register state for one expanded helper invocation.
It contains no execution callback, output assumption or numerical rule. -/
def helperReg (n : Nat) : RegName :=
  match n with
  | 0 => "__veritile_statement_jit_0"
  | 1 => "__veritile_statement_jit_1"
  | 2 => "__veritile_statement_jit_2"
  | 3 => "__veritile_statement_jit_3"
  | 4 => "__veritile_statement_jit_4"
  | 5 => "__veritile_statement_jit_5"
  | 6 => "__veritile_statement_jit_6"
  | 7 => "__veritile_statement_jit_7"
  | 8 => "__veritile_statement_jit_8"
  | 9 => "__veritile_statement_jit_9"
  | 10 => "__veritile_statement_jit_10"
  | 11 => "__veritile_statement_jit_11"
  | 12 => "__veritile_statement_jit_12"
  | 13 => "__veritile_statement_jit_13"
  | 14 => "__veritile_statement_jit_14"
  | 15 => "__veritile_statement_jit_15"
  | 16 => "__veritile_statement_jit_16"
  | 17 => "__veritile_statement_jit_17"
  | 18 => "__veritile_statement_jit_18"
  | 19 => "__veritile_statement_jit_19"
  | 20 => "__veritile_statement_jit_20"
  | 21 => "__veritile_statement_jit_21"
  | 22 => "__veritile_statement_jit_22"
  | 23 => "__veritile_statement_jit_23"
  | 24 => "__veritile_statement_jit_24"
  | 25 => "__veritile_statement_jit_25"
  | 26 => "__veritile_statement_jit_26"
  | 27 => "__veritile_statement_jit_27"
  | 28 => "__veritile_statement_jit_28"
  | 29 => "__veritile_statement_jit_29"
  | 30 => "__veritile_statement_jit_30"
  | 31 => "__veritile_statement_jit_31"
  | 32 => "__veritile_statement_jit_32"
  | 33 => "__veritile_statement_jit_33"
  | 34 => "__veritile_statement_jit_34"
  | 35 => "__veritile_statement_jit_35"
  | 36 => "__veritile_statement_jit_36"
  | 37 => "__veritile_statement_jit_37"
  | 38 => "__veritile_statement_jit_38"
  | 39 => "__veritile_statement_jit_39"
  | n => "__veritile_statement_jit_" ++ toString n

def quantState {α : Type} [Inhabited α] (a : Ops α) (n base levels : Nat)
    (x : Fin n → α) (lo hi : α) (q scaleName zeroName : RegName) (s : State α) : State α :=
  let sc := scale a levels lo hi
  let ze := zero a levels lo hi
  let codes := fun (i : TileIndex [n]) => (code a levels lo hi (x i.1)).toNat
  let z := s.setReg (helperReg base) .real [n] (fun i => x i.1)
  let z := z.setReg (helperReg (base+1)) .real [] (fun _ => lo)
  let z := z.setReg (helperReg (base+2)) .real [] (fun _ => hi)
  let z := z.setReg (helperReg (base+3)) .real [] (fun _ => a.divide (a.sub hi lo) (a.literal levels))
  let z := z.setReg (helperReg (base+4)) .real [] (fun _ => sc)
  let z := z.setReg (helperReg (base+5)) .real [] (fun _ => ze)
  let z := z.setReg (helperReg (base+6)) .nat [n] codes
  let z := z.setReg (helperReg (base+7)) .nat [n] codes
  let z := z.setReg (helperReg (base+8)) .real [] (fun _ => sc)
  let z := z.setReg (helperReg (base+9)) .real [] (fun _ => ze)
  let z := z.setReg q .nat [n] codes
  let z := z.setReg scaleName .real [] (fun _ => sc)
  z.setReg zeroName .real [] (fun _ => ze)

def key8 {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src dst : Nat) : State α :=
  let kp := pad p.ck p.bk (inputRow p s src Bool.true)
  quantState a p.bk 0 255 kp (a.low p.bk kp) (a.high p.bk kp)
    "qk" "sk" "zk" (loaded8 a p s src dst)

def prepared8 {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src dst : Nat) : State α :=
  let vp := pad p.cv p.bv (inputRow p s src Bool.false)
  quantState a p.bv 10 255 vp (a.low p.bv vp) (a.high p.bv vp)
    "qv" "sv" "zv" (key8 a p s src dst)

def result8 {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src dst : Nat) : State α :=
  let k := row8Data a p s src Bool.true
  let v := row8Data a p s src Bool.false
  writeMetadata dst k v (scatter p dst Bool.false v.values
    (scatter p dst Bool.true k.values (prepared8 a p s src dst)))

@[simp] theorem masked_regs {α ι : Type} (reg : RegionName) (off : ι → Nat)
    (val : ι → FP.Structural.Cell α) (active : ι → Prop) [DecidablePred active]
    (l : List ι) (s : State α) :
    (l.foldl (fun t i => if active i then t.write reg (off i) (val i) else t) s).regs = s.regs := by
  rw [State.masked_scatter_eq_filter, State.scatter_regs]

theorem inputs_execute {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src dst : Nat)
    (hs : (s.mem "src" (s.pids 0)).read .int = (src : Int))
    (hd : (s.mem "dst" (s.pids 0)).read .int = (dst : Int))
    (sk : p.dk = p.ck) (sv : p.dv = p.cv) (version : Bool) :
    run (algebra a) ((QuantSource.kernel8 version p).surfaceBody.take 7) s =
      some (preparedInputs p s src dst) := by
  cases version
  all_goals
    dsimp only [QuantSource.kernel8, QuantSource.original8Kernel,original8Kernel,
      QuantSource.commuted8Kernel,candidate8Kernel, ComputeKernel.surfaceBody, List.take]
  all_goals
    simp (maxSteps := 200000) [QuantSource.kernel8, QuantSource.original8Kernel,original8Kernel,
      QuantSource.commuted8Kernel,candidate8Kernel, loaded8, preparedInputs, pad, inputRow,
      hs, hd, sk, sv, Int.mul_comm, Nat.mul_comm,
      ComputeKernel.surfaceBody, run, step, evalExpr,
      evalComputeOp, evalOp_unfold, store, numeric, numericLt, numericEq, bop,
      ComputeDType.eraseDType, Region.cast, Option.bind, algebra,
      ofFloat, toFloat, TileShape.axisDim,
      CopyExecution.lanes, CopyExecution.width, CopyExecution.capacity,
      CopyExecution.input, CopyExecution.output, CopyExecution.safe,
      Broadcast.leftIndex, Broadcast.rightIndex, index_toNat,
      TileShape.insertAxisIndex, Int.toNat_mul, Int.toNat_add]

theorem scalarLoads8_execute {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src dst : Nat)
    (sk : p.dk = p.ck) (sv : p.dv = p.cv) (version : Bool) :
    run (algebra a) (((QuantSource.kernel8 version p).surfaceBody.drop 7).take 2)
      (preparedInputs p s src dst) = some (scalarLoaded8 a p s src dst) := by
  cases version
  all_goals
    dsimp only [QuantSource.kernel8, QuantSource.original8Kernel,original8Kernel,
      QuantSource.commuted8Kernel,candidate8Kernel, ComputeKernel.surfaceBody, List.drop, List.take]
  all_goals
    simp (maxSteps := 200000) [scalarLoaded8, preparedInputs, pad, inputRow, sk, sv, Int.mul_comm, Nat.mul_comm,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, numeric, numericLt, bop,
      ComputeDType.eraseDType, Region.cast, Option.bind, algebra,
      ofFloat, toFloat, lanes, width, capacity, input, output, safe,
      Broadcast.leftIndex, Broadcast.rightIndex, index_toNat,
      Int.toNat_mul, Int.toNat_add]

@[simp] theorem padded_load {α : Type} (x : Nat → α) (w i : Nat) (z : α) :
    (if i < w then (if i < w then x (safe w i) else z) else x 0) =
      x (if i < w then i else 0) := by
  by_cases h : i < w <;> simp [h,safe]

theorem load8_execute {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src dst : Nat)
    (hs : (s.mem "src" (s.pids 0)).read .int = (src : Int))
    (hd : (s.mem "dst" (s.pids 0)).read .int = (dst : Int))
    (sk : p.dk = p.ck) (sv : p.dv = p.cv) (version : Bool) :
    run (algebra a) ((QuantSource.kernel8 version p).surfaceBody.take 13) s =
      some (loaded8 a p s src dst) := by
  cases version
  all_goals
    dsimp only [QuantSource.kernel8, QuantSource.original8Kernel,original8Kernel,
      QuantSource.commuted8Kernel,candidate8Kernel, ComputeKernel.surfaceBody, List.take]
  all_goals
    simp (maxSteps := 200000) [loaded8,raw8,scalarLoaded8,preparedInputs,pad,inputRow,
      hs,hd,sk,sv,Int.mul_comm,Nat.mul_comm,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,numeric,numericLt,bop,
      ComputeDType.eraseDType,Region.cast,Option.bind,algebra,
      ofFloat,toFloat,lanes,width,capacity,input,output,safe,
      Broadcast.leftIndex,Broadcast.rightIndex,index_toNat,Int.toNat_mul,Int.toNat_add]
  all_goals congr 8
  all_goals funext i
  all_goals split_ifs <;> simp_all <;> norm_num

theorem key8_execute {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src dst : Nat) (hb : 0 < p.bk) (version : Bool) :
    run (algebra a) (((QuantSource.kernel8 version p).surfaceBody.drop 13).take 13)
      (loaded8 a p s src dst) = some (key8 a p s src dst) := by
  cases version
  all_goals
    dsimp only [QuantSource.kernel8,QuantSource.original8Kernel,original8Kernel,
      QuantSource.commuted8Kernel,candidate8Kernel,ComputeKernel.surfaceBody,List.drop,List.take]
  all_goals
    simp [quantState,helperReg,key8,row8Data,row8,loaded8,raw8,scalarLoaded8,preparedInputs,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,numeric,numericEq,bop,
      ComputeDType.eraseDType,Option.bind,algebra,reduceAxis,scale,zero,code,
      ofFloat,toFloat,lanes,capacity,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.axisDim,TileShape.insertAxisIndex,TileShape.reduceShape,TileShape.eraseAxis,
      hb,Nat.ne_of_gt hb]
  all_goals norm_num
  all_goals rfl
theorem value8_execute {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src dst : Nat) (hb : 0 < p.bv) (version : Bool) :
    run (algebra a) (((QuantSource.kernel8 version p).surfaceBody.drop 26).take 13)
      (key8 a p s src dst) = some (prepared8 a p s src dst) := by
  cases version
  all_goals
    dsimp only [QuantSource.kernel8,QuantSource.original8Kernel,original8Kernel,
      QuantSource.commuted8Kernel,candidate8Kernel,ComputeKernel.surfaceBody,List.drop,List.take]
  all_goals
    simp [quantState,helperReg,prepared8,key8,row8Data,row8,loaded8,raw8,scalarLoaded8,preparedInputs,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,numeric,numericEq,bop,
      ComputeDType.eraseDType,Option.bind,algebra,reduceAxis,scale,zero,code,
      ofFloat,toFloat,lanes,capacity,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.axisDim,TileShape.insertAxisIndex,TileShape.reduceShape,TileShape.eraseAxis,
      hb,Nat.ne_of_gt hb]
  all_goals norm_num
  all_goals rfl

theorem mul_index_toNat (a b : Nat) : Int.toNat ((a : Int)*b) = a*b := by
  simp only [← Int.natCast_mul,Int.toNat_natCast]

theorem add_index_toNat (a b i : Nat) : Int.toNat ((a : Int)*b+i) = a*b+i := by
  simp only [← Int.natCast_mul,← Int.natCast_add,Int.toNat_natCast]

theorem meta0_index_toNat (a : Nat) : Int.toNat ((a : Int)*2) = a*2 := by omega
theorem meta1_index_toNat (a : Nat) : Int.toNat ((a : Int)*2+1) = a*2+1 := by omega

theorem metadata_index_toNat (a i : Nat) : Int.toNat ((a : Int)*2+i) = a*2+i := by omega

theorem half_active_index_toNat (a b c i : Nat) :
    Int.toNat ((a : Int)*b+c+i) = a*b+(c+i) := by
  simp only [← Int.natCast_mul,← Int.natCast_add,Int.toNat_natCast,Nat.add_assoc]

theorem half_index_toNat (a b c w i : Nat) :
    Int.toNat ((a : Int)*b+c+(if i < w then (i : Int) else 0)) =
      a*b+(c+if i < w then i else 0) := by
  by_cases h : i < w <;> simp [h,half_active_index_toNat,add_index_toNat]

@[simp] theorem padded_scatter {α : Type} (reg : RegionName) (w n dst : Nat)
    (x : Nat → α) (f : α → Nat) (s : State α) :
    (TileShape.allIndices [n]).foldl (fun t i =>
      if i.1.val < w then t.write reg (dst*w+safe w i.1.val)
        (.mk .nat (f (pad w n x i.1))) else t) s =
    (TileShape.allIndices [n]).foldl (fun t i =>
      if i.1.val < w then t.write reg (dst*w+i.1.val)
        (.mk .nat (f (x i.1.val))) else t) s := by
  congr 1
  funext t i
  by_cases h : i.1.val < w <;> simp [safe,pad,h]

@[simp] theorem byte_mod (x : BitVec 8) : x.toNat % 256 = x.toNat :=
  Nat.mod_eq_of_lt x.isLt

theorem stores8_execute {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src dst : Nat) (version : Bool) :
    run (algebra a) ((QuantSource.kernel8 version p).surfaceBody.drop 39)
      (prepared8 a p s src dst) = some (result8 a p s src dst) := by
  cases version
  all_goals
    dsimp only [QuantSource.kernel8,QuantSource.original8Kernel,original8Kernel,
      QuantSource.commuted8Kernel,candidate8Kernel,ComputeKernel.surfaceBody,List.drop]
  all_goals
    simp [quantState,helperReg,result8,writeMetadata,QuantExecution.scatter,prepared8,key8,row8Data,row8,
      loaded8,raw8,scalarLoaded8,preparedInputs,
      run,step,store,evalExpr,evalComputeOp,evalOp_unfold,numeric,numericLt,bop,
      ComputeDType.eraseDType,Region.cast,Option.bind,algebra,
      ofFloat,toFloat,lanes,capacity,output,Broadcast.leftIndex,Broadcast.rightIndex,
      index_toNat,mul_index_toNat,add_index_toNat,
      show TileShape.allIndices [] = [PUnit.unit] from rfl,List.foldl_cons,List.foldl_nil]
  all_goals simp +contextual [safe,pad,meta0_index_toNat,meta1_index_toNat,integralMod]
  all_goals rfl

/-- Complete successful INT8 execution, including both caches and metadata. -/
theorem execute8 {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src dst : Nat)
    (hs : (s.mem "src" (s.pids 0)).read .int = (src : Int))
    (hd : (s.mem "dst" (s.pids 0)).read .int = (dst : Int))
    (hk : 0 < p.bk) (hv : 0 < p.bv) (sk : p.dk = p.ck) (sv : p.dv = p.cv) (version : Bool) :
    exec (algebra a) (QuantSource.kernel8 version p) s = some (result8 a p s src dst) := by
  have decomp : (QuantSource.kernel8 version p).surfaceBody =
      (QuantSource.kernel8 version p).surfaceBody.take 13 ++
      (((QuantSource.kernel8 version p).surfaceBody.drop 13).take 13 ++
      (((QuantSource.kernel8 version p).surfaceBody.drop 26).take 13 ++
      (QuantSource.kernel8 version p).surfaceBody.drop 39)) := by cases version <;> rfl
  change run (algebra a) (QuantSource.kernel8 version p).surfaceBody s = _
  rw [decomp,run_append,load8_execute a p s src dst hs hd sk sv version]
  simp only [Option.bind_some]
  rw [run_append,key8_execute a p s src dst hk version]
  simp only [Option.bind_some]
  rw [run_append,value8_execute a p s src dst hv version]
  simp only [Option.bind_some]
  exact stores8_execute a p s src dst version

def executed8 {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src dst : Nat)
    (hs : (s.mem "src" (s.pids 0)).read .int = (src : Int))
    (hd : (s.mem "dst" (s.pids 0)).read .int = (dst : Int))
    (hk : 0 < p.bk) (hv : 0 < p.bv) (sk : p.dk = p.ck) (sv : p.dv = p.cv) (version : Bool) :
    { f : State α // exec (algebra a) (QuantSource.kernel8 version p) s = some f } :=
  ⟨result8 a p s src dst,execute8 a p s src dst hs hd hk hv sk sv version⟩

#axiomsClean execute8

#axiomsClean value8_execute
#axiomsClean stores8_execute

#axiomsClean key8_execute

#axiomsClean load8_execute

#axiomsClean scalarLoads8_execute

#axiomsClean inputs_execute
end
end VeriTile.Bench.Repairs.KvCacheFilling.QuantExecution
