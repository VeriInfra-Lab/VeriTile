import bench.tritonbench_g.kv_cache_filling.Proofs.QuantExecution

namespace VeriTile.Bench.Repairs.KvCacheFilling.QuantExecution
open VeriTile.Triton FP.Structural QuantProfile CopySource CopyExecution
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 8000
set_option linter.unusedSimpArgs false

def row4Data {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src : Nat) (key : Bool) : Row α :=
  row4 a (capacity p key) (lanes p key) (inputRow p s src key)

def loaded4 {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src dst : Nat) : State α :=
  let k := inputRow p s src Bool.true
  let v := inputRow p s src Bool.false
  let k1 := pad p.ck p.bk k
  let k2 := pad p.ck p.bk (fun i => k (p.ck+i))
  let v1 := pad p.cv p.bv v
  let v2 := pad p.cv p.bv (fun i => v (p.cv+i))
  let z := preparedInputs p s src dst
  let z := z.setReg "k10" .real [] (fun _ => k 0)
  let z := z.setReg "k20" .real [] (fun _ => k p.ck)
  let z := z.setReg "v10" .real [] (fun _ => v 0)
  let z := z.setReg "v20" .real [] (fun _ => v p.cv)
  let z := z.setReg "__veritile_load_0" .real [p.bk] (fun i =>
    if i.1.val < p.ck then k (safe p.ck i.1.val) else a.literal 0)
  let z := z.setReg "k1" .real [p.bk] (fun i =>
    if i.1.val < p.ck then k (safe p.ck i.1.val) else a.literal 0)
  let z := z.setReg "__veritile_load_1" .real [p.bk] (fun i =>
    if i.1.val < p.ck then k (p.ck+safe p.ck i.1.val) else a.literal 0)
  let z := z.setReg "k2" .real [p.bk] (fun i =>
    if i.1.val < p.ck then k (p.ck+safe p.ck i.1.val) else a.literal 0)
  let z := z.setReg "__veritile_load_2" .real [p.bv] (fun i =>
    if i.1.val < p.cv then v (safe p.cv i.1.val) else a.literal 0)
  let z := z.setReg "v1" .real [p.bv] (fun i =>
    if i.1.val < p.cv then v (safe p.cv i.1.val) else a.literal 0)
  let z := z.setReg "__veritile_load_3" .real [p.bv] (fun i =>
    if i.1.val < p.cv then v (p.cv+safe p.cv i.1.val) else a.literal 0)
  let z := z.setReg "v2" .real [p.bv] (fun i =>
    if i.1.val < p.cv then v (p.cv+safe p.cv i.1.val) else a.literal 0)
  let z := z.setReg "k1" .real [p.bk] (fun i => k1 i.1)
  let z := z.setReg "k2" .real [p.bk] (fun i => k2 i.1)
  let z := z.setReg "v1" .real [p.bv] (fun i => v1 i.1)
  let z := z.setReg "v2" .real [p.bv] (fun i => v2 i.1)
  z

def reduced4 {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src dst : Nat) : State α :=
  let k := inputRow p s src Bool.true
  let v := inputRow p s src Bool.false
  let k1 := pad p.ck p.bk k
  let k2 := pad p.ck p.bk (fun i => k (p.ck+i))
  let v1 := pad p.cv p.bv v
  let v2 := pad p.cv p.bv (fun i => v (p.cv+i))
  let z := loaded4 a p s src dst
  let z := z.setReg "klo" .real [] (fun _ => a.low p.bk (fun i => a.minimum (k1 i) (k2 i)))
  let z := z.setReg "khi" .real [] (fun _ => a.high p.bk (fun i => a.maximum (k1 i) (k2 i)))
  let z := z.setReg "vlo" .real [] (fun _ => a.low p.bv (fun i => a.minimum (v1 i) (v2 i)))
  let z := z.setReg "vhi" .real [] (fun _ => a.high p.bv (fun i => a.maximum (v1 i) (v2 i)))
  z

def key4 {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src dst : Nat) : State α :=
  let x := inputRow p s src Bool.true
  let k1 := pad p.ck p.bk x
  let k2 := pad p.ck p.bk (fun i => x (p.ck+i))
  let lo := a.low p.bk (fun i => a.minimum (k1 i) (k2 i))
  let hi := a.high p.bk (fun i => a.maximum (k1 i) (k2 i))
  let z := quantState a p.bk 0 15 k1 lo hi "qk1" "sk" "zk" (reduced4 a p s src dst)
  quantState a p.bk 10 15 k2 lo hi "qk2" "_" "_" z

def value4 {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src dst : Nat) : State α :=
  let x := inputRow p s src Bool.false
  let v1 := pad p.cv p.bv x
  let v2 := pad p.cv p.bv (fun i => x (p.cv+i))
  let lo := a.low p.bv (fun i => a.minimum (v1 i) (v2 i))
  let hi := a.high p.bv (fun i => a.maximum (v1 i) (v2 i))
  let z := quantState a p.bv 20 15 v1 lo hi "qv1" "sv" "zv" (key4 a p s src dst)
  quantState a p.bv 30 15 v2 lo hi "qv2" "_" "_" z


/-- Unwrapped integer packing before the source's implicit uint8 store. -/
def packedNatural {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src : Nat) (key : Bool) (i : Fin (lanes p key)) : Nat :=
  let x := inputRow p s src key
  let left := pad (capacity p key) (lanes p key) x
  let right := pad (capacity p key) (lanes p key) (fun j => x (capacity p key+j))
  let lo := a.low (lanes p key) (fun j => a.minimum (left j) (right j))
  let hi := a.high (lanes p key) (fun j => a.maximum (left j) (right j))
  (code a 15 lo hi (left i)).toNat + (code a 15 lo hi (right i)).toNat * 16

def prepared4 {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src dst : Nat) : State α :=
  let z := value4 a p s src dst
  let z := z.setReg "qk" .nat [p.bk] (fun i => packedNatural a p s src Bool.true i.1)
  z.setReg "qv" .nat [p.bv] (fun i => packedNatural a p s src Bool.false i.1)

def result4 {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src dst : Nat) : State α :=
  let k := row4Data a p s src Bool.true
  let v := row4Data a p s src Bool.false
  writeMetadata dst k v (QuantExecution.scatter p dst Bool.false v.values
    (QuantExecution.scatter p dst Bool.true k.values (prepared4 a p s src dst)))

theorem load4_execute {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src dst : Nat)
    (hs : (s.mem "src" (s.pids 0)).read .int = (src : Int))
    (hd : (s.mem "dst" (s.pids 0)).read .int = (dst : Int)) (version : Bool) :
    run (algebra a) ((QuantSource.kernel4 version p).surfaceBody.take 23) s =
      some (loaded4 a p s src dst) := by
  cases version
  all_goals
    dsimp only [QuantSource.kernel4,QuantSource.original4Kernel,original4Kernel,
      QuantSource.commuted4Kernel,candidate4Kernel,ComputeKernel.surfaceBody,List.take]
  all_goals
    simp [loaded4,preparedInputs,inputRow,pad,safe,hs,hd,Int.mul_comm,Nat.mul_comm,
      index_toNat,mul_index_toNat,add_index_toNat,Int.toNat_mul,Int.toNat_add,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,numeric,numericLt,numericEq,bop,
      ComputeDType.eraseDType,Region.cast,Option.bind,algebra,reduceAxis,
      ofFloat,toFloat,lanes,width,capacity,input,output,
      Broadcast.leftIndex,Broadcast.rightIndex,TileShape.axisDim,
      TileShape.insertAxisIndex,TileShape.reduceShape,TileShape.eraseAxis]
  all_goals congr 20
  all_goals funext i
  all_goals split_ifs <;> (try simp_all [Nat.add_assoc,half_active_index_toNat]) <;> norm_num
  all_goals rfl

#axiomsClean load4_execute

theorem reduce4_execute {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src dst : Nat) (hk : 0 < p.bk) (hv : 0 < p.bv) (version : Bool) :
    run (algebra a) (((QuantSource.kernel4 version p).surfaceBody.drop 23).take 4)
      (loaded4 a p s src dst) = some (reduced4 a p s src dst) := by
  cases version
  all_goals
    dsimp only [QuantSource.kernel4,QuantSource.original4Kernel,original4Kernel,
      QuantSource.commuted4Kernel,candidate4Kernel,ComputeKernel.surfaceBody,List.drop,List.take]
  all_goals
    simp [reduced4,loaded4,preparedInputs,scale,zero,code,hk,hv,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,numeric,numericLt,numericEq,bop,
      ComputeDType.eraseDType,Region.cast,Option.bind,algebra,reduceAxis,
      ofFloat,toFloat,lanes,width,capacity,input,output,
      Broadcast.leftIndex,Broadcast.rightIndex,TileShape.axisDim,
      TileShape.insertAxisIndex,TileShape.reduceShape,TileShape.eraseAxis,integralMod,BitVec.toNat_add,BitVec.toNat_mul]
  all_goals norm_num
  all_goals rfl

#axiomsClean reduce4_execute

theorem key4_execute {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src dst : Nat) (hk : 0 < p.bk) (hv : 0 < p.bv) (version : Bool) :
    run (algebra a) (((QuantSource.kernel4 version p).surfaceBody.drop 27).take 26)
      (reduced4 a p s src dst) = some (key4 a p s src dst) := by
  cases version
  all_goals
    dsimp only [QuantSource.kernel4,QuantSource.original4Kernel,original4Kernel,
      QuantSource.commuted4Kernel,candidate4Kernel,ComputeKernel.surfaceBody,List.drop,List.take]
  all_goals
    simp [quantState,helperReg,key4,reduced4,loaded4,preparedInputs,scale,zero,code,hk,hv,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,numeric,numericLt,numericEq,bop,
      ComputeDType.eraseDType,Region.cast,Option.bind,algebra,reduceAxis,
      ofFloat,toFloat,lanes,width,capacity,input,output,
      Broadcast.leftIndex,Broadcast.rightIndex,TileShape.axisDim,
      TileShape.insertAxisIndex,TileShape.reduceShape,TileShape.eraseAxis,integralMod,BitVec.toNat_add,BitVec.toNat_mul]
  all_goals norm_num
  all_goals rfl

#axiomsClean key4_execute

theorem value4_execute {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src dst : Nat) (hk : 0 < p.bk) (hv : 0 < p.bv) (version : Bool) :
    run (algebra a) (((QuantSource.kernel4 version p).surfaceBody.drop 53).take 26)
      (key4 a p s src dst) = some (value4 a p s src dst) := by
  cases version
  all_goals
    dsimp only [QuantSource.kernel4,QuantSource.original4Kernel,original4Kernel,
      QuantSource.commuted4Kernel,candidate4Kernel,ComputeKernel.surfaceBody,List.drop,List.take]
  all_goals
    simp [quantState,helperReg,value4,key4,reduced4,loaded4,preparedInputs,scale,zero,code,hk,hv,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,numeric,numericLt,numericEq,bop,
      ComputeDType.eraseDType,Region.cast,Option.bind,algebra,reduceAxis,
      ofFloat,toFloat,lanes,width,capacity,input,output,
      Broadcast.leftIndex,Broadcast.rightIndex,TileShape.axisDim,
      TileShape.insertAxisIndex,TileShape.reduceShape,TileShape.eraseAxis,integralMod,BitVec.toNat_add,BitVec.toNat_mul]
  all_goals norm_num
  all_goals rfl

#axiomsClean value4_execute

theorem pack4_execute {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src dst : Nat) (hk : 0 < p.bk) (hv : 0 < p.bv) (version : Bool) :
    run (algebra a) (((QuantSource.kernel4 version p).surfaceBody.drop 79).take 2)
      (value4 a p s src dst) = some (prepared4 a p s src dst) := by
  cases version
  all_goals
    dsimp only [QuantSource.kernel4,QuantSource.original4Kernel,original4Kernel,
      QuantSource.commuted4Kernel,candidate4Kernel,ComputeKernel.surfaceBody,List.drop,List.take]
  all_goals
    simp [quantState,helperReg,packedNatural,prepared4,value4,key4,reduced4,loaded4,preparedInputs,row4Data,row4,pad,safe,scale,zero,code,hk,hv,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,numeric,numericLt,numericEq,bop,
      ComputeDType.eraseDType,Region.cast,Option.bind,algebra,reduceAxis,
      ofFloat,toFloat,lanes,width,capacity,input,output,
      Broadcast.leftIndex,Broadcast.rightIndex,TileShape.axisDim,
      TileShape.insertAxisIndex,TileShape.reduceShape,TileShape.eraseAxis,integralMod,BitVec.toNat_add,BitVec.toNat_mul]
  all_goals norm_num
  all_goals rfl

#axiomsClean pack4_execute
theorem stores4_execute {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src dst : Nat) (version : Bool) :
    run (algebra a) ((QuantSource.kernel4 version p).surfaceBody.drop 81)
      (prepared4 a p s src dst) = some (result4 a p s src dst) := by
  cases version
  all_goals
    dsimp only [QuantSource.kernel4,QuantSource.original4Kernel,original4Kernel,
      QuantSource.commuted4Kernel,candidate4Kernel,ComputeKernel.surfaceBody,List.drop]
  all_goals
    simp [quantState,helperReg,packedNatural,result4,writeMetadata,QuantExecution.scatter,prepared4,key4,row4Data,row4,
      value4,reduced4,loaded4,preparedInputs,
      run,step,store,evalExpr,evalComputeOp,evalOp_unfold,numeric,numericLt,bop,
      ComputeDType.eraseDType,Region.cast,Option.bind,algebra,
      ofFloat,toFloat,lanes,capacity,output,Broadcast.leftIndex,Broadcast.rightIndex,
      index_toNat,mul_index_toNat,add_index_toNat,
      show TileShape.allIndices [] = [PUnit.unit] from rfl,List.foldl_cons,List.foldl_nil]
  all_goals simp +contextual [safe,pad,meta0_index_toNat,meta1_index_toNat]
  all_goals rfl

/-- Complete successful INT4 execution, including both caches and metadata. -/
theorem execute4 {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src dst : Nat)
    (hs : (s.mem "src" (s.pids 0)).read .int = (src : Int))
    (hd : (s.mem "dst" (s.pids 0)).read .int = (dst : Int))
    (hk : 0 < p.bk) (hv : 0 < p.bv) (version : Bool) :
    exec (algebra a) (QuantSource.kernel4 version p) s = some (result4 a p s src dst) := by
  have decomp : (QuantSource.kernel4 version p).surfaceBody =
      (QuantSource.kernel4 version p).surfaceBody.take 23 ++
      (((QuantSource.kernel4 version p).surfaceBody.drop 23).take 4 ++
      (((QuantSource.kernel4 version p).surfaceBody.drop 27).take 26 ++
      (((QuantSource.kernel4 version p).surfaceBody.drop 53).take 26 ++
      (((QuantSource.kernel4 version p).surfaceBody.drop 79).take 2 ++
      (QuantSource.kernel4 version p).surfaceBody.drop 81)))) := by cases version <;> rfl
  change run (algebra a) (QuantSource.kernel4 version p).surfaceBody s = _
  rw [decomp,run_append,load4_execute a p s src dst hs hd version]
  simp only [Option.bind_some]
  rw [run_append,reduce4_execute a p s src dst hk hv version]
  simp only [Option.bind_some]
  rw [run_append,key4_execute a p s src dst hk hv version]
  simp only [Option.bind_some]
  rw [run_append,value4_execute a p s src dst hk hv version]
  simp only [Option.bind_some]
  rw [run_append,pack4_execute a p s src dst hk hv version]
  simp only [Option.bind_some]
  exact stores4_execute a p s src dst version

def executed4 {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src dst : Nat)
    (hs : (s.mem "src" (s.pids 0)).read .int = (src : Int))
    (hd : (s.mem "dst" (s.pids 0)).read .int = (dst : Int))
    (hk : 0 < p.bk) (hv : 0 < p.bv) (version : Bool) :
    { f : State α // exec (algebra a) (QuantSource.kernel4 version p) s = some f } :=
  ⟨result4 a p s src dst,execute4 a p s src dst hs hd hk hv version⟩

#axiomsClean stores4_execute
#axiomsClean execute4

end
end VeriTile.Bench.Repairs.KvCacheFilling.QuantExecution
