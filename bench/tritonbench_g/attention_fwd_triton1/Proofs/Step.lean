import bench.tritonbench_g.attention_fwd_triton1.Proofs.Body

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.SharedStep
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001
open TermBody
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

def mainPtr (c : Shape) (head chunk : Nat) (region : RegionName) : BlockPtr :=
  ⟨region,head*c.headSize,[c.T,c.D],[32,c.D],[c.D,1],[chunk*32,0]⟩
def keyPtr (c : Shape) (head chunk : Nat) : BlockPtr :=
  ⟨"k",head*c.headSize,[c.D,c.T],[c.D,32],[1,c.D],[0,chunk*32]⟩
def historyPtr (c : Shape) (head chunk : Nat) : BlockPtr :=
  ⟨"h",head*c.historyHead,[c.chunks*c.D,c.D],[c.D,c.D],[c.D,1],[chunk*c.D,0]⟩
def pointers (c : Shape) (head chunk : Nat) (s : State (Term α)) : State (Term α) :=
  ((((s.setReg "p_q" .blockPtr [32,c.D] (fun _ => mainPtr c head chunk "q")).setReg
    "p_k" .blockPtr [c.D,32] (fun _ => keyPtr c head chunk)).setReg
    "p_v" .blockPtr [32,c.D] (fun _ => mainPtr c head chunk "v")).setReg
    "p_h" .blockPtr [c.D,c.D] (fun _ => historyPtr c head chunk)).setReg
    "p_o" .blockPtr [32,c.D] (fun _ => mainPtr c head chunk "o")

def scatter (m n : Nat) (ptr : BlockPtr) (axes : List Nat)
    (values : Matrix α m n) (s : State (Term α)) : State (Term α) :=
  (TileShape.allIndices [m,n]).foldl (fun t i =>
    if ptr.inBounds (TileShape.indexToList [m,n] i) axes then
      t.write ptr.region (ptr.address (TileShape.indexToList [m,n] i)) (.mk .real (values i))
    else t) s

@[simp] theorem scatter_regs (m n : Nat) (ptr : BlockPtr) (axes : List Nat)
    (values : Matrix α m n) (s : State (Term α)) :
    (scatter m n ptr axes values s).regs=s.regs := by
  unfold scatter
  generalize TileShape.allIndices [m,n]=indices
  induction indices generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons,ih]
    split_ifs <;> rfl

@[simp] theorem scatter_pids (m n : Nat) (ptr : BlockPtr) (axes : List Nat)
    (values : Matrix α m n) (s : State (Term α)) :
    (scatter m n ptr axes values s).pids=s.pids := by
  unfold scatter
  generalize TileShape.allIndices [m,n]=indices
  induction indices generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons,ih]
    split_ifs <;> rfl

theorem scatter_other (m n : Nat) (ptr : BlockPtr) (axes : List Nat)
    (values : Matrix α m n) (s : State (Term α)) (region : RegionName) (off : Nat)
    (different : region≠ptr.region) :
    (scatter m n ptr axes values s).mem region off=s.mem region off := by
  apply State.masked_scatter_frame
  exact Or.inl different

def saved (c : Shape) (head chunk : Nat) (storeFlag : Bool)
    (carry : Matrix α c.D c.D) (s : State (Term α)) : State (Term α) :=
  if storeFlag then scatter c.D c.D (historyPtr c head chunk) []
    (fun i => TermBody.cast (carry i)) (pointers c head chunk s)
  else pointers c head chunk s

@[simp] theorem saved_regs (c : Shape) (head chunk : Nat) (storeFlag : Bool)
    (carry : Matrix α c.D c.D) (s : State (Term α)) :
    (saved c head chunk storeFlag carry s).regs=(pointers c head chunk s).regs := by
  cases storeFlag <;> simp [saved]

theorem saved_input (c : Shape) (head chunk : Nat) (storeFlag : Bool)
    (carry : Matrix α c.D c.D) (s : State (Term α)) (region : RegionName) (off : Nat)
    (different : region≠"h") :
    (saved c head chunk storeFlag carry s).mem region off=s.mem region off := by
  cases storeFlag with
  | «false» => rfl
  | «true» => exact scatter_other _ _ _ _ _ _ region off different

theorem pointer_run [Inhabited α] (c : Shape) (head chunk : Nat) (s : State (Term α))
    (hh : s.regs .int [] "i_bh"=some (fun _ => (head : Int)))
    (hi : s.regs .nat [] "i"=some (fun _ => chunk)) :
    run algebra (Program.pointers c) s=some (pointers c head chunk s) := by
  simp [Program.pointers,Program.pointer,Program.times,Program.headTimes,run,step,evalExpr,evalOp_unfold,
    evalOffsets,numeric,bop,State.setReg,hh,hi,pointers,mainPtr,keyPtr,historyPtr]
  simp only [← Int.natCast_mul,Int.toNat_natCast]

theorem store_step [Inhabited α] (m n : Nat) (pointer value : String) (ptr : BlockPtr)
    (axes : List Nat) (values : Matrix α m n) (s : State (Term α))
    (hp : s.regs .blockPtr [m,n] pointer=some (fun _ => ptr))
    (hv : s.regs .real [m,n] value=some values) :
    step algebra (Program.store m n pointer value axes) s=
      some (scatter m n ptr axes (fun i => TermBody.cast (values i)) s) := by
  simp [Program.store,Program.comp,Program.cast,Program.ref,step,store,evalExpr,
    evalComputeOp,evalOp_unfold,hp,hv,algebra,ofFloat,toFloat,scatter]

theorem save_run [Inhabited α] (c : Shape) (head chunk : Nat) (storeFlag : Bool)
    (carry : Matrix α c.D c.D) (s : State (Term α))
    (hc : s.regs .real [c.D,c.D] "b_h"=some carry) :
    run algebra [Program.save c storeFlag] (pointers c head chunk s)=
      some (saved c head chunk storeFlag carry s) := by
  cases storeFlag with
  | «false» => simp [Program.save,run,step,evalOp_unfold,saved]
  | «true» =>
    simp only [Program.save,run,step,evalOp_unfold,Option.bind_some,↓reduceIte]
    rw [store_step c.D c.D "p_h" "b_h" (historyPtr c head chunk) [] carry _
      (by simp [pointers]) (by simp [pointers,hc])]
    rfl

theorem main_address (c : Shape) (head chunk : Nat) (region : RegionName) (i j : Nat) :
    (mainPtr c head chunk region).address [i,j]=mainAddress c head chunk i j := by
  simp [mainPtr,BlockPtr.address_2d_offsets,mainAddress,Nat.add_assoc]
theorem key_address (c : Shape) (head chunk i j : Nat) :
    (keyPtr c head chunk).address [i,j]=mainAddress c head chunk j i := by
  simp [keyPtr,BlockPtr.address_2d_offsets,mainAddress,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
theorem history_address (c : Shape) (head chunk i j : Nat) :
    (historyPtr c head chunk).address [i,j]=historyAddress c head chunk i j := by
  simp [historyPtr,BlockPtr.address_2d_offsets,historyAddress,Nat.add_assoc]
theorem main_bound (c : Shape) (head chunk : Nat) (region : RegionName) (i j : Nat) :
    (mainPtr c head chunk region).inBounds [i,j] [0]=decide (chunk*32+i<c.T) := by
  change (decide (chunk*32+i<c.T) && Bool.true)=_
  simp
theorem key_bound (c : Shape) (head chunk i j : Nat) :
    (keyPtr c head chunk).inBounds [i,j] [1]=decide (chunk*32+j<c.T) := by
  change (decide (chunk*32+j<c.T) && Bool.true)=_
  simp

@[simp] theorem main_region (c : Shape) (head chunk : Nat) (region : RegionName) :
    (mainPtr c head chunk region).region=region := rfl
@[simp] theorem key_region (c : Shape) (head chunk : Nat) :
    (keyPtr c head chunk).region="k" := rfl

def read [Inhabited α] (s : State (Term α)) (region : RegionName) (off : Nat) : Term α :=
  (s.mem region off).read .real
def rawQuery [Inhabited α] (c : Shape) (q : Nat → Term α) (head chunk : Nat) : Matrix α 32 c.D :=
  fun i => .load (if chunk*32+i.1.val<c.T then
    q (mainAddress c head chunk i.1.val i.2.1.val) else num 0)
def prepared [Inhabited α] (c : Shape) (scale : ℝ) (head chunk : Nat) (storeFlag : Bool)
    (carry : Matrix α c.D c.D) (s : State (Term α)) : State (Term α) :=
  let q := read s "q"
  let k := read s "k"
  let v := read s "v"
  ((((((saved c head chunk storeFlag carry s).setReg "b_q" .real [32,c.D]
    (rawQuery c q head chunk)).setReg "b_q" .real [32,c.D] (query c q scale head chunk)).setReg
    "b_k" .real [c.D,32] (key c k head chunk)).setReg
    "b_v" .real [32,c.D] (value c v head chunk)).setReg
    "b_s" .real [32,32] (score c q k scale head chunk)).setReg
    "b_o" .real [32,c.D] (within c q k v scale head chunk)

theorem loads_run [Inhabited α] (c : Shape) (scale : ℝ) (head chunk : Nat) (storeFlag : Bool)
    (carry : Matrix α c.D c.D) (s : State (Term α)) :
    run algebra (Program.loads c scale) (saved c head chunk storeFlag carry s)=
      some (prepared c scale head chunk storeFlag carry s) := by
  have hq := saved_input c head chunk storeFlag carry s "q"
  have hk := saved_input c head chunk storeFlag carry s "k"
  have hv := saved_input c head chunk storeFlag carry s "v"
  simp [Program.loads,Program.load,Program.put,Program.comp,Program.cast,Program.ref,
    Program.dot,Program.preciseDot,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    algebra,dotTerms,boundaryZero,ofFloat,toFloat,numeric,bop,Option.bind,
    State.setReg,saved_regs,pointers,main_bound,key_bound,main_address,key_address,
    TileShape.indexToList,hq _ (by decide),hk _ (by decide),hv _ (by decide),
    prepared,rawQuery,read,query,key,value,score,within,TermBody.dot]
  rfl

def updated [Inhabited α] (c : Shape) (scale : ℝ) (head chunk : Nat) (storeFlag ifcond : Bool)
    (carry : Matrix α c.D c.D) (s : State (Term α)) : State (Term α) :=
  let p := prepared c scale head chunk storeFlag carry s
  let d := TermBody.dot (key c (read s "k") head chunk) (value c (read s "v") head chunk)
  let out := fun i => TermBody.add (within c (read s "q") (read s "k") (read s "v") scale head chunk i)
    (TermBody.dot (query c (read s "q") scale head chunk) (fun j => TermBody.cast (carry j)) i)
  let later := (p.setReg "b_o" .real [32,c.D] out).setReg "b_h" .real [c.D,c.D]
    (fun i => TermBody.add (carry i) (d i))
  if ifcond then
    if chunk=0 then p.setReg "b_h" .real [c.D,c.D] d else later
  else later

theorem update_step [Inhabited α] (c : Shape) (scale : ℝ) (head chunk : Nat) (storeFlag ifcond : Bool)
    (carry : Matrix α c.D c.D) (s : State (Term α))
    (hc : s.regs .real [c.D,c.D] "b_h"=some carry)
    (hi : s.regs .nat [] "i"=some (fun _ => chunk)) :
    step algebra (Program.update c ifcond) (prepared c scale head chunk storeFlag carry s)=
      some (updated c scale head chunk storeFlag ifcond carry s) := by
  cases ifcond <;> by_cases first : chunk=0
  all_goals simp [Program.update,Program.later,Program.carryDot,Program.put,Program.comp,
    Program.ref,Program.dot,Program.cast,Program.add,Program.preciseDot,
    step,run,evalExpr,evalComputeOp,evalOp_unfold,algebra,dotTerms,ofFloat,toFloat,
    numeric,numericEq,bop,Option.bind,ComputeDType.eraseDType,
    prepared,updated,State.setReg,saved_regs,pointers,hc,hi,first]
  all_goals rfl

theorem updated_output [Inhabited α] (c : Shape) (scale : ℝ) (head chunk : Nat)
    (storeFlag ifcond : Bool) (carry : Matrix α c.D c.D) (s : State (Term α)) :
    (updated c scale head chunk storeFlag ifcond carry s).regs .real [32,c.D] "b_o"=
      some (output c (read s "q") (read s "k") (read s "v") scale head chunk carry ifcond) := by
  cases ifcond <;> by_cases first : chunk=0 <;>
    simp [updated,output,prepared,first]

theorem updated_carry [Inhabited α] (c : Shape) (scale : ℝ) (head chunk : Nat)
    (storeFlag ifcond : Bool) (carry : Matrix α c.D c.D) (s : State (Term α)) :
    (updated c scale head chunk storeFlag ifcond carry s).regs .real [c.D,c.D] "b_h"=
      some (nextCarry c (read s "k") (read s "v") head chunk carry ifcond) := by
  cases ifcond <;> by_cases first : chunk=0 <;> simp [updated,nextCarry,first]

theorem updated_pointer [Inhabited α] (c : Shape) (scale : ℝ) (head chunk : Nat)
    (storeFlag ifcond : Bool) (carry : Matrix α c.D c.D) (s : State (Term α)) :
    (updated c scale head chunk storeFlag ifcond carry s).regs .blockPtr [32,c.D] "p_o"=
      some (fun _ => mainPtr c head chunk "o") := by
  cases ifcond <;> by_cases first : chunk=0 <;>
    simp [updated,prepared,saved_regs,pointers,first]

def finished [Inhabited α] (c : Shape) (scale : ℝ) (head chunk : Nat)
    (storeFlag ifcond : Bool) (carry : Matrix α c.D c.D) (s : State (Term α)) : State (Term α) :=
  scatter 32 c.D (mainPtr c head chunk "o") [0]
    (fun i => TermBody.cast (output c (read s "q") (read s "k") (read s "v") scale head chunk carry ifcond i))
    (updated c scale head chunk storeFlag ifcond carry s)

theorem body_run [Inhabited α] (c : Shape) (scale : ℝ) (head chunk : Nat) (storeFlag ifcond : Bool)
    (carry : Matrix α c.D c.D) (s : State (Term α))
    (hh : s.regs .int [] "i_bh"=some (fun _ => (head : Int)))
    (hi : s.regs .nat [] "i"=some (fun _ => chunk))
    (hc : s.regs .real [c.D,c.D] "b_h"=some carry) :
    run algebra (Program.body c scale storeFlag ifcond) s=
      some (finished c scale head chunk storeFlag ifcond carry s) := by
  simp only [Program.body,run_append,pointer_run c head chunk s hh hi,Option.bind_some,
    save_run c head chunk storeFlag carry s hc,loads_run]
  simp only [run,update_step c scale head chunk storeFlag ifcond carry s hc hi,Option.bind]
  change (step algebra (Program.store 32 c.D "p_o" "b_o" [0])
    (updated c scale head chunk storeFlag ifcond carry s)).bind (fun t => some t)=_
  rw [store_step 32 c.D "p_o" "b_o" (mainPtr c head chunk "o") [0] _ _
    (updated_pointer c scale head chunk storeFlag ifcond carry s)
    (updated_output c scale head chunk storeFlag ifcond carry s)]
  rfl

#axiomsClean pointer_run
#axiomsClean store_step
#axiomsClean save_run
#axiomsClean loads_run
#axiomsClean update_step
#axiomsClean body_run
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.SharedStep
