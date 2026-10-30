import bench.tritonbench_g.bmm_optimized.Sources.Native

namespace VeriTile.Bench.Repairs.BmmOptimized.Host
open Native
open scoped BigOperators

structure View where
  identity : Nat
  device : Nat
  batchStride : Nat
  rowStride : Nat
  columnStride : Nat
  storageWords : Nat
  negative : Bool
def View.address (v : View) (b row col : Nat) :=
  b*v.batchStride+row*v.rowStride+col*v.columnStride
def View.Valid (v : View) (batch rows cols : Nat) : Prop :=
  0<v.batchStride ∧ 0<v.rowStride ∧ 0<v.columnStride ∧
  (0<batch → 0<rows → 0<cols →
    (batch-1)*v.batchStride+(rows-1)*v.rowStride+(cols-1)*v.columnStride<v.storageWords)

theorem view_address_bound (v : View) (batch rows cols b row col : Nat)
    (valid : v.Valid batch rows cols) (hb : b<batch) (hr : row<rows) (hc : col<cols) :
    v.address b row col<v.storageWords := by
  have b := Nat.mul_le_mul_right v.batchStride (show b≤batch-1 by omega)
  have r := Nat.mul_le_mul_right v.rowStride (show row≤rows-1 by omega)
  have c := Nat.mul_le_mul_right v.columnStride (show col≤cols-1 by omega)
  have limit := valid.2.2.2 (by omega) (by omega) (by omega)
  unfold View.address
  omega

structure Call where
  shape : Shape
  a : View
  b : View
  freshIdentity : Nat
structure Valid (call : Call) : Prop where
  indices : IndexABI call.shape
  a : call.a.Valid call.shape.batch call.shape.m call.shape.k
  b : call.b.Valid call.shape.batch call.shape.k call.shape.n
  sameDevice : call.b.device=call.a.device
  inputDisjoint : 0<call.shape.batch*call.shape.m*call.shape.k →
    0<call.shape.batch*call.shape.k*call.shape.n → call.a.identity≠call.b.identity
  outputA : call.freshIdentity≠call.a.identity
  outputB : call.freshIdentity≠call.b.identity

def logical {α : Type} (neg : α → α) (v : View) (m : Memory α) (r : Region)
    (b row col : Nat) : α :=
  if v.negative then neg (m r (v.address b row col)) else m r (v.address b row col)
def packedRead {α : Type} (neg : α → α) (v : View) (m : Memory α) (r : Region)
    (rows cols address : Nat) : α :=
  logical neg v m r (address/cols/rows) (address/cols%rows) (address%cols)

theorem packedRead_address {α : Type} (neg : α → α) (v : View) (m : Memory α)
    (r : Region) (rows cols b row col : Nat) (hr : row<rows) (hc : col<cols) :
    packedRead neg v m r rows cols ((b*rows+row)*cols+col)=logical neg v m r b row col := by
  have hp : 0<cols := by omega
  have hq : 0<rows := by omega
  have divide : ((b*rows+row)*cols+col)/cols=b*rows+row := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hp,Nat.div_eq_of_lt hc];simp
  have outer : (b*rows+row)/rows=b := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hq,Nat.div_eq_of_lt hr];simp
  have rem : ((b*rows+row)*cols+col)%cols=col := by
    simp [Nat.add_mod,Nat.mod_eq_of_lt hc]
  have rowrem : (b*rows+row)%rows=row := by simp [Nat.add_mod,Nat.mod_eq_of_lt hr]
  simp only [packedRead,divide,outer,rem,rowrem]

/-- Logical host materialization of resolve_neg/resolve_conj/contiguous.
FP32 conjugation is identity. The stage regions represent normalized operands
even when a runtime contiguous operation reuses the read-only caller storage. -/
def prepare {α : Type} (neg : α → α) (call : Call) (m : Memory α) : Memory α :=
  fun r a => match r with
  | .a => packedRead neg call.a m .callerA call.shape.m call.shape.k a
  | .b => packedRead neg call.b m .callerB call.shape.k call.shape.n a
  | _ => m r a
def prepareBytes (p : Instructions) (call : Call) (m : Bytes) : Bytes :=
  fun r a => match r with
  | .a => packedRead p.neg call.a (load m) .callerA call.shape.m call.shape.k (a/4)
      ⟨a%4,Nat.mod_lt _ (by decide)⟩
  | .b => packedRead p.neg call.b (load m) .callerB call.shape.k call.shape.n (a/4)
      ⟨a%4,Nat.mod_lt _ (by decide)⟩
  | _ => m r a

theorem load_prepare (p : Instructions) (call : Call) (m : Bytes) :
    load (prepareBytes p call m)=prepare p.neg call (load m) := by
  funext r a i
  have hd : (4*a+i.val)/4=a := by omega
  have hm : (4*a+i.val)%4=i.val := by omega
  cases r <;> simp only [load,prepareBytes,prepare,hd,hm]
theorem prepare_a {α : Type} (neg : α → α) (call : Call) (m : Memory α)
    (b row k : Nat) (hr : row<call.shape.m) (hk : k<call.shape.k) :
    prepare neg call m .a (call.shape.a b row k)=logical neg call.a m .callerA b row k :=
  packedRead_address neg call.a m .callerA _ _ b row k hr hk
theorem prepare_b {α : Type} (neg : α → α) (call : Call) (m : Memory α)
    (b k col : Nat) (hk : k<call.shape.k) (hc : col<call.shape.n) :
    prepare neg call m .b (call.shape.b b k col)=logical neg call.b m .callerB b k col :=
  packedRead_address neg call.b m .callerB _ _ b k col hk hc

/-- Initial logical tensor target, independent of preparation and execution. -/
noncomputable def product (call : Call) (m : Memory ℝ) (x : Cell) : ℝ :=
  ∑ k ∈ Finset.range call.shape.k,
    logical Neg.neg call.a m .callerA x.batch x.row k *
      logical Neg.neg call.b m .callerB x.batch k x.column
def chain (p : Instructions) (call : Call) (m : Memory Word) (x : Cell) : Nat → Word
  | 0 => zero
  | k+1 => p.fmaRN
      (if k<call.shape.k then logical p.neg call.a m .callerA x.batch x.row k else zero)
      (if k<call.shape.k then logical p.neg call.b m .callerB x.batch k x.column else zero)
      (chain p call m x k)
def expected (p : Instructions) (call : Call) (m : Bytes) (x : Cell) :=
  chain p call (load m) x (((call.shape.k+31)/32)*32)

theorem prepared_product (call : Call) (m : Memory ℝ) (x : Cell) (hx : call.shape.Active x) :
    mathematical call.shape (prepare Neg.neg call m) x=product call m x := by
  apply Finset.sum_congr rfl
  intro k hk
  rw [prepare_a _ call m _ _ k hx.2.1 (Finset.mem_range.mp hk),
      prepare_b _ call m _ k _ (Finset.mem_range.mp hk) hx.2.2]
theorem prepared_chain (p : Instructions) (call : Call) (m : Memory Word)
    (x : Cell) (hx : call.shape.Active x) (n : Nat) :
    reference call.shape (scalar p) (prepare p.neg call m) x n=chain p call m x n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [reference,chain,ih]
    simp only [scalar]
    by_cases hn : n<call.shape.k
    · simp only [hn,↓reduceIte,prepare_a _ call m _ _ n hx.2.1 hn,
        prepare_b _ call m _ n _ hn hx.2.2]
    · simp only [hn,↓reduceIte]

def empty (s : Shape) : Bool := decide (s.batch=0 ∨ s.m=0 ∨ s.n=0)
theorem empty_programs (s : Shape) (c : Config) (h : empty s=true) : programs s c=[] := by
  have dims : s.batch=0 ∨ s.m=0 ∨ s.n=0 := by simpa [empty] using h
  have cb := config_bounds c
  rcases dims with hb|hm|hn
  · simp [programs,hb]
  · have grid : s.gx c=0 := by
      simp only [Shape.gx,blocks,hm,Nat.zero_add]
      exact Nat.div_eq_of_lt (by omega)
    simp [programs,grid]
  · have grid : s.gy c=0 := by
      simp only [Shape.gy,blocks,hn,Nat.zero_add]
      exact Nat.div_eq_of_lt (by omega)
    simp [programs,grid]

structure Returned (α : Type) where
  identity : Nat
  shape : Nat×Nat×Nat
  strides : Nat×Nat×Nat
  fp32 : Bool
  device : Nat
  restoredDevice : Nat
  memory : Memory α
def outputStrides (s : Shape) : Nat×Nat×Nat := (max s.m 1*max s.n 1,max s.n 1,1)
def invoke (call : Call) (c : Config) (p : Instructions) (callerDevice : Nat) (m : Bytes) : Returned Byte :=
  let staged := prepareBytes p call m
  ⟨call.freshIdentity,(call.shape.batch,call.shape.m,call.shape.n),
    outputStrides call.shape,true,call.a.device,callerDevice,
    if empty call.shape then staged else selected call.shape c p staged⟩
noncomputable def invokeReal (call : Call) (c : Config) (callerDevice : Nat) (m : Memory ℝ) : Returned ℝ :=
  let staged := prepare Neg.neg call m
  ⟨call.freshIdentity,(call.shape.batch,call.shape.m,call.shape.n),
    outputStrides call.shape,true,call.a.device,callerDevice,
    if empty call.shape then staged else source call.shape c realScalar staged⟩

theorem invoke_memory (call : Call) (c : Config) (p : Instructions) (callerDevice : Nat) (m : Bytes) :
    (invoke call c p callerDevice m).memory=selected call.shape c p (prepareBytes p call m) := by
  unfold invoke
  split_ifs with h
  · rw [selected,empty_programs call.shape c h]
    rfl
  · rfl
theorem invokeReal_memory (call : Call) (c : Config) (callerDevice : Nat) (m : Memory ℝ) :
    (invokeReal call c callerDevice m).memory=source call.shape c realScalar (prepare Neg.neg call m) := by
  unfold invokeReal
  split_ifs with h
  · rw [source,empty_programs call.shape c h]
    rfl
  · rfl

theorem whole_real (call : Call) (c : Config) (callerDevice : Nat) (m : Memory ℝ)
    (x : Cell) (hx : call.shape.Active x) :
    (invokeReal call c callerDevice m).memory .output (call.shape.o x.batch x.row x.column)=product call m x := by
  rw [invokeReal_memory,source_grid_real _ c _ x hx,prepared_product call m x hx]
theorem whole_native (call : Call) (c : Config) (p : Instructions) (callerDevice : Nat)
    (m : Bytes) (x : Cell) (hx : call.shape.Active x) (i : Fin 4) :
    (invoke call c p callerDevice m).memory .output (4*call.shape.o x.batch x.row x.column+i.val)=
      expected p call m x i := by
  rw [invoke_memory,selected_read _ c p _ x hx,load_prepare]
  exact congrFun (prepared_chain p call (load m) x hx _) i

def CallerFrame (r : Region) (a : Nat) (s : Shape) : Prop :=
  r=.callerA ∨ r=.callerB ∨ (∃ id,r=.other id) ∨ (r=.output ∧ 4*s.size≤a)
theorem whole_frame (call : Call) (c : Config) (p : Instructions) (callerDevice : Nat)
    (m : Bytes) (r : Region) (a : Nat) (hf : CallerFrame r a call.shape) :
    (invoke call c p callerDevice m).memory r a=m r a := by
  rw [invoke_memory]
  rcases hf with ha|hb|⟨id,ho⟩|⟨ho,hbound⟩
  · subst r;rw [selected_frame _ c p _ _ a (Or.inl (by simp))];rfl
  · subst r;rw [selected_frame _ c p _ _ a (Or.inl (by simp))];rfl
  · subst r;rw [selected_frame _ c p _ _ a (Or.inl (by simp))];rfl
  · subst r;rw [selected_frame _ c p _ _ a (Or.inr hbound)];rfl

theorem returned_tensor (call : Call) (valid : Valid call) (c : Config) (p : Instructions)
    (callerDevice : Nat) (m : Bytes) :
    (invoke call c p callerDevice m).identity=call.freshIdentity ∧
    (invoke call c p callerDevice m).identity≠call.a.identity ∧
    (invoke call c p callerDevice m).identity≠call.b.identity ∧
    (invoke call c p callerDevice m).shape=(call.shape.batch,call.shape.m,call.shape.n) ∧
    (invoke call c p callerDevice m).fp32=true ∧
    (invoke call c p callerDevice m).device=call.a.device ∧
    (invoke call c p callerDevice m).restoredDevice=callerDevice :=
  ⟨rfl,valid.outputA,valid.outputB,rfl,rfl,rfl,rfl⟩

#specNonCircular product avoiding [prepare,prepareBytes,invoke,invokeReal,source,sourceValue,selected]
#specNonCircular expected avoiding [prepare,prepareBytes,invoke,invokeReal,source,sourceValue,selected]
#axiomsClean view_address_bound
#axiomsClean packedRead_address
#axiomsClean load_prepare
#axiomsClean whole_real
#axiomsClean whole_native
#axiomsClean whole_frame
#axiomsClean returned_tensor
end VeriTile.Bench.Repairs.BmmOptimized.Host
