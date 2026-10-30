import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Kernels
import bench.repairs.tritonbench_g.context_attn_mistral.Proofs.FPBackend

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Reference
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
abbrev Ops := VeriTile.Bench.Repairs.ContextAttnMistral.FPBackend.Ops
noncomputable section
variable {α : Type} [Inhabited α]

@[simp] theorem empty_index (i : TileIndex []) : i=PUnit.unit := Subsingleton.elim _ _
@[simp] theorem pair_index {m n : Nat} (i : TileIndex [m,n]) : (i.1,i.2.1,PUnit.unit)=i := by
  rcases i with ⟨r,d,tail⟩
  cases tail
  rfl
@[simp] theorem float_zero : (0.0 : ℝ)=0 := by norm_num
@[simp] theorem float_one : (1.0 : ℝ)=1 := by norm_num

def parts (c : Call) : Source.IntRange :=
  match Source.intRange? (Kernels.source c).surfaceBody "start_n" with
  | some p => p
  | none => ⟨[],.constInt 0,.constInt 0,.constNat 1,[],[]⟩

theorem source_split (c : Call) :
    (Kernels.source c).surfaceBody=(parts c).before ++
      [.forRangeInt "start_n" (parts c).start (parts c).stop (parts c).stride (parts c).body] ++
      (parts c).after := rfl

/-- Six initial floating buffers and the two launch coordinates. No source
register or execution result is an input to the independent recurrence. -/
structure Inputs (α : Type) where
  read : RegionName → Nat → α
  tile : Nat
  hz : Nat

def inputs (s : State α) : Inputs α :=
  ⟨fun r i => (s.mem r i).read .real,s.pids 0,s.pids 1⟩

def row (c : Call) (a : Inputs α) (r : Nat) := a.tile*c.BM+r
def batch (c : Call) (a : Inputs α) := a.hz/c.H
def head (c : Call) (a : Inputs α) := a.hz%c.H
def kvhead (c : Call) (a : Inputs α) := head c a/(c.H/c.HK)
def carryAddress (c : Call) (a : Inputs α) (r : Nat) := a.hz*c.rounded+row c a r

def qptr (c : Call) (a : Inputs α) : BlockPtr :=
  ⟨"Q",batch c a*c.q.batch+head c a*c.q.head,[c.NQ,c.D],[c.BM,c.D],
    [c.q.row,c.q.dim],[a.tile*c.BM,0]⟩
def kptr (c : Call) (a : Inputs α) : BlockPtr :=
  ⟨"K",batch c a*c.k.batch+kvhead c a*c.k.head,[c.D,c.NK],[c.D,c.BN],
    [c.k.dim,c.k.row],[0,0]⟩
def vptr (c : Call) (a : Inputs α) : BlockPtr :=
  ⟨"V",batch c a*c.v.batch+kvhead c a*c.v.head,[c.NK,c.D],[c.BN,c.D],
    [c.v.row,c.v.dim],[0,0]⟩
def optr (c : Call) (a : Inputs α) : BlockPtr :=
  ⟨"O",batch c a*c.o.batch+head c a*c.o.head,[c.rounded,c.D],[c.BM,c.D],
    [c.o.row,c.o.dim],[a.tile*c.BM,0]⟩

def lower (c : Call) (a : Inputs α) : Int :=
  if c.window≠0 ∧ c.complement=0 then
    max ((((a.tile*c.BM : Nat) : Int)+c.offset-c.size+1).tdiv c.BN*c.BN) 0 else 0
def upper (c : Call) (a : Inputs α) : Int :=
  if c.window≠0 ∧ c.complement=0 then
    min (((((a.tile+1)*c.BM : Nat) : Int)-1+c.offset+c.BN).tdiv c.BN*c.BN) c.NK else c.NK

/-- Memory interpretation preserves both unchecked loads and explicit
boundary-zero loads. Admission subsequently proves unchecked addresses valid. -/
def loadBlock (o : Ops α) (a : Inputs α) (p : BlockPtr)
    (shape : TileShape) (checked : List Nat) : Values α .real shape := fun i =>
  o.scalar.fp32Load (if p.inBounds (TileShape.indexToList shape i) checked then
    a.read p.region (p.address (TileShape.indexToList shape i))
    else o.scalar.literal (some .fp32) .real 0)

def query (c : Call) (o : Ops α) (a : Inputs α) : Values α .real [c.BM,c.D] :=
  loadBlock o a (qptr c a) _ (if c.params.evenM≠0 then [] else [0,1])

def scale (c : Call) (o : Ops α) : α :=
  o.scalar.binary (some .fp32) .real .mul
    (o.scalar.literal (some .fp32) .real c.scale)
    (o.scalar.literal (some .fp32) .real 1.4426950408889634)

structure Carry (c : Call) (α : Type) where
  center : Values α .real [c.BM]
  mass : Values α .real [c.BM]
  weighted : Values α .real [c.BM,c.D]
  keyPointer : BlockPtr
  valuePointer : BlockPtr

def initial (c : Call) (o : Ops α) (a : Inputs α) : Carry c α :=
  let M := o.scalar
  { center := if c.init≠0 then
      fun _ => M.binary (some .fp32) .real .add (M.literal (some .fp32) .real 0) M.negInf
      else fun i => M.cast (some .fp32) .real .real
        (a.read "M" (carryAddress c a i.1.val))
    mass := if c.init≠0 then
      fun _ => M.binary (some .fp32) .real .add
        (M.literal (some .fp32) .real 0) (M.literal (some .fp32) .real 1)
      else fun i => M.cast (some .fp32) .real .real
        (a.read "L" (carryAddress c a i.1.val))
    weighted := if c.init≠0 then fun _ => M.literal (some .fp32) .real 0
      else fun i => M.cast (some .fp32) .real .real
        (a.read (optr c a).region ((optr c a).address (TileShape.indexToList [c.BM,c.D] i)))
    keyPointer := if c.window≠0 ∧ c.complement=0 then (kptr c a).advance [0,lower c a] else kptr c a
    valuePointer := if c.window≠0 ∧ c.complement=0 then (vptr c a).advance [lower c a,0] else vptr c a }

def windowMask (c : Call) (a : Inputs α) (cur : Int) (i : TileIndex [c.BM,c.BN]) : Bool :=
  let distance := (i.1.val : Int)-i.2.1.val+a.tile*c.BM-cur+c.offset
  if c.complement≠0 then decide (distance≥c.size)
  else decide (distance≥0) && decide (distance<c.size)

/-- Full online recurrence, including the raw supplied carry, both masks,
source operand order, NaN-alpha branch, and advancing K/V block addresses.
Opaque operations here are an internal generic execution abstraction. -/
def advance (c : Call) (o : Ops α) (a : Inputs α) (cur : Int) (old : Carry c α) : Carry c α :=
  let M := o.scalar
  let k := loadBlock o a old.keyPointer [c.D,c.BN] (if c.params.evenN≠0 then [] else [0,1])
  let raw := o.dot (some .fp32) (batch:=[]) (m:=c.BM) (k:=c.D) (n:=c.BN) (query c o a) k
  let score : Values α .real [c.BM,c.BN] := fun i => M.binary (some .fp32) .real .mul
    (M.binary (some .fp32) .real .add (M.literal (some .fp32) .real 0) (raw i)) (scale c o)
  let windowed : Values α .real [c.BM,c.BN] := fun i =>
    if c.window≠0 then if windowMask c a cur i then score i else M.negInf else score i
  let masked : Values α .real [c.BM,c.BN] := fun i =>
    if c.params.evenN=0 then if (i.2.1.val : Int)+cur<c.NK then windowed i else M.negInf else windowed i
  let reduced := M.reduceMax (some .fp32) (shape:=[c.BM,c.BN]) ⟨1,by simp⟩ Bool.false masked
  let center : Values α .real [c.BM] := fun i => M.binary (some .fp32) .real .max (old.center i) (reduced i)
  let probability : Values α .real [c.BM,c.BN] := fun i =>
    M.unary (some .fp32) .exp2 (M.binary (some .fp32) .real .sub (masked i) (center (i.1,())))
  let windowedProbability : Values α .real [c.BM,c.BN] := fun i =>
    if c.window≠0 then if windowMask c a cur i then probability i else M.literal (some .fp32) .real 0 else probability i
  let p : Values α .real [c.BM,c.BN] := fun i =>
    if c.params.evenN=0 then if (i.2.1.val : Int)+cur<c.NK then windowedProbability i
      else M.literal (some .fp32) .real 0 else windowedProbability i
  let sums := M.reduceSum (some .fp32) (shape:=[c.BM,c.BN]) ⟨1,by simp⟩ Bool.false p
  let alpha : Values α .real [c.BM] := fun i =>
    let diff := M.binary (some .fp32) .real .sub (old.center i) (center i)
    if !(o.eq (some .fp32) .real diff diff) then M.literal (some .fp32) .real 1
    else M.unary (some .fp32) .exp2 diff
  let v := loadBlock o a old.valuePointer [c.BN,c.D] (if c.params.evenN≠0 then [] else [0,1])
  let contribution := o.dot (some .fp32) (batch:=[]) (m:=c.BM) (k:=c.BN) (n:=c.D)
    (fun i => M.cast (some .fp32) .real .real (p i)) v
  { center := center
    mass := fun i => M.binary (some .fp32) .real .add
      (M.binary (some .fp32) .real .mul (old.mass i) (alpha i)) (sums i)
    weighted := fun i => M.binary (some .fp32) .real .add
      (M.binary (some .fp32) .real .mul (old.weighted i) (alpha (i.1,()))) (contribution i)
    keyPointer := old.keyPointer.advance [0,c.BN]
    valuePointer := old.valuePointer.advance [c.BN,0] }

def iterate (c : Call) (o : Ops α) (a : Inputs α) (cur stop : Int) (old : Carry c α) : Carry c α :=
  if hz : c.BN=0 then old else
  if hc : cur<stop then iterate c o a (cur+c.BN) stop (advance c o a cur old) else old
termination_by (stop-cur).toNat
decreasing_by omega

def result (c : Call) (o : Ops α) (a : Inputs α) : Carry c α :=
  iterate c o a (lower c a) (upper c a) (initial c o a)

def Matches (c : Call) (o : Ops α) (a : Inputs α) (old : Carry c α) (s : State α) : Prop :=
  s.regs .nat [] "start_m"=some (fun _ => a.tile) ∧
  s.regs .real [] "qk_scale"=some (fun _ => scale c o) ∧
  s.regs .real [c.BM,c.D] "q"=some (query c o a) ∧
  s.regs .ptr [c.BM] "m_ptrs"=some (fun i : TileIndex [c.BM] => (("M" : RegionName),carryAddress c a i.1.val)) ∧
  s.regs .ptr [c.BM] "l_ptrs"=some (fun i : TileIndex [c.BM] => (("L" : RegionName),carryAddress c a i.1.val)) ∧
  s.regs .blockPtr [c.BM,c.D] "O_block_ptr"=some (fun _ => optr c a) ∧
  s.regs .real [c.BM] "m_i"=some old.center ∧
  s.regs .real [c.BM] "l_i"=some old.mass ∧
  s.regs .real [c.BM,c.D] "acc"=some old.weighted ∧
  s.regs .blockPtr [c.D,c.BN] "K_block_ptr"=some (fun _ => old.keyPointer) ∧
  s.regs .blockPtr [c.BN,c.D] "V_block_ptr"=some (fun _ => old.valuePointer) ∧
  s.regs .int [] "lo"=some (fun _ => lower c a) ∧
  s.regs .int [] "hi"=some (fun _ => upper c a)

#specNonCircular result avoiding [FP.Structural.exec,FP.Structural.run,FP.Structural.step,Kernels.source,parts]
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Reference
