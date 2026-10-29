import bench.tritonbench_g.iv_dependent_matmul.HostDefinitions

namespace VeriTile.Bench.Repairs.IvDependentMatmul.Native
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Kernels Whole
noncomputable section
abbrev I64 := BitVec 64
abbrev word (n : Nat) : I64 := BitVec.ofNat 64 n

def castI32 (n : Nat) : I64 := (BitVec.ofNat 32 n).signExtend 64

def scalarABI (n : Nat) : I64 :=
  if n=1 then word 1 else if n<2^31 then castI32 n else word n

def ceilWord (n b : Nat) : I64 :=
  (scalarABI n+word b-word 1).sdiv (word b)

def rowWord (r : Request) (pid lane : Nat) : I64 :=
  (castI32 pid).sdiv (ceilWord r.columns r.blockColumns)*word r.blockRows+castI32 lane

def colWord (r : Request) (pid lane : Nat) : I64 :=
  (castI32 pid).srem (ceilWord r.columns r.blockColumns)*word r.blockColumns+castI32 lane

def loadMask (r : Request) (k lane : Nat) : Bool :=
  (castI32 lane).slt (scalarABI r.inner-word k*word r.blockInner)

def storeMask (r : Request) (pid : Nat) (q : CIndex r) : Bool :=
  (rowWord r pid q.1.val).slt (scalarABI r.rows) &&
    (colWord r pid q.2.1.val).slt (scalarABI r.columns)

def aBaseWord (r : Request) (pid : Nat) (i : TileIndex [r.blockRows,r.blockInner]) : I64 :=
  ((rowWord r pid i.1.val).srem (scalarABI r.rows))*scalarABI (max r.inner 1)+
    castI32 i.2.1.val*scalarABI 1

def bBaseWord (r : Request) (pid : Nat) (i : TileIndex [r.blockInner,r.blockColumns]) : I64 :=
  castI32 i.1.val*scalarABI (max r.columns 1)+
    ((colWord r pid i.2.1.val).srem (scalarABI r.columns))*scalarABI 1

def aAt (r : Request) (base pid k : Nat) (i : TileIndex [r.blockRows,r.blockInner]) : I64 :=
  (word base+word 4*aBaseWord r pid i)+word 4*(word k*word r.blockInner*scalarABI 1)

def bAt (r : Request) (base pid k : Nat) (i : TileIndex [r.blockInner,r.blockColumns]) : I64 :=
  (word base+word 4*bBaseWord r pid i)+word 4*(word k*word r.blockInner*scalarABI (max r.columns 1))

def cAt (r : Request) (base pid : Nat) (q : CIndex r) : I64 :=
  (word base+word 4*(scalarABI (max r.columns 1)*rowWord r pid q.1.val))+
    word 4*(scalarABI 1*colWord r pid q.2.1.val)

structure Cache (r : Request) where
  ac : TileIndex [r.blockRows,r.blockInner] → I64
  an : TileIndex [r.blockRows,r.blockInner] → I64
  ann : TileIndex [r.blockRows,r.blockInner] → I64
  bc : TileIndex [r.blockInner,r.blockColumns] → I64
  bn : TileIndex [r.blockInner,r.blockColumns] → I64
  bnn : TileIndex [r.blockInner,r.blockColumns] → I64

def initial (r : Request) (ba bb pid : Nat) : Cache r :=
  ⟨aAt r ba pid 0,
    (if r.schedule=.two ∨ r.schedule=.three then aAt r ba pid 1 else aAt r ba pid 0),
    (if r.schedule=.three then aAt r ba pid 2 else aAt r ba pid 0),
    bAt r bb pid 0,
    (if r.schedule=.two ∨ r.schedule=.three then bAt r bb pid 1 else bAt r bb pid 0),
    (if r.schedule=.three then bAt r bb pid 2 else bAt r bb pid 0)⟩

def before (r : Request) (ba bb pid k : Nat) (c : Cache r) : Cache r :=
  match r.schedule with
  | .preLoad => {c with ac:=aAt r ba pid k,bc:=bAt r bb pid k}
  | .mixed => {c with ac:=aAt r ba pid k}
  | _ => c

def after (r : Request) (ba bb pid k : Nat) (c : Cache r) : Cache r :=
  match r.schedule with
  | .preLoad => c
  | .postLoad => {c with ac:=aAt r ba pid (k+1),bc:=bAt r bb pid (k+1)}
  | .mixed => {c with bc:=bAt r bb pid (k+1)}
  | .two => {c with ac:=c.an,bc:=c.bn,an:=aAt r ba pid (k+2),bn:=bAt r bb pid (k+2)}
  | .three => { c with
      ac := c.an
      bc := c.bn
      an := c.ann
      bn := c.bnn
      ann := aAt r ba pid (k+3)
      bnn := bAt r bb pid (k+3) }

def cacheAt (r : Request) (ba bb pid : Nat) : Nat → Cache r
  | 0 => initial r ba bb pid
  | k+1 => after r ba bb pid k (before r ba bb pid k (cacheAt r ba bb pid k))

end
end VeriTile.Bench.Repairs.IvDependentMatmul.Native

namespace VeriTile.Bench.Repairs.IvDependentMatmul.Execution
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Kernels Whole
noncomputable section

def sourceRange {β : Type} (update : Nat → β → β) (cur stop stride : Nat) (initial : β) : β :=
  if hzero : stride=0 then initial
  else if hlt : cur<stop then sourceRange update (cur+stride) stop stride (update cur initial)
  else initial
termination_by stop-cur
decreasing_by omega

abbrev Acc (α : Type) (r : Request) := CIndex r → α
abbrev AValues (α : Type) (r : Request) := TileIndex [r.blockRows,r.blockInner] → α
abbrev BValues (α : Type) (r : Request) := TileIndex [r.blockInner,r.blockColumns] → α

structure Arithmetic (α : Type) (r : Request) where
  zero : α
  add : α → α → α
  dot : AValues α r → BValues α r → Acc α r
  dotAcc : AValues α r → BValues α r → Acc α r → Acc α r
  narrow : α → α
  widen : α → α

def aLoad {α : Type} [Inhabited α] (m : Arithmetic α r) (ba bb pid k : Nat)
    (s : State α) : AValues α r := fun i =>
  if Native.loadMask r k i.2.1.val then
    (s.mem r.A ((((Native.before r ba bb pid k (Native.cacheAt r ba bb pid k)).ac i).toNat-ba)/4)).read .fp32
  else m.zero

def bLoad {α : Type} [Inhabited α] (m : Arithmetic α r) (ba bb pid k : Nat)
    (s : State α) : BValues α r := fun i =>
  if Native.loadMask r k i.1.val then
    (s.mem r.B ((((Native.before r ba bb pid k (Native.cacheAt r ba bb pid k)).bc i).toNat-bb)/4)).read .fp32
  else m.zero

def step {α : Type} [Inhabited α] (m : Arithmetic α r) (graph : Graph)
    (ba bb pid : Nat) (s : State α) (k : Nat) (acc : Acc α r) : Acc α r :=
  match graph with
  | .source => fun q => m.add (acc q) (m.dot (aLoad m ba bb pid k s) (bLoad m ba bb pid k s) q)
  | .accumulator => m.dotAcc (aLoad m ba bb pid k s) (bLoad m ba bb pid k s) acc

def accumulated {α : Type} [Inhabited α] (m : Arithmetic α r) (graph : Graph)
    (ba bb pid : Nat) (s : State α) : Acc α r :=
  sourceRange (step m graph ba bb pid s) 0 (Native.ceilWord r.inner r.blockInner).toNat 1 (fun _ => m.zero)

def program {α : Type} [Inhabited α] (m : Arithmetic α r) (graph : Graph)
    (ba bb bc pid : Nat) (s : State α) : State α :=
  let values := accumulated m graph ba bb pid s
  (TileShape.allIndices [r.blockRows,r.blockColumns]).foldl
    (fun t q => if Native.storeMask r pid q then
      t.write r.C (((Native.cAt r bc pid q).toNat-bc)/4) (.mk .fp32 (m.widen (m.narrow (values q))))
      else t) {s with pids:=fun ax => if ax=0 then pid else 0}

def runPrograms {α : Type} [Inhabited α] (m : Arithmetic α r) (graph : Graph)
    (ba bb bc : Nat) : List Nat → State α → State α
  | [],s => s
  | pid::ids,s => runPrograms m graph ba bb bc ids (program m graph ba bb bc pid s)

def host {α : Type} [Inhabited α] (m : Arithmetic α r) (graph : Graph)
    (ba bb bc : Nat) (s : State α) : TensorMetadata × State α :=
  (outputMetadata r,runPrograms m graph ba bb bc (List.range (programs r)) s)

end
end VeriTile.Bench.Repairs.IvDependentMatmul.Execution
