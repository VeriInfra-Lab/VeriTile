import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.Footprint
import VeriTile.Triton.Float.ArrayProgramIO

/-! Named typed input arrays and source-ordered scatter. All floating roles may
alias output; updates synchronize every decoded view of that region. -/
namespace VeriTile.Bench.Repairs.ContextAttnMistralKernel.NativeArrays
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
open VeriTile.Bench.Repairs.ContextAttnMistral
noncomputable section
open scoped Classical
variable {α : Type} [Inhabited α]

abbrev ports (c : Host.Call) : List ArrayIOChannel :=
  [⟨.real,c.regions.query⟩,⟨.real,c.regions.key⟩,⟨.real,c.regions.value⟩,
   ⟨.int,c.regions.start⟩,⟨.int,c.regions.sequence⟩,⟨.real,c.regions.out⟩]
abbrev Inputs (c : Host.Call) (α : Type) := ArrayValues α (ports c)

def tileInput (c : Host.Call) (idx : GridIndex c.grid) (a : Inputs c α) : KernelReference.Inputs α where
  query:=a 0
  key:=a 1
  value:=a 2
  head:=idx.toPids 1
  kvHead:=idx.toPids 1/c.layout.kvGroup
  block:=idx.toPids 2
  start:=a 3 (idx.toPids 0)
  length:=a 4 (idx.toPids 0)

def metadata (c : Host.Call) (a : Inputs c α) : Prop :=
  (∀ b,b<c.batches → a 3 b=(c.start b : Int)) ∧
  (∀ b,b<c.batches → a 4 b=(c.length b : Int))

/-- One source .real store, through every possibly aliased decoded role. -/
def store (c : Host.Call) (offset : Nat) (value : α) (a : Inputs c α) : Inputs c α :=
  fun port off=>match port with
  | ⟨0,_⟩ => if c.regions.query=c.regions.out ∧ off=offset then value else a 0 off
  | ⟨1,_⟩ => if c.regions.key=c.regions.out ∧ off=offset then value else a 1 off
  | ⟨2,_⟩ => if c.regions.value=c.regions.out ∧ off=offset then value else a 2 off
  | ⟨3,_⟩ => if c.regions.start=c.regions.out ∧ off=offset then (0 : Int) else a 3 off
  | ⟨4,_⟩ => if c.regions.sequence=c.regions.out ∧ off=offset then (0 : Int) else a 4 off
  | ⟨5,_⟩ => if off=offset then value else a 5 off

/-- Exact row-major output scatter, retaining duplicate offsets. -/
def scatter (c : Host.Call) (idx : GridIndex c.grid)
    (value : Values α .real [c.layout.BM,c.layout.D]) (a : Inputs c α) : Inputs c α :=
  (TileShape.allIndices [c.layout.BM,c.layout.D]).foldl (fun t i=>
    if c.active idx i then store c (c.offset idx i) (value i) t else t) a

def tile (c : Host.Call) (o : FPBackend.Ops α)
    (idx : GridIndex c.grid) (a : Inputs c α) : Inputs c α :=
  scatter c idx (KernelReference.output c.layout o (Host.scale c.layout.D) (tileInput c idx a)) a

def grid (c : Host.Call) (o : FPBackend.Ops α) :
    List (GridIndex c.grid) → Inputs c α → Inputs c α
  | [],a=>a
  | idx::rest,a=>grid c o rest (tile c o idx a)

/-- Finite prefix/loop windows. The loop window conservatively includes each
nonnegative position before stop, with the original masks and address casts. -/
def tileReads (c : Host.Call) (idx : GridIndex c.grid) (a : Inputs c α)
    (port : Fin (ports c).length) : List Nat :=
  let v:=tileInput c idx a
  let curs:=List.range (KernelReference.stop c.layout v).toNat
  match port with
  | ⟨0,_⟩ => (TileShape.allIndices [c.layout.BM,c.layout.D]).filterMap fun i=>
      if (KernelReference.row c.layout v i.1.val : Int)<v.length then
        some (((v.start+KernelReference.row c.layout v i.1.val)*c.layout.qRow+
          (v.head*c.layout.qHead : Nat)+(i.2.1.val*c.layout.qDim : Nat)).toNat) else none
  | ⟨1,_⟩ => curs.flatMap fun (cur : Nat)=>(TileShape.allIndices [c.layout.D,c.layout.BN]).filterMap fun i=>
      if (cur : Int)+i.2.1.val<v.length then
        some (i.2.1.val*c.layout.kRow+v.kvHead*c.layout.kHead+i.1.val*c.layout.kDim+
          ((v.start+cur)*c.layout.kRow).toNat) else none
  | ⟨2,_⟩ => curs.flatMap fun (cur : Nat)=>(TileShape.allIndices [c.layout.BN,c.layout.D]).filterMap fun i=>
      if (cur : Int)+i.1.val<v.length then
        some (i.1.val*c.layout.vRow+v.kvHead*c.layout.vHead+i.2.1.val*c.layout.vDim+
          ((v.start+cur)*c.layout.vRow).toNat) else none
  | ⟨3,_⟩ | ⟨4,_⟩ => [idx.toPids 0]
  | ⟨5,_⟩ => []

def reads (c : Host.Call) (o : FPBackend.Ops α) :
    List (GridIndex c.grid) → Inputs c α → Fin (ports c).length → List Nat
  | [],_,_=>[]
  | idx::rest,a,port=>tileReads c idx a port ++ reads c o rest (tile c o idx a) port

#specNonCircular grid avoiding [FP.Structural.exec,FP.Structural.run,KernelReference.runGrid,KernelReference.grid]
#specNonCircular reads avoiding [FP.Structural.exec,FP.Structural.run,KernelReference.runGrid,KernelReference.grid]
end
end VeriTile.Bench.Repairs.ContextAttnMistralKernel.NativeArrays
