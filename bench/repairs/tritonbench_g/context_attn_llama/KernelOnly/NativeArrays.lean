import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.FPReferenceGrid
import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.NativeInstructions
import VeriTile.Triton.Float.ArrayProgramIO

/-! Independent typed arrays and ordered scatter. Named roles may alias the
output; a floating store updates floating views and decodes to zero through
integer views. No source executor or raw state occurs in these definitions. -/
namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.NativeArrays
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnLlama
noncomputable section
open scoped Classical
variable {α : Type} [Inhabited α]

abbrev ports (c : Host.Call) : List ArrayIOChannel :=
  [⟨.real,c.regions.query⟩,⟨.real,c.regions.key⟩,⟨.real,c.regions.value⟩,
   ⟨.int,c.regions.start⟩,⟨.int,c.regions.sequence⟩,⟨.int,c.regions.prompt⟩,
   ⟨.int,c.regions.request⟩,⟨.int,c.regions.pages⟩,⟨.real,c.regions.out⟩]
abbrev Inputs (c : Host.Call) (α : Type) := ArrayValues α (ports c)

def tileInput (c : Host.Call) (idx : GridIndex c.grid) (a : Inputs c α) : FPReference.Inputs α where
  query:=a 0
  key:=a 1
  value:=a 2
  batch:=Formula.batch c.layout (c.launchAt idx)
  head:=Formula.head c.layout (c.launchAt idx)
  kvHead:=Formula.head c.layout (c.launchAt idx)/c.layout.kvGroup
  block:=(c.launchAt idx).block
  start:=a 3 (Formula.batch c.layout (c.launchAt idx))
  length:=Int.sub (a 4 (Formula.batch c.layout (c.launchAt idx))) (a 5 (Formula.batch c.layout (c.launchAt idx)))
  prompt:=a 5 (Formula.batch c.layout (c.launchAt idx))
  request:=a 6 (Formula.batch c.layout (c.launchAt idx))
  pages:=a 7

def metadata (c : Host.Call) (a : Inputs c α) : Prop :=
  (∀ b,b<c.batches → a 3 b=(c.start b : Int)) ∧
  (∀ b,b<c.batches → a 4 b=((c.length b+c.prompt b : Nat) : Int)) ∧
  (∀ b,b<c.batches → a 5 b=(c.prompt b : Int))

/-- Decoded views after exactly one .real store. -/
def store (c : Host.Call) (offset : Nat) (value : α) (a : Inputs c α) : Inputs c α :=
  fun port off=>match port with
  | ⟨0,_⟩ => if c.regions.query=c.regions.out ∧ off=offset then value else a 0 off
  | ⟨1,_⟩ => if c.regions.key=c.regions.out ∧ off=offset then value else a 1 off
  | ⟨2,_⟩ => if c.regions.value=c.regions.out ∧ off=offset then value else a 2 off
  | ⟨3,_⟩ => if c.regions.start=c.regions.out ∧ off=offset then (0 : Int) else a 3 off
  | ⟨4,_⟩ => if c.regions.sequence=c.regions.out ∧ off=offset then (0 : Int) else a 4 off
  | ⟨5,_⟩ => if c.regions.prompt=c.regions.out ∧ off=offset then (0 : Int) else a 5 off
  | ⟨6,_⟩ => if c.regions.request=c.regions.out ∧ off=offset then (0 : Int) else a 6 off
  | ⟨7,_⟩ => if c.regions.pages=c.regions.out ∧ off=offset then (0 : Int) else a 7 off
  | ⟨8,_⟩ => if off=offset then value else a 8 off

/-- Same row-major scatter order as the source store, including collisions. -/
def scatter (c : Host.Call) (idx : GridIndex c.grid)
    (value : Values α .real [c.layout.BM,c.layout.D]) (a : Inputs c α) : Inputs c α :=
  (TileShape.allIndices [c.layout.BM,c.layout.D]).foldl (fun t i=>
    if c.active idx i then store c (c.offset idx i) (value i) t else t) a

def tile (c : Host.Call) (M : Algebra α) (dots : FPSuccess.DotSupport c.layout M)
    (idx : GridIndex c.grid) (a : Inputs c α) : Inputs c α :=
  scatter c idx (FPReference.output c.layout M dots (Host.scale c.layout.D) (tileInput c idx a)) a

def grid (c : Host.Call) (M : Algebra α) (dots : FPSuccess.DotSupport c.layout M) :
    List (GridIndex c.grid) → Inputs c α → Inputs c α
  | [],a=>a
  | idx::rest,a=>grid c M dots rest (tile c M dots idx a)

/-- Prefix reads include metadata even for an inactive query block. Key-loop
reads conservatively include every token before keyEnd, even for an inactive
query block. Negative requests and pages retain the original toNat conversion. -/
def tileReads (c : Host.Call) (idx : GridIndex c.grid) (a : Inputs c α)
    (port : Fin (ports c).length) : List Nat :=
  let v:=tileInput c idx a
  let tokens:=List.range (FPReference.keyEnd c.layout v).toNat
  let dims:=List.finRange c.layout.D
  let pageAddress:=fun (token : Nat)=>((c.layout.reqRow : Int)*v.request+(c.layout.reqToken : Int)*token).toNat
  match port with
  | ⟨0,_⟩ => (TileShape.allIndices [c.layout.BM,c.layout.D]).filterMap fun i=>
      if (FPReference.row c.layout v i.1.val : Int)<v.length then
        some (((v.start+FPReference.row c.layout v i.1.val)*c.layout.qRow+
          (v.head*c.layout.qHead : Nat)+(i.2.1.val*c.layout.qDim : Nat)).toNat) else none
  | ⟨1,_⟩ => tokens.flatMap fun token=>dims.map fun d=>
      ((v.pages (pageAddress token))*c.layout.kRow+(v.kvHead*c.layout.kHead : Nat)+
        (d.val*c.layout.kDim : Nat)).toNat
  | ⟨2,_⟩ => tokens.flatMap fun token=>dims.map fun d=>
      ((v.pages (pageAddress token))*c.layout.vRow+(v.kvHead*c.layout.vHead : Nat)+
        (d.val*c.layout.vDim : Nat)).toNat
  | ⟨3,_⟩ | ⟨4,_⟩ | ⟨5,_⟩ | ⟨6,_⟩ => [v.batch]
  | ⟨7,_⟩ => tokens.map pageAddress
  | ⟨8,_⟩ => []

def reads (c : Host.Call) (M : Algebra α) (dots : FPSuccess.DotSupport c.layout M) :
    List (GridIndex c.grid) → Inputs c α → Fin (ports c).length → List Nat
  | [],_,_=>[]
  | idx::rest,a,port=>tileReads c idx a port ++ reads c M dots rest (tile c M dots idx a) port

#specNonCircular grid avoiding [FP.Structural.exec,FP.Structural.run,FPSuccess.runGrid,FPReference.grid]
#specNonCircular reads avoiding [FP.Structural.exec,FP.Structural.run,FPSuccess.runGrid,FPReference.grid]
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.NativeArrays
