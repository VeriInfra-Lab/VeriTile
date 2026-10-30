import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.rotary_emb_nopad.Proofs.KernelInterface
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.RotationIO
open VeriTile.Triton FP.Structural KernelInterface
noncomputable section
set_option autoImplicit false

abbrev UShape (g : Geometry) := [4,1,g.request.blockPairs]
abbrev CShape (g : Geometry) := [g.request.blockPairs]
abbrev ULane (g : Geometry) := TileIndex (UShape g)
abbrev CLane (g : Geometry) := TileIndex (CShape g)

def region (key : Bool) : RegionName := if key then "K" else "Q"
def coefficient (sine : Bool) : RegionName := if sine then "Sin" else "Cos"
def payloadPort (key odd : Bool) : Fin 6 :=
  ⟨(if key then 2 else 0)+(if odd then 1 else 0),by cases key <;> cases odd <;> decide⟩
def outputPort (key odd : Bool) : Fin 4 :=
  ⟨(if key then 2 else 0)+(if odd then 1 else 0),by cases key <;> cases odd <;> decide⟩

def uInputs (g : Geometry) : List IOChannel :=
  [⟨.real,UShape g,"Q"⟩,⟨.real,UShape g,"Q"⟩,
   ⟨.real,UShape g,"K"⟩,⟨.real,UShape g,"K"⟩,
   ⟨.real,UShape g,"Cos"⟩,⟨.real,UShape g,"Sin"⟩]
def uOutputs (g : Geometry) : List IOChannel := (uInputs g).take 4

def uToken {g : Geometry} (pid1 : Nat) (i : ULane g) := pid1*4+i.1.val

def uOffset (g : Geometry) (pid0 pid1 : Nat) (key odd : Bool) (i : ULane g) :=
  uToken pid1 i*(g.request.heads key*g.request.width)+
    (if key then pid0/g.request.groups else pid0)*g.request.width+
    (i.2.2.1.val+if odd then g.half else 0)
def uCoefficient (g : Geometry) (pid1 : Nat) (sine : Bool) (i : ULane g) :=
  uToken pid1 i*(if sine then g.sinWidth else g.cosWidth)+i.2.2.1.val

def uCoeffMask (g : Geometry) (pid1 : Nat) (i : ULane g) : Prop :=
  uToken pid1 i<g.tokens ∧ i.2.2.1.val<g.half
def uMask (g : Geometry) (pid0 pid1 : Nat) (key : Bool) (i : ULane g) : Prop :=
  (key=Bool.true ∨ pid0<g.qHeads) ∧ uCoeffMask g pid1 i ∧
    (key=Bool.true → pid0%g.request.groups=0)

/-- The first four input channels alias the four in-place output windows.
Cos/sin have their own active mask, including executions with no payload store. -/
def uMemory (α : Type) (g : Geometry) (pid0 pid1 : Nat) : MemoryIO α where
  inputs := uInputs g
  outputs := uOutputs g
  scratch := []
  buffers := ["Q","K","Cos","Sin"]
  read
    | ⟨0,_⟩,_,i => uOffset g pid0 pid1 Bool.false Bool.false i
    | ⟨1,_⟩,_,i => uOffset g pid0 pid1 Bool.false Bool.true i
    | ⟨2,_⟩,_,i => uOffset g pid0 pid1 Bool.true Bool.false i
    | ⟨3,_⟩,_,i => uOffset g pid0 pid1 Bool.true Bool.true i
    | ⟨4,_⟩,_,i => uCoefficient g pid1 Bool.false i
    | ⟨5,_⟩,_,i => uCoefficient g pid1 Bool.true i
  inputMask
    | ⟨0,_⟩,_,i | ⟨1,_⟩,_,i => uMask g pid0 pid1 Bool.false i
    | ⟨2,_⟩,_,i | ⟨3,_⟩,_,i => uMask g pid0 pid1 Bool.true i
    | ⟨4,_⟩,_,i | ⟨5,_⟩,_,i => uCoeffMask g pid1 i
  write
    | ⟨0,_⟩,_,i => uOffset g pid0 pid1 Bool.false Bool.false i
    | ⟨1,_⟩,_,i => uOffset g pid0 pid1 Bool.false Bool.true i
    | ⟨2,_⟩,_,i => uOffset g pid0 pid1 Bool.true Bool.false i
    | ⟨3,_⟩,_,i => uOffset g pid0 pid1 Bool.true Bool.true i
  outputMask
    | ⟨0,_⟩,_,i | ⟨1,_⟩,_,i => uMask g pid0 pid1 Bool.false i
    | ⟨2,_⟩,_,i | ⟨3,_⟩,_,i => uMask g pid0 pid1 Bool.true i
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k

def uInvocation {α : Type} (rest : Algebra α) (ops : Ops α)
    (g : Geometry) (pid0 pid1 : Nat) : Invocation α where
  kernel := Kernels.uncached (Typed.uLayout g.request)
  backend := Typed.algebra rest ops
  pids := fun ax => if ax=0 then pid0 else pid1
  setPids := fun ax => ax==0 || ax==1

def uncachedIO {α : Type} (rest : Algebra α) (ops : Ops α)
    (g : Geometry) (pid0 pid1 : Nat) : ProgramIO α α where
  toMemoryIO := uMemory α g pid0 pid1
  schedule := fun _ => [uInvocation rest ops g pid0 pid1]
  encode := id
  observe := fun _ _ => id

/-- Independent split-half rotation; the ordinary Real instance is
(a*c-b*z, a*z+b*c). -/
def rotation {α : Type} (ops : Ops α) (a b c z : α) (odd : Bool) : α :=
  if odd then ops.add (ops.mul a z) (ops.mul b c) else ops.sub (ops.mul a c) (ops.mul b z)

def uTarget {α : Type} (ops : Ops α) (g : Geometry) (v : ChannelValues α (uInputs g)) :
    ChannelValues α (uOutputs g)
  | ⟨0,_⟩ => fun i => rotation ops (v (0 : Fin 6) i) (v (1 : Fin 6) i) (v (4 : Fin 6) i) (v (5 : Fin 6) i) Bool.false
  | ⟨1,_⟩ => fun i => rotation ops (v (0 : Fin 6) i) (v (1 : Fin 6) i) (v (4 : Fin 6) i) (v (5 : Fin 6) i) Bool.true
  | ⟨2,_⟩ => fun i => rotation ops (v (2 : Fin 6) i) (v (3 : Fin 6) i) (v (4 : Fin 6) i) (v (5 : Fin 6) i) Bool.false
  | ⟨3,_⟩ => fun i => rotation ops (v (2 : Fin 6) i) (v (3 : Fin 6) i) (v (4 : Fin 6) i) (v (5 : Fin 6) i) Bool.true

def cInputs (g : Geometry) : List IOChannel :=
  [⟨.real,CShape g,"Q"⟩,⟨.real,CShape g,"Q"⟩,
   ⟨.real,CShape g,"K"⟩,⟨.real,CShape g,"K"⟩,
   ⟨.real,CShape g,"Cos"⟩,⟨.real,CShape g,"Sin"⟩,
   ⟨.int,[],"Lengths"⟩,⟨.int,[],"Tables"⟩]
def cOutputs (g : Geometry) : List IOChannel :=
  [⟨.real,CShape g,"Q"⟩,⟨.real,CShape g,"Q"⟩,
   ⟨.real,CShape g,"Cache"⟩,⟨.real,CShape g,"Cache"⟩]
def contextLength {α : Type} (g : Geometry) (v : ChannelValues α (cInputs g)) : Int := v (6 : Fin 8) PUnit.unit
def past {α : Type} (g : Geometry) (v : ChannelValues α (cInputs g)) : Int := contextLength g v-1
def page {α : Type} (g : Geometry) (v : ChannelValues α (cInputs g)) : Int := v (7 : Fin 8) PUnit.unit

def cInput (g : Geometry) (pid0 pid1 : Nat) (key odd : Bool) (i : CLane g) :=
  pid1*(g.request.heads key*g.request.width)+pid0*g.request.width+(i.1.val+if odd then g.half else 0)
def cCoefficient (g : Geometry) (pid1 : Nat) (sine : Bool) (i : CLane g) :=
  pid1*(if sine then g.sinWidth else g.cosWidth)+i.1.val
def tableOffset {α : Type} (g : Geometry) (pid1 : Nat) (v : ChannelValues α (cInputs g)) :=
  pid1*g.tableColumns+((past g v).tdiv g.blockSize).toNat
def cacheOffset {α : Type} (g : Geometry) (pid0 : Nat) (v : ChannelValues α (cInputs g))
    (odd : Bool) (i : CLane g) :=
  (page g v*(g.kHeads*g.blockSize*g.request.width)+(pid0*(g.blockSize*g.request.width) : Nat)+
    (past g v).tmod g.blockSize*g.request.width+(i.1.val+if odd then g.half else 0 : Nat)).toNat

def cCoeffMask (g : Geometry) (pid0 : Nat) (i : CLane g) : Prop :=
  pid0<g.qHeads ∧ i.1.val<g.half
def cMask (g : Geometry) (pid0 : Nat) (key : Bool) (i : CLane g) : Prop :=
  cCoeffMask g pid0 i ∧ (key=Bool.false ∨ pid0<g.kHeads)

/-- Signed metadata is an ordinary logical input. The selected table cell and
cache destinations depend on it. Every access is suppressed on the early head
return; K is read only, Q aliases its output, and cache has separate outputs. -/
def cMemory (α : Type) (g : Geometry) (pid0 pid1 : Nat) : MemoryIO α where
  inputs := cInputs g
  outputs := cOutputs g
  scratch := []
  buffers := ["Q","K","Cos","Sin","Lengths","Tables","Cache"]
  read
    | ⟨0,_⟩,_,i => cInput g pid0 pid1 Bool.false Bool.false i
    | ⟨1,_⟩,_,i => cInput g pid0 pid1 Bool.false Bool.true i
    | ⟨2,_⟩,_,i => cInput g pid0 pid1 Bool.true Bool.false i
    | ⟨3,_⟩,_,i => cInput g pid0 pid1 Bool.true Bool.true i
    | ⟨4,_⟩,_,i => cCoefficient g pid1 Bool.false i
    | ⟨5,_⟩,_,i => cCoefficient g pid1 Bool.true i
    | ⟨6,_⟩,_,_ => pid1
    | ⟨7,_⟩,v,_ => tableOffset g pid1 v
  inputMask
    | ⟨0,_⟩,_,i | ⟨1,_⟩,_,i => cMask g pid0 Bool.false i
    | ⟨2,_⟩,_,i | ⟨3,_⟩,_,i => cMask g pid0 Bool.true i
    | ⟨4,_⟩,_,i | ⟨5,_⟩,_,i => cCoeffMask g pid0 i
    | ⟨6,_⟩,_,_ | ⟨7,_⟩,_,_ => pid0<g.qHeads
  write
    | ⟨0,_⟩,_,i => cInput g pid0 pid1 Bool.false Bool.false i
    | ⟨1,_⟩,_,i => cInput g pid0 pid1 Bool.false Bool.true i
    | ⟨2,_⟩,v,i => cacheOffset g pid0 v Bool.false i
    | ⟨3,_⟩,v,i => cacheOffset g pid0 v Bool.true i
  outputMask
    | ⟨0,_⟩,_,i | ⟨1,_⟩,_,i => cMask g pid0 Bool.false i
    | ⟨2,_⟩,_,i | ⟨3,_⟩,_,i => cMask g pid0 Bool.true i
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k

def cInvocation {α : Type} (rest : Algebra α) (ops : Ops α)
    (g : Geometry) (pid0 pid1 : Nat) : Invocation α where
  kernel := Kernels.cached (Typed.cLayout g.request)
  backend := Typed.algebra rest ops
  pids := fun ax => if ax=0 then pid0 else pid1
  setPids := fun ax => ax==0 || ax==1

def cachedIO {α : Type} (rest : Algebra α) (ops : Ops α)
    (g : Geometry) (pid0 pid1 : Nat) : ProgramIO α α where
  toMemoryIO := cMemory α g pid0 pid1
  schedule := fun _ => [cInvocation rest ops g pid0 pid1]
  encode := id
  observe := fun _ _ => id

def cTarget {α : Type} (ops : Ops α) (g : Geometry) (v : ChannelValues α (cInputs g)) :
    ChannelValues α (cOutputs g)
  | ⟨0,_⟩ => fun i => rotation ops (v (0 : Fin 8) i) (v (1 : Fin 8) i) (v (4 : Fin 8) i) (v (5 : Fin 8) i) Bool.false
  | ⟨1,_⟩ => fun i => rotation ops (v (0 : Fin 8) i) (v (1 : Fin 8) i) (v (4 : Fin 8) i) (v (5 : Fin 8) i) Bool.true
  | ⟨2,_⟩ => fun i => rotation ops (v (2 : Fin 8) i) (v (3 : Fin 8) i) (v (4 : Fin 8) i) (v (5 : Fin 8) i) Bool.false
  | ⟨3,_⟩ => fun i => rotation ops (v (2 : Fin 8) i) (v (3 : Fin 8) i) (v (4 : Fin 8) i) (v (5 : Fin 8) i) Bool.true

/-- Original signed page/slot premises, expressed on logical metadata. -/
def MetadataDomain {α : Type} (g : Geometry) (v : ChannelValues α (cInputs g)) : Prop :=
  0≤past g v ∧ 0≤page g v

/-- Selected native scalar graph; input provenance is unrestricted, output
provenance is the exact atom generated by the source arithmetic. -/
def nativeRotation (p : Native.Instructions) (cached : Bool) (a b c z : Native.Node) (odd : Bool) :=
  Native.atom (Typed.splitGraph p cached a.value b.value c.value z.value odd)
def uNativeTarget (p : Native.Instructions) (g : Geometry) (v : ChannelValues Native.Node (uInputs g)) :
    ChannelValues Native.Node (uOutputs g)
  | ⟨0,_⟩ => fun i => nativeRotation p Bool.false (v (0 : Fin 6) i) (v (1 : Fin 6) i) (v (4 : Fin 6) i) (v (5 : Fin 6) i) Bool.false
  | ⟨1,_⟩ => fun i => nativeRotation p Bool.false (v (0 : Fin 6) i) (v (1 : Fin 6) i) (v (4 : Fin 6) i) (v (5 : Fin 6) i) Bool.true
  | ⟨2,_⟩ => fun i => nativeRotation p Bool.false (v (2 : Fin 6) i) (v (3 : Fin 6) i) (v (4 : Fin 6) i) (v (5 : Fin 6) i) Bool.false
  | ⟨3,_⟩ => fun i => nativeRotation p Bool.false (v (2 : Fin 6) i) (v (3 : Fin 6) i) (v (4 : Fin 6) i) (v (5 : Fin 6) i) Bool.true

def cNativeTarget (p : Native.Instructions) (g : Geometry) (v : ChannelValues Native.Node (cInputs g)) :
    ChannelValues Native.Node (cOutputs g)
  | ⟨0,_⟩ => fun i => nativeRotation p Bool.true (v (0 : Fin 8) i) (v (1 : Fin 8) i) (v (4 : Fin 8) i) (v (5 : Fin 8) i) Bool.false
  | ⟨1,_⟩ => fun i => nativeRotation p Bool.true (v (0 : Fin 8) i) (v (1 : Fin 8) i) (v (4 : Fin 8) i) (v (5 : Fin 8) i) Bool.true
  | ⟨2,_⟩ => fun i => nativeRotation p Bool.true (v (2 : Fin 8) i) (v (3 : Fin 8) i) (v (4 : Fin 8) i) (v (5 : Fin 8) i) Bool.false
  | ⟨3,_⟩ => fun i => nativeRotation p Bool.true (v (2 : Fin 8) i) (v (3 : Fin 8) i) (v (4 : Fin 8) i) (v (5 : Fin 8) i) Bool.true
/-- Named logical split-half values. All Q/K and coefficient tile lanes remain. -/
structure SplitInputs (α I : Type) where
  payload : Bool → Bool → I → α
  cosine : I → α
  sine : I → α

structure CachedInputs (α I : Type) extends SplitInputs α I where
  contextLength : Int
  page : Int

def logicalUncachedInputs {α : Type} (g : Geometry) :
    ChannelValues α (uInputs g) ≃ SplitInputs α (ULane g) where
  toFun := fun xs => {
    payload := fun key odd => if key then if odd then xs (3 : Fin 6) else xs (2 : Fin 6)
      else if odd then xs (1 : Fin 6) else xs (0 : Fin 6)
    cosine := xs (4 : Fin 6)
    sine := xs (5 : Fin 6) }
  invFun := fun xs => Fin.cases (xs.payload Bool.false Bool.false)
    (Fin.cases (xs.payload Bool.false Bool.true) (Fin.cases (xs.payload Bool.true Bool.false)
      (Fin.cases (xs.payload Bool.true Bool.true) (Fin.cases xs.cosine
        (Fin.cases xs.sine (fun i => i.elim0))))))
  left_inv := by intro xs; funext port; fin_cases port <;> rfl
  right_inv := by
    rintro ⟨values,co,si⟩
    dsimp only
    congr 1
    funext key odd
    cases key <;> cases odd <;> rfl

def logicalCachedInputs {α : Type} (g : Geometry) :
    ChannelValues α (cInputs g) ≃ CachedInputs α (CLane g) where
  toFun := fun xs => {
    payload := fun key odd => if key then if odd then xs (3 : Fin 8) else xs (2 : Fin 8)
      else if odd then xs (1 : Fin 8) else xs (0 : Fin 8)
    cosine := xs (4 : Fin 8)
    sine := xs (5 : Fin 8)
    contextLength := xs (6 : Fin 8) ()
    page := xs (7 : Fin 8) () }
  invFun := fun xs => Fin.cases (xs.payload Bool.false Bool.false)
    (Fin.cases (xs.payload Bool.false Bool.true) (Fin.cases (xs.payload Bool.true Bool.false)
      (Fin.cases (xs.payload Bool.true Bool.true) (Fin.cases xs.cosine
        (Fin.cases xs.sine (Fin.cases (fun _ => xs.contextLength)
          (Fin.cases (fun _ => xs.page) (fun i => i.elim0))))))))
  left_inv := by
    intro xs
    funext port
    fin_cases port <;> try rfl
  right_inv := by
    rintro ⟨⟨values,co,si⟩,length,page⟩
    dsimp only
    congr 2
    funext key odd
    cases key <;> cases odd <;> rfl

/-- Q/cache outputs directly name their key and split-half coordinates. -/
def splitOutputs {α : Type} (shape : TileShape) (q k : RegionName) :
    ChannelValues α [⟨.real,shape,q⟩,⟨.real,shape,q⟩,
      ⟨.real,shape,k⟩,⟨.real,shape,k⟩] ≃
    (Bool → Bool → TileIndex shape → α) where
  toFun := fun xs key odd => if key then if odd then xs 3 else xs 2
    else if odd then xs 1 else xs 0
  invFun := fun f => Fin.cases (f Bool.false Bool.false)
    (Fin.cases (f Bool.false Bool.true) (Fin.cases (f Bool.true Bool.false)
      (Fin.cases (f Bool.true Bool.true) (fun i => i.elim0))))
  left_inv := by intro xs; funext port; fin_cases port <;> rfl
  right_inv := by intro f; funext key odd; cases key <;> cases odd <;> rfl

def logicalUncachedIO {α : Type} (rest : Algebra α) (ops : Ops α)
    (g : Geometry) (pid0 pid1 : Nat) :
    LogicalProgramIO α α (SplitInputs α (ULane g)) (Bool → Bool → ULane g → α) where
  core := uncachedIO rest ops g pid0 pid1
  inputs := logicalUncachedInputs g
  outputs := splitOutputs (UShape g) "Q" "K"

def logicalCachedIO {α : Type} (rest : Algebra α) (ops : Ops α)
    (g : Geometry) (pid0 pid1 : Nat) :
    LogicalProgramIO α α (CachedInputs α (CLane g)) (Bool → Bool → CLane g → α) where
  core := cachedIO rest ops g pid0 pid1
  inputs := logicalCachedInputs g
  outputs := splitOutputs (CShape g) "Q" "Cache"

def LogicalMetadataDomain {α : Type} {I : Type} (xs : CachedInputs α I) : Prop :=
  0≤xs.contextLength-1 ∧ 0≤xs.page

end
end VeriTile.Bench.Repairs.RotaryEmbNopad.RotationIO
