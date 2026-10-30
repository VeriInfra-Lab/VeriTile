import bench.tritonbench_g.chunk_gate_recurrence.Proofs.KernelReference

/-! Selected FP32 scalar instructions are model data. Product provenance
records source contraction sites, so no split-multiply/add equality is assumed.
The complete reduction plan records all 1024 products, including fused leaves. -/
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelCarrier
open VeriTile.Triton FP.Structural KernelProgram
noncomputable section
open Classical
abbrev Word := BitVec 32
structure Instructions where
  addRN : Word → Word → Word
  mulRN : Word → Word → Word
  fmaRN : Word → Word → Word → Word
def ops (isa : Instructions) : Ops Word := ⟨0,isa.addRN,isa.mulRN,isa.fmaRN⟩
structure Node where
  value : Word
  factors : Option (Word × Word) := none
instance : Inhabited Node := ⟨⟨0,none⟩⟩
def atom (x : Word) : Node := ⟨x,none⟩
def cast (x : Node) : Node := atom x.value
def multiply (isa : Instructions) (a b : Node) : Node :=
  ⟨isa.mulRN a.value b.value,some (a.value,b.value)⟩
def add (isa : Instructions) (swap : Bool) (a b : Node) : Node :=
  match a.factors with
  | some (x,y) => atom (if swap then isa.fmaRN y x b.value else isa.fmaRN x y b.value)
  | none => atom (isa.addRN a.value b.value)
def operands (x : Node) : Word × Word := x.factors.getD (x.value,BitVec.ofNat 32 0x3f800000)
def reduce (isa : Instructions) (plan : Reduction.ProductPlan 1024)
    {shape : TileShape} (axisIndex : Fin shape.length) (keep : Bool) (values : Values Node .real shape) :
    Values Node .real (TileShape.reduceShape shape axisIndex keep) :=
  if hs : shape=[1024] then
    let v : Values Node .real [1024] := hs ▸ values
    fun _ => atom (Reduction.product (ops isa) plan (fun i => operands (v (i,PUnit.unit))))
  else fun _ => atom 0
def algebra (isa : Instructions) (plan : Reduction.ProductPlan 1024) (swap : Bool) : Algebra Node where
  literal := fun _ _ _ => atom 0
  negInf := atom (BitVec.ofNat 32 0xff800000)
  binary := fun _ _ op a b => match op with
    | .mul => multiply isa a b
    | .add => add isa swap a b
    | _ => atom 0
  unary := fun _ _ _ => atom 0
  cast := fun _ _ _ => cast
  fromNat := fun _ _ => atom 0
  fromInt := fun _ _ => atom 0
  fp32Bits := fun w => atom w.bits
  fp32Load := cast
  reduceMax := fun _ {_} _ _ _ _ => atom 0
  reduceSum := fun _ {_} => reduce isa plan

def swapped (kind : Kind) (c : Shape) : Bool := kind==.backward && decide (c.N=1)
def model (isa : Instructions) (plan : Reduction.ProductPlan 1024) (kind : Kind) (c : Shape) :=
  algebra isa plan (swapped kind c)
def storedWord (dtype : FloatDType) (bits : Word) : Cell Node :=
  .mk dtype.toTileDType (ofFloat dtype (atom bits))

theorem step (isa : Instructions) (plan : Reduction.ProductPlan 1024) (swap : Bool) (a b x : Node) :
    KernelReference.update (algebra isa plan swap) a b x=
      atom (if swap then isa.fmaRN b.value a.value x.value else isa.fmaRN a.value b.value x.value) := by
  rfl
theorem stored (isa : Instructions) (plan : Reduction.ProductPlan 1024) (swap : Bool)
    (dtype : FloatDType) (value : Node) :
    KernelReference.stored (algebra isa plan swap) dtype value=storedWord dtype value.value := rfl

#axiomsClean step
#specNonCircular algebra avoiding [FP.Structural.exec,FP.Structural.run]
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelCarrier
