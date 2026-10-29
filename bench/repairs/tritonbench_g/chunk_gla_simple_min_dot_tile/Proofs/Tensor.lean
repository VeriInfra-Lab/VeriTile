import bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.Proofs.ByteMemory

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.Tensor
open VeriTile Triton FP.Structural
open Host ByteMemory
open VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions (Graph)
noncomputable section
set_option maxHeartbeats 8000000

/-- Ordinary empty_like metadata and freshness. Every input base is the
original checked tensor's actual data pointer; no input materialization occurs. -/
structure Binding (r : Admission.Request) where
  layout : PhysicalDirect.Layout r.shape
  tensor : Admission.Tensor
  valid : tensor.Valid r.key.device [r.shape.B,r.shape.H,r.shape.T,r.shape.V]
  outputPointer : tensor.byteBase=4*layout.base "o"
  outputDevice : layout.outputDevice=r.key.device
  queryPointer : layout.base "q"=r.query.byteBase/4
  keyPointer : layout.base "k"=r.key.byteBase/4
  valuePointer : layout.base "v"=r.value.byteBase/4
  historyPointer : layout.base "h"=r.history.byteBase/4
  gatePointer : layout.base "g"=r.gate.byteBase/4

inductive Input where | query | key | value | history | gate
def Input.region : Input → RegionName
  | .query=>"q" | .key=>"k" | .value=>"v" | .history=>"h" | .gate=>"g"
def Input.tensor (r : Admission.Request) : Input → Admission.Tensor
  | .query=>r.query | .key=>r.key | .value=>r.value | .history=>r.history | .gate=>r.gate

theorem Input.member (r : Admission.Request) (input : Input) : input.tensor r∈r.inputs := by
  cases input <;> simp [Input.tensor,Admission.Request.inputs]
theorem Input.role (input : Input) : input.region∈(["q","k","v","h","g"] : List RegionName) := by
  cases input <;> simp [Input.region]

theorem input_address (r : Admission.Request) (checked : Admission.check r=Bool.true)
    (nonempty : r.shape.heads*r.shape.T*r.shape.V≠0) (input : Input) :
    Admission.Address (input.tensor r) :=
  (((Admission.check_iff r).mp checked).2.2.2.2.2.2.2 nonempty).2.2 _ (input.member r)

theorem input_pointer (r : Admission.Request) (a : Binding r) (input : Input) :
    a.layout.base input.region=(input.tensor r).byteBase/4 := by
  cases input
  · exact a.queryPointer
  · exact a.keyPointer
  · exact a.valuePointer
  · exact a.historyPointer
  · exact a.gatePointer

theorem input_byteBase (r : Admission.Request) (checked : Admission.check r=Bool.true)
    (nonempty : r.shape.heads*r.shape.T*r.shape.V≠0) (a : Binding r) (input : Input) :
    (input.tensor r).byteBase=4*a.layout.base input.region := by
  rw [input_pointer]
  have aligned:=(input_address r checked nonempty input).1
  omega

/-- Every original source input is decoded from the four bytes beginning at
the actual checked data pointer, including offset and overlapping input views. -/
theorem input_read (r : Admission.Request) (checked : Admission.check r=Bool.true)
    (nonempty : r.shape.heads*r.shape.T*r.shape.V≠0) (a : Binding r)
    (before : Memory) (scratch : State NativeGraph.Expr) (input : Input) (i : Nat) :
    ((PhysicalDirect.fpInput a.layout (words before) scratch).mem input.region i).read .real=
      .leaf (read before ((input.tensor r).byteBase+4*i)) := by
  rw [PhysicalDirect.fpInput_read]
  simp only [words,input_byteBase r checked nonempty a input,Nat.mul_add]

theorem output_count (r : Admission.Request) (a : Binding r) :
    a.tensor.shape.prod=capacity r.shape "o" := by
  have shape:=a.valid.2.2.2.2.2.2.2.2
  rw [shape]
  simp [capacity,Shape.heads,Nat.mul_assoc]

/-- The result reads a constructed source execution. No successful runtime,
compiler, launch, allocator, or whole-output premise supplies these values. -/
def source (r : Admission.Request) (a : Binding r) (before : Memory)
    (scratch : State NativeGraph.Expr) : State NativeGraph.Expr :=
  PhysicalDirect.graphSource r.shape (PhysicalDirect.fpInput a.layout (words before) scratch)

def values (r : Admission.Request) (a : Binding r) (before : Memory)
    (scratch : State NativeGraph.Expr) (I : ScalarGraph.WordPrimitives) : Nat → Word :=
  fun i => (ScalarGraph.lower r.scale.runtimeWord ((source r a before scratch).mem "o" i |>.read .fp32)).eval I.isa

structure Returned where
  objectId : Nat
  tensor : Admission.Tensor
  bytes : Memory

def result (r : Admission.Request) (a : Binding r) (before : Memory)
    (scratch : State NativeGraph.Expr) (I : ScalarGraph.WordPrimitives) : Returned :=
  ⟨a.layout.objectId,a.tensor,stores before a.tensor.byteBase a.tensor.shape.prod (values r a before scratch I)⟩

/-- Complete source-to-returned-Tensor endpoint. Word primitives are fixed
scalar operations. All masks, loops, stores, casts and source success are proved.
Device allocation and resource availability delimit physical applicability. -/
theorem complete (r : Admission.Request) (checked : Admission.check r=Bool.true)
    (a : Binding r) (before : Memory) (scratch : State NativeGraph.Expr)
    (I : ScalarGraph.WordPrimitives) :
    FPGrid.run ScalarGraph.algebra r.shape ScalarGraph.scaleTag (FPGrid.programs r.shape)
      (PhysicalDirect.fpInput a.layout (words before) scratch)=some (source r a before scratch) ∧
    (∀ nonempty : r.shape.heads*r.shape.T*r.shape.V≠0,
      (source r a before scratch).mem=(NativeABI.grid ScalarGraph.algebra r.shape (ScalarGraph.dots r.shape)
        (Admission.accepted r checked nonempty) ScalarGraph.scaleTag (FPGrid.programs r.shape)
        (PhysicalDirect.fpInput a.layout (words before) scratch)).mem) ∧
    (result r a before scratch I).objectId=a.layout.objectId ∧
    (result r a before scratch I).tensor=a.tensor ∧
    (∀ h t v,read (result r a before scratch I).bytes
      (a.tensor.byteBase+4*(h.val*(r.shape.T*r.shape.V)+t.val*r.shape.V+v.val))=
      (ScalarGraph.value r.shape (PhysicalDirect.fpInput a.layout (words before) scratch) r.scale.runtimeWord h t v
        (Admission.block_bounds r.shape ((Admission.check_iff r).mp checked).1).1).eval I.isa) ∧
    (∀ p,p<a.tensor.byteBase ∨ a.tensor.byteBase+4*a.tensor.shape.prod≤p →
      (result r a before scratch I).bytes p=before p) ∧
    (r.shape.heads*r.shape.T*r.shape.V≠0 → ∀ (input : Input) i,
      ((PhysicalDirect.fpInput a.layout (words before) scratch).mem input.region i).read .real=
        .leaf (read before ((input.tensor r).byteBase+4*i))) ∧
    (r.shape.heads*r.shape.T*r.shape.V≠0 → ∀ (input : Input) i,i<capacity r.shape input.region → ∀ j : Fin 4,
      (result r a before scratch I).bytes ((input.tensor r).byteBase+4*i+j.val)=
        before ((input.tensor r).byteBase+4*i+j.val)) := by
  refine ⟨(PhysicalDirect.graphSource_spec _ _).1,?_,rfl,rfl,?_,?_,
    fun nonempty=>input_read r checked nonempty a before scratch,?_⟩
  · intro nonempty
    exact PhysicalDirect.graph_native _ (Admission.accepted r checked nonempty) _
  · intro h t v
    dsimp only [result]
    rw [stores_render,render_read before a.tensor.byteBase a.tensor.shape.prod _ _
      (by rw [output_count]; exact PhysicalDirect.tensor_bound r.shape h t v)]
    unfold values source
    rw [FPGrid.returned ScalarGraph.algebra r.shape
      (Admission.block_bounds r.shape ((Admission.check_iff r).mp checked).1).1
      ScalarGraph.scaleTag (ScalarGraph.dots r.shape) _ _ (PhysicalDirect.graphSource_spec _ _).2 h t v]
    simp only [Cell.read_same,ScalarGraph.value]
  · intro p outside
    dsimp only [result]
    rw [stores_render]
    exact render_frame _ _ _ _ p outside
  · intro nonempty input i hi j
    dsimp only [result]
    rw [stores_render]
    apply render_frame
    rw [input_byteBase r checked nonempty a input,a.outputPointer,output_count]
    have fresh:=a.layout.fresh input.region input.role
    omega

/-- Empty output still has its returned tensor metadata; no byte is touched. -/
theorem empty (r : Admission.Request) (a : Binding r) (before : Memory)
    (scratch : State NativeGraph.Expr) (I : ScalarGraph.WordPrimitives)
    (h : r.shape.heads*r.shape.T*r.shape.V=0) :
    (result r a before scratch I).bytes=before := by
  have zero : a.tensor.shape.prod=0 := by rw [output_count]; simpa [capacity] using h
  funext p
  simp [result,stores,zero]

#axiomsClean input_read
#axiomsClean complete
#axiomsClean empty
end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.Tensor
