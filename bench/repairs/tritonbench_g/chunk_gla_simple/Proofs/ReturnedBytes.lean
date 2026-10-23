import bench.repairs.tritonbench_g.chunk_gla_simple.Proofs.ByteMemory

namespace VeriTile.Bench.Repairs.ChunkGlaSimple.ReturnedBytes
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple Host
open VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions (Graph)
open ByteMemory
noncomputable section
set_option maxHeartbeats 8000000

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

theorem input_pointer (r : Admission.Request) (a : CheckedHost.Allocation r) (input : Input) :
    a.layout.base input.region=(input.tensor r).byteBase/4 := by
  cases input
  · exact a.queryPointer
  · exact a.keyPointer
  · exact a.valuePointer
  · exact a.historyPointer
  · exact a.gatePointer

theorem input_byteBase (r : Admission.Request) (checked : Admission.check r=Bool.true)
    (nonempty : r.shape.heads*r.shape.T*r.shape.V≠0) (a : CheckedHost.Allocation r) (input : Input) :
    (input.tensor r).byteBase=4*a.layout.base input.region := by
  rw [input_pointer]
  have aligned:=(input_address r checked nonempty input).1
  omega

/-- Every original source input is decoded from the four bytes beginning at
the actual checked data pointer, including offset and overlapping input views. -/
theorem input_read (r : Admission.Request) (checked : Admission.check r=Bool.true)
    (nonempty : r.shape.heads*r.shape.T*r.shape.V≠0) (a : CheckedHost.Allocation r)
    (before : Memory) (scratch : State NativeGraph.Expr) (input : Input) (i : Nat) :
    ((PhysicalDirect.fpInput a.layout (words before) scratch).mem input.region i).read .real=
      .leaf (read before ((input.tensor r).byteBase+4*i)) := by
  rw [PhysicalDirect.fpInput_read]
  simp only [words,input_byteBase r checked nonempty a input,Nat.mul_add]

theorem output_count (r : Admission.Request) (a : CheckedHost.Allocation r) :
    a.tensor.shape.prod=capacity r.shape "o" := by
  have shape:=a.valid.2.2.2.2.2.2.2.2
  rw [shape]
  simp [capacity,Shape.heads,Nat.mul_assoc]

structure Returned where
  host : CheckedHost.Returned
  bytes : Memory

/-- The checked source supplies each stored word. The bridge only performs
the scalar four-byte stores at the returned tensor's actual byte pointer. -/
def realize (I : ScalarGraph.WordPrimitives) (before : Memory) (result : CheckedHost.Returned) : Returned :=
  ⟨result,stores before result.tensor.byteBase result.tensor.shape.prod
    (fun i=>(result.memory (result.tensor.byteBase/4+i)).eval I.isa)⟩

def run (I : ScalarGraph.WordPrimitives) (r : Admission.Request) (rt : CheckedHost.Runtime r)
    (before : Memory) (scratch : State NativeGraph.Expr) (caller : Nat) : Except CheckedHost.Failure Returned :=
  match CheckedHost.run r rt (words before) scratch caller with
  | .error failure=>.error failure
  | .ok result=>.ok (realize I before result)

theorem returned (I : ScalarGraph.WordPrimitives) (r : Admission.Request) (checked : Admission.check r=Bool.true)
    (rt : CheckedHost.Runtime r) (before : Memory) (scratch : State NativeGraph.Expr)
    (caller : Nat) (result : Returned) (success : run I r rt before scratch caller=.ok result) :
    ∃ a : CheckedHost.Allocation r,rt.allocate=.ok a ∧ result.host.objectId=a.layout.objectId ∧
      result.host.tensor=a.tensor ∧ result.host.currentDevice=caller ∧
      CheckedHost.run r rt (words before) scratch caller=.ok result.host ∧
      (∀ h t v,read result.bytes (a.tensor.byteBase+4*(h.val*(r.shape.T*r.shape.V)+t.val*r.shape.V+v.val))=
        (ScalarGraph.value r.shape (PhysicalDirect.fpInput a.layout (words before) scratch) r.scale.runtimeWord h t v
          (Admission.block_bounds r.shape ((Admission.check_iff r).mp checked).1).1).eval I.isa) ∧
      (∀ p,p<a.tensor.byteBase ∨ a.tensor.byteBase+4*a.tensor.shape.prod≤p → result.bytes p=before p) ∧
      (r.shape.heads*r.shape.T*r.shape.V≠0 → ∀ (input : Input) i,
        ((PhysicalDirect.fpInput a.layout (words before) scratch).mem input.region i).read .real=
          .leaf (read before ((input.tensor r).byteBase+4*i))) ∧
      (r.shape.heads*r.shape.T*r.shape.V≠0 → ∀ (input : Input) i,i<capacity r.shape input.region → ∀ j : Fin 4,
        result.bytes ((input.tensor r).byteBase+4*i+j.val)=before ((input.tensor r).byteBase+4*i+j.val)) := by
  unfold run at success
  cases source : CheckedHost.run r rt (words before) scratch caller with
  | error failure=>simp [source] at success
  | ok host=>
    simp only [source,Except.ok.injEq] at success
    subst result
    have post:=(CheckedHost.wrapper_complete r rt (words before) scratch caller).2
    rw [source] at post
    rcases post with ⟨a,allocation,identity,tensor,device,launch,memory,setup,frame,inputs⟩
    refine ⟨a,allocation,identity,tensor,device,rfl,?_,?_,
      fun nonempty=>input_read r checked nonempty a before scratch,?_⟩
    · intro h t v
      dsimp only [realize]
      rw [stores_render,tensor]
      rw [render_read before a.tensor.byteBase a.tensor.shape.prod _ _
        (by rw [output_count]; exact PhysicalDirect.tensor_bound r.shape h t v)]
      rw [a.outputPointer]
      have div : (4*a.layout.base "o")/4=a.layout.base "o" := by omega
      rw [div]
      rw [memory,PhysicalDirect.graph_readback]
    · intro p outside
      dsimp only [realize]
      rw [stores_render,tensor]
      exact render_frame _ _ _ _ p outside
    · intro nonempty input i hi j
      dsimp only [realize]
      rw [stores_render,tensor]
      apply render_frame
      rw [input_byteBase r checked nonempty a input,a.outputPointer,output_count]
      have fresh:=a.layout.fresh input.region input.role
      omega

#axiomsClean input_read
#axiomsClean returned
end
end VeriTile.Bench.Repairs.ChunkGlaSimple.ReturnedBytes
