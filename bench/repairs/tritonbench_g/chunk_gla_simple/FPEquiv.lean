import bench.repairs.tritonbench_g.chunk_gla_simple.Proofs.ReturnedBytes

namespace VeriTile.Bench.Repairs.ChunkGlaSimple.FPEquiv
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple Host
open VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions
noncomputable section

/-- Actual returned FP32 words: the complete source graph retains dot carries,
ordered scalar FMA chains, masked zero words, gates, scale word and casts.
Only fixed local scalar/ISA operations are numerical ports. -/
theorem returned_fp32 (I : ScalarGraph.WordPrimitives) (r : Admission.Request) (checked : Admission.check r=Bool.true)
    (rt : CheckedHost.Runtime r) (before : Nat → BitVec 32) (scratch : State NativeGraph.Expr)
    (caller : Nat) (result : CheckedHost.Returned)
    (success : CheckedHost.run r rt before scratch caller=.ok result) :
    ∃ a : CheckedHost.Allocation r,rt.allocate=.ok a ∧ result.objectId=a.layout.objectId ∧ result.tensor=a.tensor ∧
      (∀ h t v,(result.memory (a.layout.base "o"+(h.val*(r.shape.T*r.shape.V)+t.val*r.shape.V+v.val))).eval I.isa=
        (ScalarGraph.value r.shape (PhysicalDirect.fpInput a.layout before scratch) r.scale.runtimeWord h t v
          (Admission.block_bounds r.shape ((Admission.check_iff r).mp checked).1).1).eval I.isa) ∧
      (∀ p,¬(PhysicalDirect.output a.layout).Contains p → (result.memory p).eval I.isa=before p) := by
  have post:=(CheckedHost.wrapper_complete r rt before scratch caller).2
  rw [success] at post
  rcases post with ⟨a,allocation,identity,tensor,device,launch,memory,setup,frame,inputs⟩
  refine ⟨a,allocation,identity,tensor,?_,?_⟩
  · intro h t v
    rw [memory,PhysicalDirect.graph_readback]
  · intro p outside
    exact congrArg (Graph.eval I.isa) (frame p outside)

theorem wrapper_fp32 (I : ScalarGraph.WordPrimitives) (r : Admission.Request) (checked : Admission.check r=Bool.true)
    (rt : CheckedHost.Runtime r) (before : Nat → BitVec 32) (scratch : State NativeGraph.Expr) (caller : Nat) :
    CheckedHost.ResultSpec r rt before scratch caller (CheckedHost.run r rt before scratch caller) ∧
    (∀ result,CheckedHost.run r rt before scratch caller=.ok result →
      ∃ a : CheckedHost.Allocation r,rt.allocate=.ok a ∧ result.objectId=a.layout.objectId ∧ result.tensor=a.tensor ∧
      (∀ h t v,(result.memory (a.layout.base "o"+(h.val*(r.shape.T*r.shape.V)+t.val*r.shape.V+v.val))).eval I.isa=
        (ScalarGraph.value r.shape (PhysicalDirect.fpInput a.layout before scratch) r.scale.runtimeWord h t v
          (Admission.block_bounds r.shape ((Admission.check_iff r).mp checked).1).1).eval I.isa) ∧
      (∀ p,¬(PhysicalDirect.output a.layout).Contains p → (result.memory p).eval I.isa=before p)) :=
  ⟨(CheckedHost.wrapper_complete r rt before scratch caller).2,
    fun result success=>returned_fp32 I r checked rt before scratch caller result success⟩

abbrev native_abi := @CheckedHost.native_abi
abbrev local_numerical_transport := @ScalarGraph.transport
/-- Raw four-byte FP32 input decoding and returned tensor readback at the
checked actual byte pointers, with sequential scalar stores and a full byte
frame. Read-only input aliases remain permitted. -/
abbrev returned_bytes := @ReturnedBytes.returned
abbrev scalar_byte_stores := @ByteMemory.stores_render
#axiomsClean wrapper_fp32
#axiomsClean returned_fp32
#axiomsClean returned_bytes
#axiomsClean scalar_byte_stores
#axiomsClean native_abi
#axiomsClean local_numerical_transport
end
end VeriTile.Bench.Repairs.ChunkGlaSimple.FPEquiv
