import bench.repairs.tritonbench_g.triton_matmul.Proofs.ByteHost
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.FPEquiv
open VeriTile.Triton FP.Structural
open Contract TensorBinding InputAdapter
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Original logical payloads, fixed K16/K32 live-carry instruction groups,
all padded lanes, and the original storage cast. No source execution or output
memory is consulted by this independent floating-point recurrence. -/
def expected (R : Native.Reference) (read : Read) (a b : Tensor)
    (la : Laws (storage a.metadata) a) (lb : Laws (storage b.metadata) b) (i : OutputIndex a b) : Word :=
  logicalReference (Native.numerics (Native.referenceInstructions R) (storage a.metadata)) (request a b)
    (CheckedHost.left read id a b la) (CheckedHost.right read id a b lb) i.1 i.2

/-- Complete checked native source and physical-memory result. Only fixed
instruction/encoding correspondences are numerical premises; variable-shape
dot, finite indexing, grid composition, allocation/return, and byte readback
are derived. FP16 and E4M3FN keep their original tile and pipeline profiles. -/
specification matmul_checked_if_local (rest : Algebra Word) (I : Native.Instructions) (R : Native.Reference)
    (numeric : Native.Correspondence I R) (allocate : ByteHost.Allocator)
    (read : Read) (a b : Tensor) (la : Laws (storage a.metadata) a) (lb : Laws (storage b.metadata) b)
    (h : InputContract a.metadata b.metadata) (initial : State Word)
    (allocates : ByteHost.AllocationLaw allocate a b (initial,read)) (bound : Inputs read id a b initial) :
    ∃ c final bytes, ∃ lc : Laws (storage a.metadata) c,
      ByteHost.Allocated allocate a b (initial,read) c ∧
      ByteHost.call rest I allocate a b (initial,read)=some (c,(final,bytes)) ∧
      CheckedHost.Post a b c initial final (expected R read a b la lb) ∧
      ByteHost.Post read bytes a b c lc (expected R read a b la lb) := by
  obtain ⟨c,final,bytes,lc,allocated,called,logical,physical⟩ := ByteHost.correct rest I allocate read a b la lb h initial allocates bound
  have instruction := Native.instructions_eq I R numeric
  refine ⟨c,final,bytes,lc,allocated,called,?_,?_⟩
  · simpa only [expected,instruction] using logical
  · simpa only [expected,instruction] using physical

#specNonCircular expected avoiding [ByteHost.call,FiniteHost.call,Source.typed,FP.Structural.exec]
#axiomsClean matmul_checked_if_local
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.FPEquiv
