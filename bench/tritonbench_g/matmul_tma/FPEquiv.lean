import bench.tritonbench_g.matmul_tma.Kernels
import bench.tritonbench_g.matmul_tma.Proofs.Native
import bench.tritonbench_g.matmul_tma.Proofs.FPSoftware
import bench.tritonbench_g.matmul_tma.Proofs.PrimitivePorts

/-! Complete source-wrapper contracts. Inputs are the source's explicit FP16,
outputs FP32 or FP16. The generic primitive is opaque. The closed software
profile uses separately rounded FP32 products/additions in ascending K order,
positive-zero seed, and direct output rounding. Equality to native WGMMA and
native conversion policies remains a numerical obligation, not an axiom here. -/
namespace VeriTile.Bench.TritonBenchG.MatmulTma.FPEquiv
open VeriTile.Triton FP.Structural FPValue Host
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- The actual randn/empty sequence, transpose views, single tile launch and
returned object, all eight flags, and arbitrary symbolic shape/configuration.
`ready` admits exactly the source request; it cannot supply execution or data.
No value is assumed for either fresh output before the kernel runs. -/
specification matmul_tma_wrapper_backend_correctness {α : Type} [Inhabited α]
    (M : Algebra α) (D : Dot α) (alloc : Allocator α) (admit : Admission)
    (h : Shape) (f : Flags) (cfg : Config) (r : Regions) (a : FlatAlloc)
    (initial s : State α) (prepared : Prepared alloc h f r initial s)
    (binding : AllocationBinding h f r a)
    (ready : admit (request h f r cfg) = .ok ()) :
    ∃ t, wrapper M D alloc admit h f cfg initial = .ok (returnedTensor h f r,t) ∧
      Post (parameters h f r) f.half s t (outputValues M D (parameters h f r) f.half s) ∧
      (f.half = Bool.true → ∀ off, t.mem r.first off = s.mem r.first off) := by
  obtain ⟨t,he,hp⟩ := wrapper_correct M D alloc admit h f cfg r initial s prepared ready
  exact ⟨t,he,hp,fun half => replacement_framed M D h f r a s t binding half hp⟩

/-- Fully specified software dot and conversion, composed through the same
allocation calls and returned tensor. This is not a native tensor-core claim. -/
specification matmul_tma_wrapper_software_correctness
    (rest : Algebra Word) (alloc : Allocator Word) (admit : Admission)
    (h : Shape) (f : Flags) (cfg : Config) (r : Regions) (a : FlatAlloc)
    (initial s : State Word) (prepared : Prepared alloc h f r initial s)
    (binding : AllocationBinding h f r a)
    (ready : admit (request h f r cfg) = .ok ()) :
    ∃ t, wrapper (software rest .ieeeValue) sequentialDot alloc admit h f cfg initial =
        .ok (returnedTensor h f r,t) ∧
      Post (parameters h f r) f.half s t
        (bitOutput .ieeeValue sequentialDot (parameters h f r) f.half s) ∧
      (f.half = Bool.true → ∀ off, t.mem r.first off = s.mem r.first off) := by
  simpa only [output_software] using matmul_tma_wrapper_backend_correctness
    (software rest .ieeeValue) sequentialDot alloc admit h f cfg r a initial s prepared binding ready

/-- The full wrapper also preserves every pre-existing cell outside its
new allocations. This composes the ordinary allocator API frame with the
proved source frame, relative to memory before the first randn call. -/
specification matmul_tma_wrapper_frame {α : Type} [Inhabited α]
    (M : Algebra α) (D : Dot α) (alloc : Allocator α) (admit : Admission)
    (h : Shape) (f : Flags) (cfg : Config) (r : Regions)
    (initial s : State α) (prepared : Prepared alloc h f r initial s)
    (frame : AllocatorFrame alloc) (ready : admit (request h f r cfg) = .ok ()) :
    ∃ t, wrapper M D alloc admit h f cfg initial = .ok (returnedTensor h f r,t) ∧
      Post (parameters h f r) f.half s t (outputValues M D (parameters h f r) f.half s) ∧
      (∀ R off, OutsideAllocated f r R → t.mem R off = initial.mem R off) := by
  obtain ⟨t,he,hp⟩ := wrapper_correct M D alloc admit h f cfg r initial s prepared ready
  exact ⟨t,he,hp,returned_frame_initial M D alloc h f r initial s t prepared frame hp⟩

/-- Packed native pointer/configuration admission composes with the whole
wrapper. Native arithmetic is still the explicit primitive D; this theorem
also derives allocation-byte bounds, including transpose layouts. -/
specification matmul_tma_admitted_wrapper {α : Type} [Inhabited α]
    (M : Algebra α) (D : Dot α) (alloc : Allocator α)
    (storage : Native.Storage) (setup : Native.PackedRequest → Except String Unit)
    (h : Shape) (f : Flags) (cfg : Config) (r : Regions) (a : FlatAlloc)
    (initial s : State α) (prepared : Prepared alloc h f r initial s)
    (binding : AllocationBinding h f r a) (nativeBinding : Native.Binding h f r storage)
    (ready : Native.Admitted h f r cfg storage setup) :
    Native.AccessesBounded h f r storage ∧
    ∃ t, wrapper M D alloc (Native.admission storage setup) h f cfg initial =
        .ok (returnedTensor h f r,t) ∧
      Post (parameters h f r) f.half s t (outputValues M D (parameters h f r) f.half s) ∧
      (f.half = Bool.true → ∀ off, t.mem r.first off = s.mem r.first off) :=
  ⟨Native.accesses_bounded h f r storage nativeBinding,
    matmul_tma_wrapper_backend_correctness M D alloc (Native.admission storage setup)
      h f cfg r a initial s prepared binding ready.configuration⟩

/-- The public flag enumeration is exhaustive; no transpose/output branch is
replaced by a fixture or omitted from either wrapper theorem above. -/
theorem covers_every_source_branch (f : Flags) : f ∈ allFlags := all_flags f

/-- Exact numerical registration entry: one local row/column dot and the two
scalar output conversions, composed with the same complete admitted wrapper.
The primitive laws are explicit pending numerical obligations. In particular,
the native dot is not equated with the sequential software dot. -/
specification matmul_tma_primitive_wrapper {α : Type} [Inhabited α]
    (M : Algebra α) (D : Dot α) (alloc : Allocator α)
    (storage : Native.Storage) (setup : Native.PackedRequest → Except String Unit)
    (h : Shape) (Q : Primitive.Numerics h.inner α) (law : Primitive.Laws h M D Q)
    (f : Flags) (cfg : Config) (r : Regions) (a : FlatAlloc)
    (initial s : State α) (prepared : Prepared alloc h f r initial s)
    (binding : AllocationBinding h f r a) (nativeBinding : Native.Binding h f r storage)
    (frame : AllocatorFrame alloc) (ready : Native.Admitted h f r cfg storage setup) :
    Native.AccessesBounded h f r storage ∧
    ∃ t, wrapper M D alloc (Native.admission storage setup) h f cfg initial =
        .ok (returnedTensor h f r,t) ∧
      Post (parameters h f r) f.half s t (Primitive.reference h Q f r s) ∧
      (f.half = Bool.true → ∀ off, t.mem r.first off = s.mem r.first off) ∧
      (∀ R off, OutsideAllocated f r R → t.mem R off = initial.mem R off) := by
  obtain ⟨bounds,t,he,hp,hfirst⟩ := matmul_tma_admitted_wrapper M D alloc storage setup
    h f cfg r a initial s prepared binding nativeBinding ready
  refine ⟨bounds,t,he,?_,hfirst,?_⟩
  · simpa only [Primitive.output_eq h M D Q law] using hp
  · exact returned_frame_initial M D alloc h f r initial s t prepared frame hp

-- These are successful-execution contracts, not admitted FP rewrite judgments.
-- Their numerical primitives are explicit parameters; no FP rewrite atom is used.
#axiomsClean matmul_tma_wrapper_frame
#axiomsClean matmul_tma_admitted_wrapper
#axiomsClean matmul_tma_wrapper_backend_correctness
#axiomsClean matmul_tma_wrapper_software_correctness
#axiomsClean matmul_tma_primitive_wrapper
end
end VeriTile.Bench.TritonBenchG.MatmulTma.FPEquiv
