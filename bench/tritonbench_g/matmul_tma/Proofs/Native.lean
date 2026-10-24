import bench.tritonbench_g.matmul_tma.Proofs.Allocation

/-! Native API boundary: the exact source descriptors/configuration are sent
to fallible setup. This proves byte-range and pointer-binding facts; compiler,
allocator and GPU-runtime implementation correctness are outside the theorem. -/
namespace VeriTile.Bench.TritonBenchG.MatmulTma.Native
open VeriTile.Triton FP.Structural FPValue Host
noncomputable section
set_option linter.unusedSimpArgs false

structure Storage where
  base : RegionName → Nat
  argumentPointer : RegionName → Nat
  capacityBytes : RegionName → Nat

/-- Distinct live allocations occupy disjoint intervals in the native byte
address space, separately from the element-addressed Real flat placement. -/
def Storage.Disjoint (s : Storage) (a b : RegionName) : Prop :=
  s.base a + s.capacityBytes a ≤ s.base b ∨
  s.base b + s.capacityBytes b ≤ s.base a

def outputBytes (f : Flags) : Nat := if f.half then 2 else 4

/-- Extracted tensor pointers and allocation sizes, independent of any result.
The precondition states only the buffers actually consumed by this launch. -/
structure Binding (h : Shape) (f : Flags) (r : Regions) (s : Storage) : Prop where
  a_pointer : s.argumentPointer r.a = s.base r.a
  b_pointer : s.argumentPointer r.b = s.base r.b
  c_pointer : s.argumentPointer (outputRegion r f) = s.base (outputRegion r f)
  a_capacity : 2 * (h.rows * h.inner) ≤ s.capacityBytes r.a
  b_capacity : 2 * (h.inner * h.columns) ≤ s.capacityBytes r.b
  c_capacity : outputBytes f * (h.rows * h.columns) ≤ s.capacityBytes (outputRegion r f)
  a_width : s.base r.a + s.capacityBytes r.a ≤ 2^64
  b_width : s.base r.b + s.capacityBytes r.b ≤ 2^64
  c_width : s.base (outputRegion r f) + s.capacityBytes (outputRegion r f) ≤ 2^64
  a_b_disjoint : s.Disjoint r.a r.b
  a_c_disjoint : s.Disjoint r.a (outputRegion r f)
  b_c_disjoint : s.Disjoint r.b (outputRegion r f)

structure PackedRequest where
  logical : Request
  aPointer : Nat
  bPointer : Nat
  cPointer : Nat

def pack (s : Storage) (q : Request) : PackedRequest :=
  ⟨q, s.argumentPointer q.parameters.a,
    s.argumentPointer q.parameters.b, s.argumentPointer q.parameters.c⟩

/-- The setup callback receives the exact three extracted pointers, descriptors,
constexpr dimensions, output format, orders, grid and launch configuration.
It cannot see memory contents or provide an execution result. -/
def admission (s : Storage) (setup : PackedRequest → Except String Unit) : Admission :=
  fun q => setup (pack s q)

/-- Required source block-shape form for native tl.make_block_ptr compilation.
Power-of-two dimensions alone do not establish dot/config/architecture support. -/
def PowerOfTwoBlocks (h : Shape) : Prop :=
  (∃ e : Nat, h.rows = 2^e) ∧ (∃ e : Nat, h.columns = 2^e) ∧ (∃ e : Nat, h.inner = 2^e)

/-- Native success is admitted for this exact packed request/configuration.
The compiler remains responsible for tensor-core size, architecture, resources,
NUM_CTAS/NUM_WARPS and block-pointer primitive support. Failed requests remain
ordinary setup errors, and are covered by Host.setup_error. -/
structure Admitted (h : Shape) (f : Flags) (r : Regions) (cfg : Config)
    (s : Storage) (setup : PackedRequest → Except String Unit) : Prop where
  blockShape : PowerOfTwoBlocks h
  configuration : setup (pack s (request h f r cfg)) = .ok ()

theorem demo_block_shape : PowerOfTwoBlocks ⟨128,128,128⟩ :=
  ⟨⟨7,by decide⟩,⟨7,by decide⟩,⟨7,by decide⟩⟩

def AccessesBounded (h : Shape) (f : Flags) (r : Regions) (s : Storage) : Prop :=
  (∀ i : AIndex (parameters h f r),
    s.argumentPointer r.a + 2 * (aOffset (parameters h f r) i + 1) ≤
      s.base r.a + s.capacityBytes r.a ∧
    s.argumentPointer r.a + 2 * (aOffset (parameters h f r) i + 1) ≤ 2^64) ∧
  (∀ i : BIndex (parameters h f r),
    s.argumentPointer r.b + 2 * (bOffset (parameters h f r) i + 1) ≤
      s.base r.b + s.capacityBytes r.b ∧
    s.argumentPointer r.b + 2 * (bOffset (parameters h f r) i + 1) ≤ 2^64) ∧
  (∀ i : CIndex (parameters h f r),
    s.argumentPointer (outputRegion r f) + outputBytes f * (FPValue.cOffset (parameters h f r) i + 1) ≤
      s.base (outputRegion r f) + s.capacityBytes (outputRegion r f) ∧
    s.argumentPointer (outputRegion r f) + outputBytes f * (FPValue.cOffset (parameters h f r) i + 1) ≤ 2^64)

theorem accesses_bounded (h : Shape) (f : Flags) (r : Regions) (s : Storage)
    (binding : Binding h f r s) : AccessesBounded h f r s := by
  refine ⟨fun i => ?_, fun i => ?_, fun i => ?_⟩
  · have hb := Nat.mul_le_mul_left 2 (Nat.succ_le_of_lt (a_bound h f r i))
    have hp : s.argumentPointer r.a + 2 * (aOffset (parameters h f r) i + 1) ≤
        s.base r.a + s.capacityBytes r.a := by
      rw [binding.a_pointer]
      exact Nat.add_le_add_left (hb.trans binding.a_capacity) _
    exact ⟨hp,hp.trans binding.a_width⟩
  · have hb := Nat.mul_le_mul_left 2 (Nat.succ_le_of_lt (b_bound h f r i))
    have hp : s.argumentPointer r.b + 2 * (bOffset (parameters h f r) i + 1) ≤
        s.base r.b + s.capacityBytes r.b := by
      rw [binding.b_pointer]
      exact Nat.add_le_add_left (hb.trans binding.b_capacity) _
    exact ⟨hp,hp.trans binding.b_width⟩
  · have hb := Nat.mul_le_mul_left (outputBytes f) (Nat.succ_le_of_lt (c_bound h f r i))
    have hp : s.argumentPointer (outputRegion r f) +
        outputBytes f * (FPValue.cOffset (parameters h f r) i + 1) ≤
        s.base (outputRegion r f) + s.capacityBytes (outputRegion r f) := by
      rw [binding.c_pointer]
      exact Nat.add_le_add_left (hb.trans binding.c_capacity) _
    exact ⟨hp,hp.trans binding.c_width⟩

/-- Admission stays configuration-specific, including the source's unusual B
order tuple for non-transposed B. No fastest-axis equality is asserted. -/
theorem exact_request (h : Shape) (f : Flags) (r : Regions) (cfg : Config) (s : Storage) :
    (pack s (request h f r cfg)).aPointer = s.argumentPointer r.a ∧
    (pack s (request h f r cfg)).bPointer = s.argumentPointer r.b ∧
    (pack s (request h f r cfg)).cPointer = s.argumentPointer (returnedTensor h f r).region ∧
    (pack s (request h f r cfg)).logical.grid = (1,1) ∧
    (pack s (request h f r cfg)).logical.config = cfg ∧
    (pack s (request h f r cfg)).logical.aOrder = [1,0] ∧
    (pack s (request h f r cfg)).logical.bOrder = [0,1] ∧
    (pack s (request h f r cfg)).logical.cOrder = [1,0] := by
  simp [pack,request,parameters]

#axiomsClean accesses_bounded
#axiomsClean exact_request
end
end VeriTile.Bench.TritonBenchG.MatmulTma.Native
