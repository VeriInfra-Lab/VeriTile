import bench.tritonbench_g.rotary_emb.Proofs.FlatIO
import bench.tritonbench_g.rotary_emb.Proofs.Domain

/-! A simultaneous two-buffer IO surface for the complete 3D rotary kernel.
The existing uniform-mask skins cannot express both distinct Q/K head masks and
the sequence-only trigonometric reads when there are no data-head lanes.
This surface keeps successful execution, access safety, every active even/odd
result and every cell outside the actual four-store footprint together.
-/
namespace VeriTile.Bench.TritonBenchG.RotaryEmb.FullIO
open VeriTile Triton FullMath

structure IO (p : Parameters) where
  kernel : ComputeKernel
  projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel
  Q : RegionName
  K : RegionName
  Cos : RegionName
  Sin : RegionName

abbrev Formula (p : Parameters) := BlockState → Bool → p.Lane → ℝ

/-- Region output equations use independent initial-memory formulas. Both
outputs belong to one final state, and the frame preserves raw typed cells.
-/
def IO.RegionResult {p : Parameters} (io : IO p) (s t : BlockState)
    (f : Formula p × Formula p) : Prop :=
  (∀ odd i, p.activeQ s i → t.readMem io.Q (p.qAddress s odd i) = f.1 s odd i) ∧
  (∀ odd i, p.activeK s i → t.readMem io.K (p.kAddress s odd i) = f.2 s odd i) ∧
  BlockState.WriteWithin (p.footprint s io.Q io.K) s t

/-- The identical formula and footprint at concrete flat addresses. -/
def IO.FlatResult {p : Parameters} (io : IO p) (A : FlatAlloc) (s t : BlockState)
    (f : Formula p × Formula p) : Prop :=
  (∀ odd i, p.activeQ s i →
    t.readMem A.flat (A.addr io.Q (p.qAddress s odd i)) = f.1 s odd i) ∧
  (∀ odd i, p.activeK s i →
    t.readMem A.flat (A.addr io.K (p.kAddress s odd i)) = f.2 s odd i) ∧
  BlockState.WriteWithin
    (SoftmaxFlaggems.RealFlatFrame.footprint A (p.footprint s io.Q io.K))
    (A.flattenState s) t

/-- Source-specific IO contract. Region execution permits arbitrary undefined
values and registers. The current flattening bridge's zero-undef restriction
is confined to its flat-memory clause. Geometry is required only for active
stores; inactive lanes and cosine/sine buffers may alias.
-/
def IO.Implements {p : Parameters} (io : IO p) (f : Formula p × Formula p) : Prop :=
  ∀ (A : FlatAlloc), A.Disjoint →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ s : BlockState,
    RegionsSeparate io.Q io.K io.Cos io.Sin → p.Geometry s →
    p.Bounds s A.extent io.Q io.K io.Cos io.Sin →
    io.Q ∈ A.regions → io.K ∈ A.regions →
    io.kernel.eraseDType.toAlgKernel.TraceSafe A.extent s ∧
    (∃ t, exec io.kernel.eraseDType.toAlgKernel s = some t ∧ io.RegionResult s t f) ∧
    (s.undef = (fun _ _ ↦ 0) →
      ∃ t, exec (A.flattenKernel io.kernel.eraseDType.toAlgKernel)
        (A.flattenState s) = some t ∧ io.FlatResult A s t f)

scoped infix:25 " ⊨ " => IO.Implements

def sourceIO (p : Parameters) (q k cos sin : FullSource.Storage) (v : Bool)
    (Q K Cos Sin : RegionName) : IO p where
  kernel := FullSource.kernel q k cos sin v Q K Cos Sin p.qBatch p.qHead p.qDim
    p.kBatch p.kHead p.kDim p.cosBatch p.cosDim p.sinBatch p.sinDim p.length
    p.headsQ p.headsK p.blockHead p.blockSeq p.dimension
  projection := FullSource.kernel_supported q k cos sin v p Q K Cos Sin
  Q := Q
  K := K
  Cos := Cos
  Sin := Sin

/-- The paired formula mentions only the initial state and source addresses. -/
noncomputable def expected (p : Parameters) (Q K Cos Sin : RegionName) :
    Formula p × Formula p :=
  (fun s => p.expectedQ s Q Cos Sin, fun s => p.expectedK s K Cos Sin)

theorem source_implements (p : Parameters) (q k cos sin : FullSource.Storage)
    (v : Bool) (Q K Cos Sin : RegionName) :
    (sourceIO p q k cos sin v Q K Cos Sin).Implements (expected p Q K Cos Sin) := by
  intro A hd hcov s hsep hGeometry hb hQr hKr
  refine ⟨typed_safe q k cos sin v A.extent p Q K Cos Sin s hb, ?_, ?_⟩
  · exact typed_io q k cos sin v p Q K Cos Sin s hsep hGeometry
  · intro hu
    exact typed_flat_io q k cos sin v A hd hcov p Q K Cos Sin s
      hsep hGeometry hb hQr hKr hu

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb.FullIO
