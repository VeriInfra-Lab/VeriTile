import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.HostPrepared
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.RealPrepared

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.HostReal
open VeriTile.Triton FP.Structural HostSnapshots HostPrepared
noncomputable section

abbrev Carrier (_ : Host.Role) := ℝ
def input (shape : Shape) (entry : (role : Host.Role) → View Carrier role) :=
  matrix shape.inner (entry .input).values
def weight (shape : Shape) (entry : (role : Host.Role) → View Carrier role) :=
  HostPrepared.weight shape.columns shape.inner (entry .weight).values
def output (shape : Shape) (entry : (role : Host.Role) → View Carrier role) :=
  matrix shape.outputColumns (entry .output).values

def initial (abi : SourceABI) (shape : Shape) (batches : Nat)
    {entry : (role : Host.Role) → View Carrier role} {caller : List Window}
    (p : Prepared entry caller) (lists : Lists p) : State ℝ :=
  PreparedMemory.state abi shape (matrix shape.inner p.x.values)
    (HostPrepared.weight shape.columns shape.inner p.w.values) (matrix shape.outputColumns p.y.values)
    (raw batches lists) (lists.lengths.val.drop batches)

/-- Independent exact Real arithmetic on the actual typed-source evaluator.
This theorem has no native-word backend or numerical-agreement assumption.
It does not assert an unproved bridge to the distinct BlockState interpreter. -/
theorem source_result (rest : Algebra ℝ) (abi : SourceABI) (shape : Shape) (batches : Nat)
    (entry : (role : Host.Role) → View Carrier role) (caller : List Window)
    (domain : Domain abi shape batches entry) (p : Prepared entry caller) (lists : Lists p)
    (segments : List Segment) (valid : validateSegments shape (raw batches lists) = some segments) :
    ∃ final, runSourceJobs (SourceBody.install (RealBackend.algebra rest) RealBackend.dot)
        abi (PreparedMemory.arguments shape (strides p)) (jobs shape.columns segments)
        (initial abi shape batches p lists) = some final ∧
      SourceSuffix.OutputMemory abi (PreparedMemory.arguments shape (strides p)) shape final
        (mathematicalResult shape (input shape entry) (weight shape entry) abi.addInputs segments (output shape entry)) ∧
      SourceGrid.ReadFrame (PreparedMemory.arguments shape (strides p)) (initial abi shape batches p lists) final := by
  obtain ⟨hx,hw,hy⟩ := layouts abi shape batches entry caller domain p
  obtain ⟨final,run,values,frame⟩ := RealBackend.prepared_result rest abi shape (strides p)
    (matrix shape.inner p.x.values) (HostPrepared.weight shape.columns shape.inner p.w.values)
    (matrix shape.outputColumns p.y.values) (raw batches lists) (lists.lengths.val.drop batches)
    segments valid hx hw hy domain.capacity
  refine ⟨final,run,?_,frame⟩
  simpa only [input,weight,output,p.x.preserved,p.w.preserved,p.y.preserved] using values

/-- Ordered logical slice copies from one immutable final private output.
Real contents are logical values and are never injected into finite bytes. -/
def copy (shape : Shape) (final : Nat → Nat → ℝ) (segments : List Segment)
    (entry : Nat → Nat → ℝ) : Nat → Nat → ℝ :=
  segments.foldl (fun previous segment r c =>
    if ActiveCell shape segment r c then final r c else previous r c) entry

theorem copy_value (shape : Shape) (final : Nat → Nat → ℝ) (segments : List Segment)
    (entry : Nat → Nat → ℝ) (r c : Nat) :
    copy shape final segments entry r c =
      if ∃ segment ∈ segments, ActiveCell shape segment r c then final r c else entry r c := by
  induction segments generalizing entry with
  | nil => simp [copy]
  | cons segment segments ih =>
      change copy shape final segments (fun r c => if ActiveCell shape segment r c then final r c else entry r c) r c = _
      rw [ih]
      by_cases ha : ActiveCell shape segment r c <;>
        by_cases ht : ∃ s ∈ segments, ActiveCell shape s r c <;> simp [ha,ht]

theorem caller_result (rest : Algebra ℝ) (abi : SourceABI) (shape : Shape) (batches : Nat)
    (entry : (role : Host.Role) → View Carrier role) (caller : List Window)
    (domain : Domain abi shape batches entry) (p : Prepared entry caller) (lists : Lists p)
    (segments : List Segment) (valid : validateSegments shape (raw batches lists) = some segments) :
    ∃ final, runSourceJobs (SourceBody.install (RealBackend.algebra rest) RealBackend.dot)
        abi (PreparedMemory.arguments shape (strides p)) (jobs shape.columns segments)
        (initial abi shape batches p lists) = some final ∧
      ∀ r c, r < shape.outputRows → c < shape.outputColumns →
        copy shape (SourceSuffix.readOutput abi (PreparedMemory.arguments shape (strides p)) final)
          segments (output shape entry) r c =
        mathematicalResult shape (input shape entry) (weight shape entry) abi.addInputs segments (output shape entry) r c := by
  obtain ⟨final,run,values,_⟩ := source_result rest abi shape batches entry caller domain p lists segments valid
  refine ⟨final,run,?_⟩
  intro r c hr hc
  rw [copy_value]
  split_ifs with active
  · exact values r c hr hc
  · exact (mathematicalResult_frame shape _ _ _ segments _ r c (by simpa using active)).symm

#axiomsClean source_result
#axiomsClean copy_value
#axiomsClean caller_result
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.HostReal
