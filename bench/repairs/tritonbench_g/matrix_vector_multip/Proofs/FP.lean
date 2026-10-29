import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.Host
import bench.tritonbench_g.matrix_vector_multip.Proofs.WholeNativePublic
import bench.tritonbench_g.matrix_vector_multip.Proofs.WholeNativeChoices

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.FP
open VeriTile.Triton VeriTile.Bench.TritonBenchG.MatrixVectorMultip
open WholeHost WholeSource WholeNative
noncomputable section
set_option maxHeartbeats 2000000

/-- Native FMA operands at logical tensor coordinates, including zero-filled
masked lanes. No separate mul/add is identified with FMA by an FP law. -/
def next (B : Scalars) (x : Inputs) (c : Config) (g : Graph c.blockRows c.blockColumns)
    (matrix : Nat → Nat → BitVec 32) (vector : Nat → BitVec 32)
    (pid off : Nat) (acc : Acc (BitVec 32) (x.parameters c)) :
    Acc (BitVec 32) (x.parameters c) := fun i =>
  let a := if pid*c.blockRows+i.1.val < x.rows ∧ off+i.2.1.val < x.columns then
    matrix (pid*c.blockRows+i.1.val) (off+i.2.1.val) else 0
  let b := if off+i.2.1.val < x.columns then vector (off+i.2.1.val) else 0
  if g.reverseFMA i.1 i.2.1 then B.fma b a (acc i) else B.fma a b (acc i)

def accumulate (B : Scalars) (x : Inputs) (c : Config) (g : Graph c.blockRows c.blockColumns)
    (matrix : Nat → Nat → BitVec 32) (vector : Nat → BitVec 32)
    (pid off : Nat) (acc : Acc (BitVec 32) (x.parameters c)) :
    Acc (BitVec 32) (x.parameters c) :=
  if h : off < x.columns ∧ 0 < c.blockColumns then
    accumulate B x c g matrix vector pid (off+c.blockColumns)
      (next B x c g matrix vector pid off acc)
  else acc
termination_by x.columns-off

/-- Independently evaluated ordered FP32 graph on logical inputs. -/
def formula (B : Scalars) (graphs : (c : Config) → Nat → Graph c.blockRows c.blockColumns)
    (c : Config) (x : Inputs) (matrix : Nat → Nat → BitVec 32)
    (vector : Nat → BitVec 32) (i : Nat) : BitVec 32 :=
  let g := graphs c (i/c.blockRows)
  (g.tree ⟨i%c.blockRows, Nat.mod_lt _ c.rows_pos⟩).eval B.add (fun j =>
    accumulate B x c g matrix vector (i/c.blockRows) 0 (fun _ => 0)
      (⟨i%c.blockRows, Nat.mod_lt _ c.rows_pos⟩, j, PUnit.unit))

variable {x : Inputs} {before : Nat → BitVec 32} {old : Nat → Prop}
    {matrix : Nat → Nat → BitVec 32} {vector : Nat → BitVec 32}

theorem matrixLoad_bound (r : DirectBinding (BitVec 32) x before matrix vector)
    (c : Config) (pid off : Nat) (i : Fin c.blockRows) (j : Fin c.blockColumns) :
    WholeNative.matrixLoad (x.parameters c) pid off
      (WholeNativeHost.entryMemory x before) i j =
      if pid*c.blockRows+i.val < x.rows ∧ off+j.val < x.columns then
        matrix (pid*c.blockRows+i.val) (off+j.val) else 0 := by
  unfold WholeNative.matrixLoad
  split
  · rename_i active
    change pid*c.blockRows+i.val < x.rows ∧ off+j.val < x.columns at active
    have addr : x.matrixBase+aOffset (x.parameters c) pid off i.val j.val =
        x.matrixBase+(pid*c.blockRows+i.val)*x.strideRow+(off+j.val)*x.strideColumn := by
      unfold aOffset row Inputs.parameters
      ring
    change before (x.matrixBase+aOffset (x.parameters c) pid off i.val j.val) = _
    have load := r.matrixCoordinate (pid*c.blockRows+i.val) (off+j.val) active.1 active.2
    simp only [Inputs.matrixAddress] at load
    rw [addr, load]
    simp only [if_pos active]
  · rename_i inactive
    exact (if_neg inactive).symm

theorem vectorLoad_bound (r : DirectBinding (BitVec 32) x before matrix vector)
    (compatible : x.columns = x.vectorLength) (c : Config) (off : Nat) (j : Fin c.blockColumns) :
    WholeNative.vectorLoad (x.parameters c) off
      (WholeNativeHost.entryMemory x before) j =
      if off+j.val < x.columns then vector (off+j.val) else 0 := by
  unfold WholeNative.vectorLoad
  split
  · rename_i active
    change off+j.val < x.columns at active
    have addr : x.vectorBase+bOffset (x.parameters c) off j.val =
        x.vectorBase+(off+j.val)*x.strideVector := by
      unfold bOffset Inputs.parameters
      ring
    change before (x.vectorBase+bOffset (x.parameters c) off j.val) = _
    have load := r.vectorCoordinate (off+j.val) (by simpa only [←compatible] using active)
    simp only [Inputs.vectorAddress] at load
    rw [addr, load]
    simp only [if_pos active]
  · rename_i inactive
    exact (if_neg inactive).symm

theorem next_bound (r : DirectBinding (BitVec 32) x before matrix vector)
    (compatible : x.columns = x.vectorLength) (B : Scalars) (c : Config)
    (g : Graph c.blockRows c.blockColumns) (pid off : Nat)
    (acc : Acc (BitVec 32) (x.parameters c)) :
    WholeNative.next B (x.parameters c) g pid off
      (WholeNativeHost.entryMemory x before) acc = next B x c g matrix vector pid off acc := by
  funext i
  simp only [WholeNative.next, matrixLoad_bound r, vectorLoad_bound r compatible, next]

theorem accumulate_bound (r : DirectBinding (BitVec 32) x before matrix vector)
    (compatible : x.columns = x.vectorLength) (B : Scalars) (c : Config)
    (g : Graph c.blockRows c.blockColumns) (pid off : Nat)
    (acc : Acc (BitVec 32) (x.parameters c)) :
    WholeNative.accumulate B (x.parameters c) g pid
      (WholeNativeHost.entryMemory x before) off acc =
      accumulate B x c g matrix vector pid off acc := by
  conv_lhs => rw [WholeNative.accumulate]
  conv_rhs => rw [accumulate]
  simp only [show (x.parameters c).columns = x.columns from rfl,
    show (x.parameters c).blockColumns = c.blockColumns from rfl]
  split
  · rw [next_bound r compatible]
    exact accumulate_bound r compatible B c g pid (off+c.blockColumns) _
  · rfl
termination_by x.columns-off

theorem bound_formula (r : DirectBinding (BitVec 32) x before matrix vector)
    (compatible : x.columns = x.vectorLength) (B : Scalars)
    (graphs : (c : Config) → Nat → Graph c.blockRows c.blockColumns)
    (c : Config) (i : Nat) :
    WholeNativeHost.nativeFormula B graphs c x before i = formula B graphs c x matrix vector i := by
  unfold WholeNativeHost.nativeFormula WholeNative.tileFormula formula
  rw [accumulate_bound r compatible]
  rfl

theorem outcomes (B : Scalars)
    (graphs : (c : Config) → Nat → Graph c.blockRows c.blockColumns)
    (b : DirectBinding (BitVec 32) x before matrix vector)
    (compatible : x.columns = x.vectorLength)
    (rt : WholeHost.Runtime (BitVec 32) x before old) :
    WholePublic.Outcomes (fun a i => formula B graphs (rt.plan a).selected.config
      x matrix vector i.val) rt (WholeNativeHost.execute B graphs rt) := by
  have equal : (fun a i => WholeNativeHost.nativeFormula B graphs
      (rt.plan a).selected.config x before i.val) =
      (fun a (i : Fin x.rows) => formula B graphs (rt.plan a).selected.config
      x matrix vector i.val) := by
    funext a i
    exact bound_formula b compatible B graphs _ i.val
  rw [←equal]
  exact WholePublic.native_outcomes B graphs rt

#axiomsClean outcomes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.FP
