import bench.repairs.tritonbench_g.flash_decode2_llama.Proofs.PublicIO
import bench.repairs.tritonbench_g.flash_decode2_llama.Proofs.RawNative

namespace VeriTile.Bench.Repairs.FlashDecode2Llama.NativeIO
open VeriTile.Triton FP.Structural FlashDecode2 KernelInterface Native
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def target (p : RawNative.Ops) (g : KernelInterface.Geometry)
    (vals : IO.Inputs Word g) (row dim : Nat) : Word :=
  let a := RawNative.scan p g (IO.logValues vals row) (IO.midValues vals row)
    ((IO.lengths vals (row / g.H) + g.Q - 1) / g.Q)
  p.div (a.numerator dim) a.denominator

def mathematical (p : RawNative.Ops) (g : KernelInterface.Geometry) (vals : IO.Inputs Word g) :=
  IO.outputValues (g := g) (fun row dim => target p g vals row.val dim.val)

def observe (p : RawNative.Ops) (g : KernelInterface.Geometry) : Term → Word :=
  Native.eval (RawNative.lower p) (RawNative.schedule g)

theorem eval_bitsLoad (p : RawNative.Ops) (g : KernelInterface.Geometry) (word : Word) :
    Native.eval (RawNative.lower p) (RawNative.schedule g) (.load (.bits word)) = word := rfl

theorem eval_div (p : RawNative.Ops) (g : KernelInterface.Geometry) (a b : Term) :
    Native.eval (RawNative.lower p) (RawNative.schedule g)
      (.binary (some .fp32) .real .div a b) =
        p.div (Native.eval (RawNative.lower p) (RawNative.schedule g) a)
          (Native.eval (RawNative.lower p) (RawNative.schedule g) b) := by
  simp [Native.eval, Native.binary, RawNative.schedule, RawNative.lower,
    KernelNative.lower, Numerical.lower]

/-- Reuse the native instruction recurrence for arbitrary encoded input words.
The induction consumes public input pins only at actual in-bounds loads. -/
theorem recurrence (p : RawNative.Ops) (g : KernelInterface.Geometry) (order : List Nat)
    (vals : IO.Inputs Word g) (s : State Term)
    (pins : (IO.decodeIO algebra g order Term.bits (observe p g)).InputBinding vals s)
    (row : Fin (g.B * g.H)) (n : Nat) (hn : n ≤ g.S) :
    let entry := Grid.setup g.B g.H row.val s
    let a := RawNative.scan p g (IO.logValues vals row.val) (IO.midValues vals row.val) n
    Native.eval (RawNative.lower p) (RawNative.schedule g)
      (values algebra entry Bool.false g.H g.S g.D g.D n).m = a.maximum ∧
    Native.eval (RawNative.lower p) (RawNative.schedule g)
      (values algebra entry Bool.false g.H g.S g.D g.D n).z = a.denominator ∧
    ∀ k, Native.eval (RawNative.lower p) (RawNative.schedule g)
      ((values algebra entry Bool.false g.H g.S g.D g.D n).a k) = a.numerator (lane k) := by
  let entry := Grid.setup g.B g.H row.val s
  have hx (j : Nat) (hj : j < g.S) : logvalue algebra entry g.H g.S j =
      .load (.bits (IO.logValues vals row.val j)) := by
    simp only [logvalue, entry, row_log_offset]
    change algebra.fp32Load ((s.mem "E" (row.val * g.S + j)).read .real) = _
    rw [PublicIO.log_binding algebra g order Term.bits (observe p g) vals s pins row.val j row.isLt hj]
    rfl
  have hv (j : Nat) (hj : j < g.S) (k : TileIndex [g.D]) :
      FlashDecode2.value algebra entry Bool.false g.H g.S g.D j k =
        .load (.bits (IO.midValues vals row.val j (lane k))) := by
    simp only [FlashDecode2.value, active, true_or, ↓reduceIte, entry, row_mid_offset]
    change algebra.fp32Load ((s.mem "M" (row.val * g.S * g.D + j * g.D + lane k)).read .real) = _
    rw [PublicIO.mid_binding algebra g order Term.bits (observe p g) vals s pins
      row.val j (lane k) row.isLt hj k.1.isLt]
    rfl
  simp only [entry] at hx hv
  induction n with
  | zero =>
    simp [values, FlashDecode2.initial, zero, algebra, Native.eval, RawNative.lower,
      KernelNative.lower, Numerical.lower, RawNative.scan, RawNative.initial]
  | succ n ih =>
    have hsplit : n < g.S := by omega
    have ih := ih (by omega)
    dsimp only at ih ⊢
    have shapes := RawNative.carry_shape (Grid.setup g.B g.H row.val s)
      Bool.false g.H g.S g.D g.D n
    simp only [values, next, hx n hsplit, hv n hsplit, RawNative.max_term,
      RawNative.exp_term, RawNative.sub_term, RawNative.mul_term, RawNative.add_term]
    simp only [RawNative.eval_max, RawNative.eval_sub, RawNative.eval_exp,
      RawNative.eval_acc, RawNative.eval_fresh_mul, shapes.1, shapes.2,
      eval_bitsLoad, RawNative.scan, RawNative.merge, ih.1, ih.2.1, ih.2.2]
    simp

theorem value (p : RawNative.Ops) (g : KernelInterface.Geometry) (order : List Nat)
    (vals : IO.Inputs Word g) (domain : IO.Lengths g vals) (s : State Term)
    (pins : (IO.decodeIO algebra g order Term.bits (observe p g)).InputBinding vals s)
    (row : Fin (g.B * g.H)) (dim : Fin g.D) :
    observe p g (Grid.value algebra s Bool.false g.B g.H g.S g.D g.Q g.D row.val
      (dim, PUnit.unit)) = target p g vals row.val dim.val := by
  have bounds := PublicIO.lengths_bound algebra g order Term.bits (observe p g) vals domain s pins
  have counted := KernelInterface.count_bound g s bounds row.val row.isLt
  have recur := recurrence p g order vals s pins row (steps g s row.val) counted.2.2
  have length := PublicIO.length_binding algebra g order Term.bits (observe p g) vals s pins
    (row.val / g.H) (PublicIO.batch_bound g row.val row.isLt)
  have countEq : steps g s row.val = (IO.lengths vals (row.val / g.H) + g.Q - 1) / g.Q := by
    simp only [steps, length]
  simp only [observe, Grid.value, output, counted.1, divide, algebra]
  rw [eval_div]
  unfold target
  rw [← countEq]
  exact congrArg₂ p.div (recur.2.2 (dim, PUnit.unit)) recur.2.1

theorem implements (p : RawNative.Ops) (g : KernelInterface.Geometry) (order : List Nat)
    (coverage : order.Perm (List.range (g.B * g.H))) :
    (IO.decodeIO algebra g order Term.bits (observe p g)).ImplementsOn (IO.Lengths g)
      (mathematical p g) :=
  PublicIO.implements algebra g order coverage Term.bits (observe p g) _ (value p g order)

#specNonCircular target avoiding [Grid.source, FP.Structural.exec, values, Native.eval]
#axiomsClean implements
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FlashDecode2Llama.NativeIO
