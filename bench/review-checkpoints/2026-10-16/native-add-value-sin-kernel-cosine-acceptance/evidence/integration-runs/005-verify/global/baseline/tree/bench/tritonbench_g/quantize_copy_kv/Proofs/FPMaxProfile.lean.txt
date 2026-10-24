import bench.tritonbench_g.quantize_kv_transform.Proofs.MixedBits

/-! Task-local maximum profile. Triton 3.8 standard.max promotes half inputs
to fp32 and reduces with core.maximum's default PropagateNan.NONE/maxnumf.
Consequently a NaN must not dominate a numeric operand. The exact reduction
schedule, all-NaN payload and zero tie lowering remain native bindings.
The earlier rank-based profile is not silently reused on mixed-NaN inputs. -/
namespace VeriTile.Bench.TritonBenchG.QuantizeCopyKv.FPProfile
open VeriTile.Triton FP.Structural
open QuantizeKvTransform
abbrev Word := FP.Value .fp32

def maxnum (cfg : FP.Config) (x y : Word) : Word :=
  let x := FP.prepare cfg x
  let y := FP.prepare cfg y
  if x.isNaN then (if y.isNaN then FP.Value.qNaN .fp32 else y)
  else if y.isNaN then x
  else if x.isZero && y.isZero then FP.Value.zero .fp32 (x.sign && y.sign)
  else if FP.lt cfg y x then x else y

/-- Explicit ascending software reduction. A nonempty row starts with its
first actual value, not an artificial numeric seed that would erase all-NaN
rows. This is not a claim about the GPU reduction tree. -/
def maximum (cfg : FP.Config) {n : Nat} (x : Fin n → Word) : Word :=
  if hn : 0 < n then
    ((List.finRange n).drop 1).foldl (fun v i => maxnum cfg v (x i)) (x ⟨0, hn⟩)
  else FP.Value.infinity .fp32 Bool.true

def reduceMaximum (cfg : FP.Config) {shape : TileShape} (ax : Fin shape.length) :
    (keepDims : Bool) → Values Word .real shape →
      Values Word .real (TileShape.reduceShape shape ax keepDims)
  | Bool.false, x => fun out => maximum cfg (fun i => x (TileShape.insertAxisIndex shape ax out i))
  | Bool.true, x => fun out => maximum cfg (fun i => x (TileShape.replaceAxisIndex shape ax out i))

noncomputable def backend (rest : Algebra Word) (cfg : FP.Config)
    (undefinedValue : Nat → Word → Int) : Algebra Word :=
  { MixedBits.backend rest cfg undefinedValue with
    reduceMax := fun _ => reduceMaximum cfg }

theorem maxnum_boundary_vectors :
    maxnum .ieeeValue (FP.Value.ofBits .fp32 0x3f800000) (FP.Value.qNaN .fp32) =
      FP.Value.ofBits .fp32 0x3f800000 ∧
    maxnum .ieeeValue (FP.Value.qNaN .fp32) (FP.Value.ofBits .fp32 0x3f800000) =
      FP.Value.ofBits .fp32 0x3f800000 ∧
    (maxnum .ieeeValue (FP.Value.qNaN .fp32) (FP.Value.qNaN .fp32)).isNaN = Bool.true ∧
    (maxnum .ieeeValue (FP.Value.zero .fp32 Bool.true) (FP.Value.zero .fp32)).bits = 0 ∧
    (maxnum .ieeeValue (FP.Value.zero .fp32 Bool.true) (FP.Value.zero .fp32 Bool.true)).bits = 0x80000000 := by
  decide +kernel

/-- A checked counterexample to treating the old encoded-rank reducer as
the default maxnum reducer on arbitrary inputs. -/
theorem rank_maximum_not_maxnum :
    let xs : Fin 2 → Word := fun i =>
      if i.val = 0 then FP.Value.ofBits .fp32 0x3f800000 else FP.Value.qNaN .fp32
    QuantizeKvCopy.TypedQuantize.maximum xs ≠ maximum .ieeeValue xs := by
  decide +kernel

#axiomsClean maxnum_boundary_vectors
#axiomsClean rank_maximum_not_maxnum
end VeriTile.Bench.TritonBenchG.QuantizeCopyKv.FPProfile
