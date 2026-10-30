import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Targets

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Native
open VeriTile.Bench.Repairs.MatmulPersistentTriton Floating
noncomputable section
set_option maxHeartbeats 7000000

def allWords : Domain where
  cast := fun _ _ => True
  instruction := fun _ _ _ _ => True

/-- One named fixed-size MMA, with its actual two operands and incoming FP32
accumulator. It is strictly smaller than the 128-row source dot. -/
def stage {width : Nat} (layout : DotLayout width) (ops : Numerics)
    (a : Fin 128 → Fin width → Word) (b : Fin width → Fin 256 → Word)
    (acc : Source.Acc Word) (chunk : Fin 4) : Source.Acc Word :=
  fun row column => ops.instruction layout.operation
    (fun i k => a (fragmentRow layout.operation row i) (layout.fragmentK chunk k))
    (fun k j => b (layout.fragmentK chunk k) (fragmentColumn layout.operation column j))
    (fun i j => acc (fragmentRow layout.operation row i) (fragmentColumn layout.operation column j))
    (localRow layout.operation row) (localColumn layout.operation column)

def dot {width : Nat} (layout : DotLayout width) (ops : Numerics)
    (a : Fin 128 → Fin width → Word) (b : Fin width → Fin 256 → Word)
    (acc : Source.Acc Word) : Source.Acc Word :=
  (List.ofFn layout.order).foldl (stage layout ops a b) acc

/-- Every source K block expands into four ordered small ISA applications.
The unused-width case is concrete and unreachable at the quoted call site. -/
def arithmetic (profile : Storage) (layout : DotLayout profile.blockK) (ops : Numerics) :
    Source.Arithmetic Word Word where
  inputZero := 0
  accumulatorZero := 0
  cast := ops.cast
  dot width a b acc :=
    if hw : width=profile.blockK then
      dot (hw.symm ▸ layout) ops a b acc
    else acc

theorem stage_equivalent {width : Nat} (layout : DotLayout width)
    (native reference : Numerics) (ports : LocalPorts native reference allWords)
    (a : Fin 128 → Fin width → Word) (b : Fin width → Fin 256 → Word)
    (acc : Source.Acc Word) (chunk : Fin 4) :
    stage layout native a b acc chunk=stage layout reference a b acc chunk := by
  funext r c
  exact congrFun (congrFun (ports.instruction layout.operation _ _ _ trivial)
    (localRow layout.operation r)) (localColumn layout.operation c)

theorem arithmetic_equivalent (profile : Storage) (layout : DotLayout profile.blockK)
    (native reference : Numerics) (ports : LocalPorts native reference allWords) :
    arithmetic profile layout native=arithmetic profile layout reference := by
  have casts : native.cast=reference.cast := by
    funext p x
    exact ports.cast p x trivial
  have dots : ∀ width (selected : DotLayout width) a b acc,
      dot selected native a b acc=dot selected reference a b acc := by
    intro width selected a b acc
    unfold dot
    have hstage : stage selected native a b=stage selected reference a b := by
      funext old chunk
      exact stage_equivalent selected native reference ports a b old chunk
    rw [hstage]
  unfold arithmetic
  congr 1
  · funext width a b acc
    split_ifs with h
    · exact dots width _ a b acc
    · rfl

theorem target_equivalent (p : Problem) (b : Buffers) (entry : Memory Word)
    (layout : DotLayout p.profile.blockK) (native reference : Numerics)
    (ports : LocalPorts native reference allWords) (row column : Nat) :
    Targets.result (arithmetic p.profile layout native) p b entry row column=
      Targets.result (arithmetic p.profile layout reference) p b entry row column := by
  rw [arithmetic_equivalent p.profile layout native reference ports]

/-- The independent scalar conversion preserves the literal positive zero.
This is one scalar conversion law, and does not constrain any matrix output. -/
structure Reference where
  numerics : Numerics
  positiveZero : ∀ profile,numerics.cast profile 0=0

theorem padding_zero (profile : Storage) (layout : DotLayout profile.blockK)
    (native : Numerics) (reference : Reference)
    (ports : LocalPorts native reference.numerics allWords) :
    (arithmetic profile layout native).cast profile
      (arithmetic profile layout native).accumulatorZero=
        (arithmetic profile layout native).inputZero := by
  change native.cast profile 0=0
  rw [ports.cast profile 0 trivial,reference.positiveZero]

#axiomsClean padding_zero
#axiomsClean stage_equivalent
#axiomsClean arithmetic_equivalent
#axiomsClean target_equivalent
#specNonCircular LocalPorts avoiding [Targets.result,Source.run,Source.accumulator]
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Native
