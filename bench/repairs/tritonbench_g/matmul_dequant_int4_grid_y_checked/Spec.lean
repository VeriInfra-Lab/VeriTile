import bench.repairs.tritonbench_g.matmul_dequant_int4_grid_y_checked.Proofs.Tensor
import bench.repairs.tritonbench_g.matmul_dequant_int4.Proofs.Real

namespace VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.Spec
open VeriTile.Triton FP.Structural Tensor
open MatmulDequantInt4.Direct
noncomputable section
set_option maxHeartbeats 4000000

/-- Independent base-16 definition on the actual input Tensor coordinates. -/
def difference {α : Type} (p : PackedInfo) (o : Info) (t : Arguments p o)
    (heap : Heap α) (k n : Nat) : Int :=
  MatmulDequantInt4.digit (heap.packed t.weight.storageObject (t.weight.physicalIndex (k/8) n)) k -
  MatmulDequantInt4.digit (heap.packed t.zero.storageObject (t.zero.physicalIndex (k/p.group) (n/8))) n

def scale {α : Type} (p : PackedInfo) (o : Info) (t : Arguments p o)
    (heap : Heap α) (k n : Nat) : α :=
  heap.values t.scale.storageObject (t.scale.physicalIndex (k/p.group) n)

def mathematical (p : PackedInfo) (o : Info) (t : Arguments p o)
    (heap : Heap ℝ) (k n : Nat) : ℝ := (difference p o t heap k n : ℝ)*scale p o t heap k n

/-- Formal Real interpretation of precisely the source operations used here. -/
def real : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op x y => match op with | .add => x+y | .sub => x-y | .mul => x*y | _ => 0
  unary := fun _ _ x => x
  cast := fun _ _ _ x => x
  fromNat := fun _ x => x
  fromInt := fun _ x => x
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ _ _ _ _ _ => 0
  reduceSum := fun _ _ _ _ _ _ => 0

/-- Source operation-order target without any reference to source execution. -/
def scalarResult {α : Type} (f : Source.Profile) (M : Algebra α)
    (p : PackedInfo) (o : Info) (t : Arguments p o) (heap : Heap α) (k n : Nat) : α :=
  M.binary none f.dtype .mul
    (M.cast (some .fp32) .real f.dtype (M.fromInt (some .fp32) (difference p o t heap k n)))
    (scale p o t heap k n)

theorem signed_digit (w : BitVec 32) (j : Nat) : Source.nibble w.toInt j=MatmulDequantInt4.digit w j := by
  change VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.nibble w.toInt j=_
  rw [VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.nibble_packed_word]
  exact MatmulDequantInt4.nibble_digit w j

theorem result_eq {α : Type} [Inhabited α] (f : Source.Profile) (M : Algebra α)
    (p : PackedInfo) (o : Info) (t : Arguments p o) (heap : Heap α) (i : Fin 8)
    (x : KernelGrid.Coordinate (params p i)) :
    KernelGrid.result f M (params p i) (entry f p o t heap) x=
      scalarResult f M p o t heap x.1.val x.2.val := by
  cases f <;> simp [KernelGrid.result,scalarResult,params,entry,Source.Profile.dtype,Source.cell,
    difference,scale,t.weight.coordinates,t.zero.coordinates,t.scale.coordinates,
    MatmulDequantInt4.Strides.Layout.address,signed_digit]

theorem real_result (f : Source.Profile) (p : PackedInfo) (o : Info) (t : Arguments p o)
    (heap : Heap ℝ) (k n : Nat) : scalarResult f real p o t heap k n=mathematical p o t heap k n := by
  cases f <;> rfl

theorem difference_bounds {α : Type} (p : PackedInfo) (o : Info) (t : Arguments p o)
    (heap : Heap α) (k n : Nat) : -15≤difference p o t heap k n ∧ difference p o t heap k n≤15 := by
  have hq := Nat.mod_lt ((heap.packed t.weight.storageObject (t.weight.physicalIndex (k/8) n)).toNat /
    2^(4*(k%8))) (by decide : 0<16)
  have hz := Nat.mod_lt ((heap.packed t.zero.storageObject (t.zero.physicalIndex (k/p.group) (n/8))).toNat /
    2^(4*(n%8))) (by decide : 0<16)
  dsimp [difference,MatmulDequantInt4.digit]
  omega

#specNonCircular mathematical avoiding [KernelGrid.run,Source.source,Source.execution,KernelGrid.result]
#specNonCircular scalarResult avoiding [KernelGrid.run,Source.source,Source.execution,KernelGrid.result]
#axiomsClean result_eq
#axiomsClean difference_bounds
end
end VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.Spec
