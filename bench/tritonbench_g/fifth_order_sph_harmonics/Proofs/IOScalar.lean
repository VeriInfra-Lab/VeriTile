import bench.tritonbench_g.fifth_order_sph_harmonics.IO

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.IOScalar
open VeriTile.Triton FP.Structural HarmonicIO KernelScalar
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000

theorem real_forward {rows : Nat} (xs : TileIndex [rows, 3] → ℝ)
    (i : TileIndex [rows, 11]) :
    Reference.forwardValues realBackend (realBackend.fp32Load (xs (i.1, 0, ())))
      (realBackend.fp32Load (xs (i.1, 1, ()))) (realBackend.fp32Load (xs (i.1, 2, ()))) i.2.1 =
      HarmonicIO.realForward xs i :=
  ScalarReal.forward (fallback 0) _ _ _ _

theorem real_backward (p : Parameters)
    (vals : (backwardIO realBackend p).Inputs) (zero : prepared p vals)
    (i : TileIndex [p.rows, 3]) :
    Reference.backwardValues realBackend
      (realBackend.fp32Load (vals (0 : Fin 3) (i.1, 0, ())))
      (realBackend.fp32Load (vals (0 : Fin 3) (i.1, 1, ())))
      (realBackend.fp32Load (vals (0 : Fin 3) (i.1, 2, ())))
      (fun c => realBackend.fp32Load (vals (1 : Fin 3) (i.1, c, ())))
      (fun c => realBackend.fp32Load (vals (2 : Fin 3) (i.1, c, ()))) i.2.1 =
      HarmonicIO.realBackward (vals (0 : Fin 3)) (vals (1 : Fin 3)) i := by
  have preparedValues : (fun c : Fin 3 => realBackend.fp32Load (vals (2 : Fin 3) (i.1, c, ()))) =
      (fun _ => (0 : ℝ)) := by funext c; exact zero (i.1, c, ())
  rw [preparedValues]
  exact ScalarReal.backward (fallback 0) _ _ _ _ _

theorem native_forward {rows : Nat} (xs : TileIndex [rows, 3] → Packing.Word)
    (i : TileIndex [rows, 11]) :
    Reference.forwardValues (forwardBackend selectedProfile)
      ((forwardBackend selectedProfile).fp32Load (xs (i.1, 0, ())))
      ((forwardBackend selectedProfile).fp32Load (xs (i.1, 1, ())))
      ((forwardBackend selectedProfile).fp32Load (xs (i.1, 2, ()))) i.2.1 =
      nativeForward xs i := by
  let inputs : Nat → Packing.Word := fun k =>
    if k = 0 then xs (i.1, 0, ()) else if k = 1 then xs (i.1, 1, ()) else xs (i.1, 2, ())
  have h := FPProof.scalar_forward (forwardBackend selectedProfile) wordBase valueBase selectedProfile
    inputs i.2.1 (forward_laws selectedProfile inputs i.2.1)
  simpa only [inputs, ↓reduceIte, nativeForward] using h

theorem native_backward {rows : Nat} (xs : TileIndex [rows, 3] → Packing.Word)
    (gs : TileIndex [rows, 11] → Packing.Word) (old : TileIndex [rows, 3] → Packing.Word)
    (i : TileIndex [rows, 3]) :
    Reference.backwardValues (backwardBackend selectedProfile)
      ((backwardBackend selectedProfile).fp32Load (xs (i.1, 0, ())))
      ((backwardBackend selectedProfile).fp32Load (xs (i.1, 1, ())))
      ((backwardBackend selectedProfile).fp32Load (xs (i.1, 2, ())))
      (fun c => (backwardBackend selectedProfile).fp32Load (gs (i.1, c, ())))
      (fun c => (backwardBackend selectedProfile).fp32Load (old (i.1, c, ()))) i.2.1 =
      nativeBackward xs gs old i := by
  let inputs : Nat → Packing.Word := fun k =>
    if h : k < 3 then xs (i.1, ⟨k, h⟩, ())
    else if h : k < 14 then gs (i.1, ⟨k-3, by omega⟩, ())
    else if h : k < 17 then old (i.1, ⟨k-14, by omega⟩, ())
    else xs (i.1, 0, ())
  have up (c : Fin 11) : inputs (3+c.val) = gs (i.1, c, ()) := by
    have hc := c.isLt
    simp only [inputs, show ¬3+c.val < 3 from by omega,
      show 3+c.val < 14 from by omega, ↓reduceDIte]
    simp only [Nat.add_sub_cancel_left]
  have previous (c : Fin 3) : inputs (14+c.val) = old (i.1, c, ()) := by
    have hc := c.isLt
    simp only [inputs, show ¬14+c.val < 3 from by omega,
      show ¬14+c.val < 14 from by omega, show 14+c.val < 17 from by omega, ↓reduceDIte]
    simp only [Nat.add_sub_cancel_left]
  have h := FPProof.scalar_backward (backwardBackend selectedProfile) wordBase valueBase selectedProfile
    inputs i.2.1 (backward_laws selectedProfile inputs i.2.1)
  simpa only [up, previous, inputs, show (0 : Nat) < 3 from by decide,
    show (1 : Nat) < 3 from by decide, show (2 : Nat) < 3 from by decide, ↓reduceDIte,
    nativeBackward] using h

#axiomsClean real_forward
#axiomsClean real_backward
#axiomsClean native_forward
#axiomsClean native_backward
end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.IOScalar
