import bench.repairs.tritonbench_g.matmul_dequant_int4.Proofs.Strides
import bench.repairs.tritonbench_g.matmul_dequant_int4.Proofs.Memory
namespace VeriTile.Bench.Repairs.MatmulDequantInt4.SourceExecution
set_option maxHeartbeats 4000000

/-- Literal standalone source scalar, retaining actual runtime Tensor strides.
The work list is the masked source grid enumeration, already proved in Grid. -/
def value (s : Shape) (strides : Strides.Packed) (p : DequantOps α) (current : State α) (w : Work) :=
  p.mul (p.fromInt
    (nibble (current.packed .weight (Strides.qAddress strides w.row w.col)) w.row-
     nibble (current.packed .zero (Strides.zAddress strides s w.row w.col)) w.col))
    (current.values .scale (Strides.sAddress strides s w.row w.col))
def run (s : Shape) (strides : Strides.Packed) (p : DequantOps α) (order : List Work) (current : State α) :=
  Writes.run .dequant s.n (value s strides p) order current
theorem value_eq (s : Shape) (hs : s.Valid) (strides : Strides.Packed) (admitted : strides.Admitted s)
    (p : DequantOps α) (current : State α) (w : Work) (hw : w.Active s.k s.n) :
    value s strides p current w=dequantValue s p current w := by
  have h := Strides.active_packed strides s hs admitted w.row w.col hw.1 hw.2
  simp only [value,dequantValue,difference,h.1,h.2.1,h.2.2]
theorem run_eq (s : Shape) (hs : s.Valid) (strides : Strides.Packed) (admitted : strides.Admitted s)
    (p : DequantOps α) (order : List Work) (active : ∀ w∈order,w.Active s.k s.n) (current : State α) :
    run s strides p order current=dequantRun s p order current := by
  induction order generalizing current with
  | nil => rfl
  | cons w ws ih =>
    change run s strides p ws (store current .dequant (w.row*s.n+w.col) (value s strides p current w))=_
    rw [value_eq s hs strides admitted p current w (active w (by simp)),
      ih (fun v hv => active v (by simp [hv]))]
    rfl
theorem complete_run_eq (s : Shape) (hs : s.Valid) (strides : Strides.Packed) (admitted : strides.Admitted s)
    (p : DequantOps α) (g : Grid s.k s.n) (order : List Work) (complete : g.Complete order) (current : State α) :
    run s strides p order current=dequantRun s p order current :=
  run_eq s hs strides admitted p order (fun w hw => (g.complete_mem order complete w).mp hw) current

#axiomsClean value_eq
#axiomsClean run_eq
#axiomsClean complete_run_eq
end VeriTile.Bench.Repairs.MatmulDequantInt4.SourceExecution
