import bench.repairs.tritonbench_g.token_attn_reduceV_grid_y_checked.Proofs.LogicalValues

/-! Every old arbitrary typed heap has a logical IO representation. In
particular, native non-FP32 cells retain the source's actual default-read term;
there is no hidden requirement that all floating inputs be word leaves. -/
namespace VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked.Interface
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenAttnReduceV
noncomputable section
variable {A : Type} [Inhabited A]

def heapCounts (p : Parameters) (B : Nat) (m : Grid.Memory A) (b : Fin B) : Nat :=
  (m (Region.cast p.length) (b.val*p.lengthStride)).read .nat

def fromHeap (p : Parameters) (B H : Nat) (counts : Fin B → Nat) (m : Grid.Memory A) : Inputs A p B H counts :=
  let starts := fun b => (m (Region.cast p.start) (b*p.startStride)).read .nat
  let requests := fun b => (m (Region.cast p.request) (b*p.requestStride)).read .nat
  let tokens := fun b n => (m (Region.cast p.tokens) (requests b*p.requestBatchStride+n*p.requestTokenStride)).read .nat
  fun q => match q with
  | ⟨0,_⟩ => fun j => (m (Region.cast p.length) (j.1.val*p.lengthStride)).read .nat
  | ⟨1,_⟩ => fun j => starts j.1.val
  | ⟨2,_⟩ => fun j => requests j.1.val
  | ⟨3,_⟩ => fun j => tokens j.1.val j.2.1.val
  | ⟨4,_⟩ => fun j => (m p.prob (j.2.1.val*p.probHeadStride+(starts j.1.val+j.2.2.1.val)*p.probTokenStride)).read .fp32
  | ⟨5,_⟩ => fun j => (m p.value (j.2.1.val/p.kvGroup*p.valueHeadStride+
      j.2.2.2.1.val*p.valueDimStride+tokens j.1.val j.2.2.1.val*p.valueBatchStride)).read .fp32
  | ⟨n+6,h⟩ => False.elim (by simp [inputs] at h)

theorem heap_domain (p : Parameters) (B H : Nat) (m : Grid.Memory A) :
    LengthsMatch p B H (heapCounts p B m) (fromHeap p B H (heapCounts p B m) m) := by
  intro b _
  rfl

theorem heap_binding (M : Algebra A) (R : SumPrimitive A) (p : Parameters)
    (B H : Nat) (counts : Fin B → Nat) (jobs : List Grid.PID) (s : State A) :
    (programIO M R p B H counts jobs).InputBinding (fromHeap p B H counts s.mem) s := by
  intro q j _
  fin_cases q <;> rfl

theorem heap_formula (M : Algebra A) (R : SumPrimitive A) (p : Parameters)
    (B H : Nat) (m : Grid.Memory A) (b : Fin B) (h : Fin H) (d : Fin p.width) :
    orderedTarget M R p B H (heapCounts p B m) (fromHeap p B H (heapCounts p B m) m)
      ⟨0,by simp [outputs]⟩ (b,h,d,PUnit.unit) =
      Grid.values M R p m ((b.val,h.val),(d,PUnit.unit)) := by
  apply Eq.symm
  apply source_values M R p B H (heapCounts p B m) _ _ b h
  apply call_inputs M R p B H (heapCounts p B m) [] _ (heap_domain p B H m)
    (Grid.state m (b.val,h.val)) (heap_binding M R p B H (heapCounts p B m) [] (Grid.state m (b.val,h.val))) b h rfl rfl

#axiomsClean heap_domain
#axiomsClean heap_binding
#axiomsClean heap_formula
end
end VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked.Interface
