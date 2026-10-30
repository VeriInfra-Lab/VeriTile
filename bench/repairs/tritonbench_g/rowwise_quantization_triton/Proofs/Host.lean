import VeriTile.Torch.Tensor
import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.Execution
namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton
set_option autoImplicit false
set_option maxHeartbeats 2000000

structure State where
  heap : Heap
  live : Nat → Prop

/-- Ordinary allocator/Tensor operations preserve preexisting live bytes. -/
structure Extension (before after : State) : Prop where
  live : ∀ a, before.live a → after.live a
  frame : ∀ a, before.live a → after.heap a = before.heap a

theorem Extension.trans {a b c : State} (ab : Extension a b) (bc : Extension b c) : Extension a c :=
  ⟨fun p hp => bc.live p (ab.live p hp), fun p hp => (bc.frame p (ab.live p hp)).trans (ab.frame p hp)⟩

/-- Selected public contract: CUDA FP32 rank-two contiguous Tensor, positive
width. `logical` is its ordinary coordinate value interpretation. The actual
wrapper checks rank/dtype/contiguity/width and resolves the negative flag.
Mathematical comparison additionally selects finite values, without changing
the total Word execution statement. -/
structure Input (rows columns : Nat) where
  object : Nat
  logical : Nat → Nat → Word

structure Normalized (rows columns : Nat) (input : Input rows columns) (before : State) where
  after : State
  base : Nat
  extension : Extension before after
  resident : ∀ a, InRange base (4 * (rows * columns)) a → after.live a
  coordinate : ∀ r, r < rows → ∀ c, c < columns →
    load32 after.heap (base + 4 * (r * columns + c)) = input.logical r c

inductive DType where
  | int8 | float16
  deriving DecidableEq

def DType.bytes : DType → Nat
  | .int8 => 1
  | .float16 => 2

structure Tensor where
  device : VeriTile.Torch.Device
  object : Nat
  base : Nat
  dtype : DType
  shape : List Nat

structure Allocation (dtype : DType) (shape : List Nat) (before : State) where
  tensor : Tensor
  after : State
  type_eq : tensor.dtype = dtype
  shape_eq : tensor.shape = shape
  spanFits : tensor.base + dtype.bytes * shape.prod ≤ 2^64
  aligned : tensor.base % dtype.bytes = 0
  extension : Extension before after
  resident : ∀ a, InRange tensor.base (dtype.bytes * shape.prod) a → after.live a
  fresh : ∀ a, InRange tensor.base (dtype.bytes * shape.prod) a → ¬ before.live a

/-- The only external services are coordinate-preserving resolve_neg and
fresh allocation. `none` is an API failure; no universal success is assumed.
No service receives or supplies a quantized value or row maximum. -/
structure Services where
  normalize : ∀ rows columns (input : Input rows columns) (before : State),
    Option (Normalized rows columns input before)
  allocate : ∀ (dtype : DType) (shape : List Nat) (before : State), Option (Allocation dtype shape before)

structure Prepared (p : Launch) (input : Input p.rows p.columns) (before : State) where
  normalized : Normalized p.rows p.columns input before
  quantized : Allocation .int8 [p.rows, p.columns] normalized.after
  maxima : Allocation .float16 [p.rows] quantized.after

def Prepared.layout {p : Launch} {input : Input p.rows p.columns} {before : State}
    (prepared : Prepared p input before) : Layout :=
  ⟨prepared.normalized.base, prepared.quantized.tensor.base, prepared.maxima.tensor.base⟩

def prepare (api : Services) (p : Launch) (input : Input p.rows p.columns) (before : State) :
    Option (Prepared p input before) := do
  let normalized ← api.normalize p.rows p.columns input before
  let quantized ← api.allocate .int8 [p.rows, p.columns] normalized.after
  let maxima ← api.allocate .float16 [p.rows] quantized.after
  return ⟨normalized, quantized, maxima⟩

def forward (api : Services) (ops : Operations Word) (p : Launch) (schedule : Schedule p)
    (input : Input p.rows p.columns) (before : State) : Option (Tensor × Tensor × Heap) := do
  let prepared ← prepare api p input before
  return (prepared.quantized.tensor, prepared.maxima.tensor,
    runGrid ops p prepared.layout schedule.order prepared.maxima.after.heap)

theorem Prepared.separation {p : Launch} {input : Input p.rows p.columns} {before : State}
    (prepared : Prepared p input before) : Separation p prepared.layout := by
  constructor
  · intro address inside output
    have live := prepared.normalized.resident address inside
    rcases output with hq | hm
    · exact prepared.quantized.fresh address (by simpa [DType.bytes] using hq) live
    · exact prepared.maxima.fresh address (by simpa [DType.bytes] using hm)
        (prepared.quantized.extension.live address live)
  · intro address hq hm
    exact prepared.maxima.fresh address (by simpa [DType.bytes] using hm)
      (prepared.quantized.resident address (by simpa [DType.bytes] using hq))

theorem Prepared.input_preserved {p : Launch} {input : Input p.rows p.columns} {before : State}
    (prepared : Prepared p input before) :
    SameInput p prepared.layout prepared.normalized.after.heap prepared.maxima.after.heap := by
  intro address inside
  exact (prepared.quantized.extension.trans prepared.maxima.extension).frame address
    (prepared.normalized.resident address inside)

theorem Prepared.coordinate {p : Launch} {input : Input p.rows p.columns} {before : State}
    (prepared : Prepared p input before) (r c : Nat) (hr : r < p.rows) (hc : c < p.columns) :
    load32 prepared.maxima.after.heap (inputAddress p prepared.layout r c) = input.logical r c := by
  rw [load32_same p prepared.layout _ _ prepared.input_preserved r c hr hc]
  exact prepared.normalized.coordinate r hr c hc

theorem Prepared.caller_outside {p : Launch} {input : Input p.rows p.columns} {before : State}
    (prepared : Prepared p input before) (address : Nat) (live : before.live address) :
    ¬ OutputCell p prepared.layout address := by
  have hn := prepared.normalized.extension.live address live
  rintro (hq | hm)
  · exact prepared.quantized.fresh address (by simpa [DType.bytes] using hq) hn
  · exact prepared.maxima.fresh address (by simpa [DType.bytes] using hm)
      (prepared.quantized.extension.live address hn)

theorem returned (api : Services) (ops : Operations Word) (p : Launch) (schedule : Schedule p)
    (input : Input p.rows p.columns) (before : State) (q m : Tensor) (after : Heap)
    (run : forward api ops p schedule input before = some (q, m, after)) :
    q.dtype = .int8 ∧ q.shape = [p.rows, p.columns] ∧
    m.dtype = .float16 ∧ m.shape = [p.rows] ∧
    (∀ r, r < p.rows → ∀ c, c < p.columns →
      after (q.base + r * p.columns + c) = rowQuantized ops (input.logical r c)
        (rowMaximum ops p (input.logical r))) ∧
    (∀ r, r < p.rows → ∀ byte : Fin 2,
      after (m.base + 2 * r + byte.val) = halfByte (ops.toHalf (rowMaximum ops p (input.logical r))) byte) ∧
    (∀ address, before.live address → after address = before.heap address) := by
  unfold forward at run
  cases hp : prepare api p input before with
  | none => simp [hp] at run
  | some prepared =>
    simp only [hp] at run
    rcases run with ⟨rfl, rfl, rfl⟩
    obtain ⟨hq, hm⟩ := runGrid_readback ops p prepared.layout prepared.separation schedule prepared.maxima.after.heap
    refine ⟨prepared.quantized.type_eq, prepared.quantized.shape_eq,
      prepared.maxima.type_eq, prepared.maxima.shape_eq, ?_, ?_, ?_⟩
    · intro row hr column hc
      have hmax := rowMaximum_congr ops p _ (input.logical row) (fun c hcc => prepared.coordinate row c hr hcc)
      have result := hq row hr column hc
      unfold qValue at result
      rw [prepared.coordinate row column hr hc, hmax] at result
      exact result
    · intro row hr byte
      have hmax := rowMaximum_congr ops p _ (input.logical row) (fun c hc => prepared.coordinate row c hr hc)
      have result := hm row hr byte
      unfold mValue at result
      rw [hmax] at result
      exact result
    · intro address live
      rw [runGrid_frame ops p prepared.layout schedule.order
        (fun r hr => List.mem_range.mp (schedule.complete.mem_iff.mp hr)) _ address
        (prepared.caller_outside address live)]
      exact (prepared.normalized.extension.trans (prepared.quantized.extension.trans prepared.maxima.extension)).frame address live

#axiomsClean Prepared.separation
#axiomsClean Prepared.coordinate
#axiomsClean returned
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton
