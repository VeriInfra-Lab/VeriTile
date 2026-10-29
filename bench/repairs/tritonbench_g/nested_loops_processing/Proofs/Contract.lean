import bench.repairs.tritonbench_g.nested_loops_processing.Contract
import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.Wrapper

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.Admission
open Host Shared.Storage Shared.Address
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section

theorem dimensions_iff (rowIsInt colIsInt : Bool) (rows cols : Int) :
    dimensions rowIsInt colIsInt rows cols = Bool.true ↔
      rowIsInt = Bool.true ∧ colIsInt = Bool.true ∧
      0 ≤ rows ∧ rows < 2^63 ∧ 0 ≤ cols ∧ cols < 2^63 ∧ rows*cols ≤ 2^62 := by
  simp only [dimensions,Bool.and_eq_true,decide_eq_true_eq,and_assoc]

def checkedDense (count base : Nat) (h : buffer count base = Bool.true) : Dense count where
  base := base
  finite := by simpa only [buffer,decide_eq_true_eq,Nat.mul_comm] using (show base%4=0 ∧ base+4*count≤2^64 from by simpa [buffer] using h).2
  aligned := (by simpa only [buffer,decide_eq_true_eq] using h : base%4=0 ∧ base+4*count≤2^64).1

theorem allocated_buffer (p : Packed shape) : buffer shape.prod p.dense.base = Bool.true := by
  simp only [buffer,decide_eq_true_eq]
  exact ⟨p.dense.aligned,by simpa only [Nat.mul_comm] using p.dense.finite⟩

def reject (api : API Word Store Byte) (backings : List Backing) :
    Action (world api backings) Unit := checked .admission (.error (.assertion .admission))

def dispatch (api : API Word Store Byte) (lib : Library api backings)
    (x : Shaped request) (y : Output backings x) : Action (world api backings) Unit :=
  (call .unpackShape (lib.numel x).call).bind fun count =>
  if count.actual = 0 then Action.pure () else
    if buffer (x.rows*x.cols) x.tensor.dense.base && buffer (x.rows*x.cols) y.tensor.dense.base then
      Host.launch api lib x y count else reject api backings

theorem dispatch_eq (api : API Word Store Byte) (lib : Library api backings)
    (x : Shaped request) (y : Output backings x) : dispatch api lib x y = Host.dispatch api lib x y := by
  have hx : buffer (x.rows*x.cols) x.tensor.dense.base = Bool.true := by simpa using allocated_buffer x.tensor
  have hy : buffer (x.rows*x.cols) y.tensor.dense.base = Bool.true := by simpa using allocated_buffer y.tensor
  simp only [dispatch,Host.dispatch,hx,hy,Bool.true_and,↓reduceIte]

def wrapper (api : API Word Store Byte) (lib : Library api backings)
    (rowIsInt colIsInt : Bool) (request : Request) : Action (world api backings) Unit :=
  if dimensions rowIsInt colIsInt request.rows request.cols then
    (input api lib request).bind fun x =>
    (allocate api lib x).bind fun y =>
    (dispatch api lib x y).bind fun _ => call .returnTuple (lib.print x y).call
  else reject api backings

theorem wrapper_eq (api : API Word Store Byte) (lib : Library api backings)
    (rowIsInt colIsInt : Bool) (request : Request)
    (h : dimensions rowIsInt colIsInt request.rows request.cols = Bool.true) :
    wrapper api lib rowIsInt colIsInt request = Host.wrapper api lib request := by
  simp only [wrapper,h,↓reduceIte,dispatch_eq,Host.wrapper]

theorem success_checked (api : API Word Store Byte) (lib : Library api backings)
    (rowIsInt colIsInt : Bool) (request : Request) (s : State Store)
    (ok : ((wrapper api lib rowIsInt colIsInt request).run s).result = .ok ()) :
    dimensions rowIsInt colIsInt request.rows request.cols = Bool.true := by
  by_contra h
  simp [wrapper,h,reject,checked] at ok

theorem rejected_unchanged (api : API Word Store Byte) (lib : Library api backings)
    (rowIsInt colIsInt : Bool) (request : Request) (s : State Store)
    (h : dimensions rowIsInt colIsInt request.rows request.cols = Bool.false) :
    ((wrapper api lib rowIsInt colIsInt request).run s).state = s := by
  simp [wrapper,h,reject,checked]

/-- Every pointer, including inactive lanes, is clamped within a checked
four-byte allocation. The actual raw offsets fit signed64 before clamping. -/
theorem checked_addresses (rows cols base : Nat)
    (h : buffer (rows*cols) base = Bool.true) (positive : 0 < rows*cols)
    (shift : Nat) (hs : shift ≤ 48) (lane : Lane) (byte : Nat) (hb : byte < 4) :
    address cols shift lane < 2^63 ∧
    (checkedDense (rows*cols) base h).contains
      ((checkedDense (rows*cols) base h).address (clamp (rows*cols) (address cols shift lane))+byte) ∧
    (checkedDense (rows*cols) base h).address (clamp (rows*cols) (address cols shift lane))+byte < 2^64 := by
  exact ⟨(offset_intermediates (checkedDense _ _ h) positive shift hs lane).2.1,
    all_pointer_bytes (checkedDense _ _ h) positive _ byte hb⟩

#axiomsClean dimensions_iff
#axiomsClean checked_addresses
#axiomsClean success_checked
#axiomsClean rejected_unchanged
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.NestedLoopsProcessing.Admission
