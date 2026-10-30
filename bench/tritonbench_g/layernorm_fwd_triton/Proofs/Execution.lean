import bench.tritonbench_g.layernorm_fwd_triton.Proofs.Model
import bench.tritonbench_g.layernorm_fwd_triton.Proofs.Memory

/-! Physical three-pass execution. Every output iteration reloads X/W from
the current memory. Fresh output windows prove those reloads equal the initial
logical inputs; that fact is not assumed as a kernel-output contract. -/
namespace VeriTile.Bench.Repairs.LayernormFwdTriton
namespace Execution
open Address Memory
open VeriTile.Bench.Repairs.LayerNormFwd.Address (byteAddress)
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000

variable {shape : Shape} {strides : Strides} {α Store Byte : Type}

def columns (N off : Nat) : List (Fin N) :=
  if h : off < N then
    ((List.finRange 128).filterMap fun lane =>
      if hc : off+lane.val < N then some ⟨off+lane.val,hc⟩ else none) ++ columns N (off+128)
  else []
termination_by N-off
decreasing_by omega

theorem mem_columns (N off : Nat) (col : Fin N) (h : off ≤ col.val) : col∈columns N off := by
  have hactive : off < N := by omega
  rw [columns,dif_pos hactive,List.mem_append]
  by_cases hlocal : col.val < off+128
  · left
    apply List.mem_filterMap.mpr
    let lane : Fin 128 := ⟨col.val-off,by omega⟩
    refine ⟨lane,List.mem_finRange _,?_⟩
    have he : off+lane.val=col.val := by dsimp [lane];omega
    simp only [he,col.isLt,dif_pos]
  · right
    exact mem_columns N (off+128) col (by omega)
termination_by N-off
decreasing_by omega

def allRows (shape : Shape) : List (Row shape) :=
  (List.finRange shape.sequences).flatMap fun s => (List.finRange shape.heads).map fun h => (s,h)

theorem mem_rows (row : Row shape) : row∈allRows shape := by
  simp [allRows]

/-- Span fact for every byte in a live FP32 element. -/
theorem cell_span (a : Allocation) (width : a.width=4) {n : Nat} (bound : n < a.extent) :
    a.base ≤ byteAddress a n  ∧  byteAddress a n+4 ≤ a.base+a.extent*a.width := by
  have hb := Nat.mul_le_mul_right 4 (Nat.succ_le_of_lt bound)
  rw [width]
  simp only [byteAddress,width]
  omega

theorem ranges_apart (a b : Allocation) (a4 : a.width=4) (b4 : b.width=4)
    {i j : Nat} (hi : i < a.extent) (hj : j < b.extent)
    (apart : ∀ k, a.base ≤ k → k < a.base+a.extent*a.width →
      ¬(b.base ≤ k  ∧  k < b.base+b.extent*b.width)) :
    Apart (byteAddress a i) (byteAddress b j) := by
  obtain ⟨ha,ha'⟩ := cell_span a a4 hi
  obtain ⟨hb,hb'⟩ := cell_span b b4 hj
  by_contra h
  unfold Apart at h
  have h1 : byteAddress a i+4>byteAddress b j := by omega
  have h2 : byteAddress b j+4>byteAddress a i := by omega
  by_cases ord : byteAddress a i ≤ byteAddress b j
  · exact apart _ (by omega) (by omega) ⟨hb,by omega⟩
  · exact apart _ ha (by omega) ⟨by omega,by omega⟩

theorem inputs_apart (layout : Layout shape strides) (a b : Row shape) (i j : Fin shape.columns) :
    Apart (xAddress layout a i) (yAddress layout b j)  ∧ 
      Apart (wAddress layout a i) (yAddress layout b j) :=
  ⟨ranges_apart _ _ layout.xWidth layout.yWidth (layout.xBounds a i) (layout.yBounds b j) layout.xApart,
   ranges_apart _ _ layout.wWidth layout.yWidth (layout.wBounds a i) (layout.yBounds b j) layout.wApart⟩

theorem outputs_apart (layout : Layout shape strides) (a b : Coordinate shape) (h : a ≠ b) :
    Apart (yAddress layout a.1 a.2) (yAddress layout b.1 b.2) := by
  have hi : yOffset strides a.1 a.2.val ≠ yOffset strides b.1 b.2.val :=
    fun he => h (layout.yInjective he)
  unfold Apart yAddress byteAddress
  rw [layout.yWidth]
  omega

def InputsEqual (api : API α Store Byte) (layout : Layout shape strides) (initial current : Store) : Prop :=
  ∀ row col, api.read current (xAddress layout row col)=api.read initial (xAddress layout row col)  ∧ 
    api.read current (wAddress layout row col)=api.read initial (wAddress layout row col)

theorem inputs_refl (api : API α Store Byte) (layout : Layout shape strides) (m : Store) :
    InputsEqual api layout m m := fun _ _ => ⟨rfl,rfl⟩

theorem write_inputs (api : API α Store Byte) (layout : Layout shape strides)
    (initial current : Store) (h : InputsEqual api layout initial current)
    (row : Row shape) (col : Fin shape.columns) (value : α) :
    InputsEqual api layout initial (api.write current (yAddress layout row col) value) := by
  intro r c
  obtain ⟨hx,hw⟩ := inputs_apart layout r row c col
  rw [api.read_apart _ _ _ _ hx.symm,api.read_apart _ _ _ _ hw.symm]
  exact h r c

def xSnapshot (ops : Native.Ops α) (api : API α Store Byte) (layout : Layout shape strides)
    (m : Store) (row : Row shape) (col : Nat) : α :=
  if h : col < shape.columns then api.read m (xAddress layout row ⟨col,h⟩) else ops.zero

def wSnapshot (ops : Native.Ops α) (api : API α Store Byte) (layout : Layout shape strides)
    (m : Store) (row : Row shape) (col : Nat) : α :=
  if h : col < shape.columns then api.read m (wAddress layout row ⟨col,h⟩) else ops.zero

theorem snapshots_equal (ops : Native.Ops α) (api : API α Store Byte) (layout : Layout shape strides)
    (initial current : Store) (h : InputsEqual api layout initial current) (row : Row shape) :
    xSnapshot ops api layout current row=xSnapshot ops api layout initial row  ∧ 
    wSnapshot ops api layout current row=wSnapshot ops api layout initial row := by
  constructor  <;> funext col  <;> by_cases hc : col < shape.columns
  · simpa only [xSnapshot,dif_pos hc] using (h row ⟨col,hc⟩).1
  · simp [xSnapshot,hc]
  · simpa only [wSnapshot,dif_pos hc] using (h row ⟨col,hc⟩).2
  · simp [wSnapshot,hc]

def emit (ops : Native.Ops α) (plan : Native.Plan shape.columns) (api : API α Store Byte)
    (layout : Layout shape strides) (row : Row shape) (mu inv : α)
    (m : Store) (col : Fin shape.columns) : Store :=
  api.write m (yAddress layout row col)
    (Native.output ops plan mu inv (api.read m (xAddress layout row col))
      (api.read m (wAddress layout row col)))

def executeOutput (ops : Native.Ops α) (plan : Native.Plan shape.columns) (api : API α Store Byte)
    (layout : Layout shape strides) (row : Row shape) (mu inv : α) (m : Store) : Store :=
  (columns shape.columns 0).foldl (emit ops plan api layout row mu inv) m

def executeRow (ops : Native.Ops α) (plan : Native.Plan shape.columns) (api : API α Store Byte)
    (layout : Layout shape strides) (eps : α) (row : Row shape) (m : Store) : Store :=
  let x := xSnapshot ops api layout m row
  let mu := Native.mean ops plan x
  let v := Native.variance ops plan x mu
  executeOutput ops plan api layout row mu (Native.inverseStd ops plan eps v) m

def outputTrace (ops : Native.Ops α) (plan : Native.Plan shape.columns) (api : API α Store Byte)
    (layout : Layout shape strides) (initial : Store) (row : Row shape) (mu inv : α)
    (cols : List (Fin shape.columns)) : List (Write α) := cols.map fun col =>
  ⟨yAddress layout row col,Native.output ops plan mu inv
    (api.read initial (xAddress layout row col)) (api.read initial (wAddress layout row col))⟩

theorem output_trace (ops : Native.Ops α) (plan : Native.Plan shape.columns) (api : API α Store Byte)
    (layout : Layout shape strides) (initial current : Store) (h : InputsEqual api layout initial current)
    (row : Row shape) (mu inv : α) (cols : List (Fin shape.columns)) :
    cols.foldl (emit ops plan api layout row mu inv) current=
      run api (outputTrace ops plan api layout initial row mu inv cols) current  ∧ 
    InputsEqual api layout initial (cols.foldl (emit ops plan api layout row mu inv) current) := by
  induction cols generalizing current with
  | nil => exact ⟨rfl,h⟩
  | cons col cols ih =>
    have hh := write_inputs api layout initial current h row col
      (Native.output ops plan mu inv (api.read current (xAddress layout row col))
        (api.read current (wAddress layout row col)))
    have hrec := ih (emit ops plan api layout row mu inv current col) hh
    refine ⟨?_,hrec.2⟩
    rw [List.foldl_cons,hrec.1]
    simp only [outputTrace,List.map_cons,run,List.foldl_cons,emit,(h row col).1,(h row col).2]

def rowTrace (ops : Native.Ops α) (plan : Native.Plan shape.columns) (api : API α Store Byte)
    (layout : Layout shape strides) (initial : Store) (eps : α) (row : Row shape) : List (Write α) :=
  (columns shape.columns 0).map fun col =>
    ⟨yAddress layout row col,Native.reference ops plan (xSnapshot ops api layout initial row)
      (wSnapshot ops api layout initial row) eps col.val⟩

theorem row_trace (ops : Native.Ops α) (plan : Native.Plan shape.columns) (api : API α Store Byte)
    (layout : Layout shape strides) (initial current : Store) (h : InputsEqual api layout initial current)
    (eps : α) (row : Row shape) :
    executeRow ops plan api layout eps row current=run api (rowTrace ops plan api layout initial eps row) current  ∧ 
    InputsEqual api layout initial (executeRow ops plan api layout eps row current) := by
  have hx := (snapshots_equal ops api layout initial current h row).1
  unfold executeRow executeOutput
  rw [hx]
  have result := output_trace ops plan api layout initial current h row
    (Native.mean ops plan (xSnapshot ops api layout initial row))
    (Native.inverseStd ops plan eps (Native.variance ops plan (xSnapshot ops api layout initial row)
      (Native.mean ops plan (xSnapshot ops api layout initial row)))) (columns shape.columns 0)
  have traces : rowTrace ops plan api layout initial eps row=
      outputTrace ops plan api layout initial row
        (Native.mean ops plan (xSnapshot ops api layout initial row))
        (Native.inverseStd ops plan eps (Native.variance ops plan (xSnapshot ops api layout initial row)
          (Native.mean ops plan (xSnapshot ops api layout initial row)))) (columns shape.columns 0) := by
    unfold rowTrace outputTrace
    apply List.map_congr_left
    intro col _
    simp [Native.reference,xSnapshot,wSnapshot,col.isLt]
  rw [←traces] at result
  exact result


end
end Execution
end VeriTile.Bench.Repairs.LayernormFwdTriton
