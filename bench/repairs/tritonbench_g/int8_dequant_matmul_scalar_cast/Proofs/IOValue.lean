import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.IORecurrence
import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.QuoteValue

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOValue
open VeriTile Triton FP.Structural Quote
open IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

/-- The actual quotation's output register equals the finite independent
instruction expression for every active lane and arbitrary logical inputs. -/
theorem source_value (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool)
    (vals : (IOLayout.memory F H s c hasBias).Inputs) (initial : State (Sum F H))
    (cells : IOInputReads.DataCells s c hasBias vals initial)
    (selected : Selection s c) (plans : q.plans=selectedPlan s c selected)
    (valid : initial.pids 0<Geometry.count (geometry s c)) (hz : initial.pids 1<c.S)
    (divisor : F) (i : TileIndex [c.BM,c.BN])
    (hr : row s c (initial.pids 0) i.1<s.M) (hc : col s c (initial.pids 0) i.2.1<s.N) :
    let base:=(preludeExecution (floating q) q s c hasBias initial).val
    value q s c hasBias base divisor i=
      Mathematical.splitValue q.ops s c selected divisor hasBias (IOLayout.data q.ops s vals)
        (⟨row s c (initial.pids 0) i.1,hr⟩,⟨col s c (initial.pids 0) i.2.1,hc⟩)
        ⟨initial.pids 1,hz⟩ := by
  let cell : MatrixCell s :=
    (⟨row s c (initial.pids 0) i.1,hr⟩,⟨col s c (initial.pids 0) i.2.1,hc⟩)
  have wrapped : IORecurrence.wrappedCell s c (initial.pids 0) valid i=cell := by
    apply Prod.ext
    · apply Fin.ext
      simp [IORecurrence.wrappedCell,cell,Nat.mod_eq_of_lt hr]
    · apply Fin.ext
      simp [IORecurrence.wrappedCell,cell,Nat.mod_eq_of_lt hc]
  have hx:=factorX q s c hasBias initial valid
  have hw:=factorW q s c hasBias initial valid
  have rn:=prelude_rn (floating q) q s c hasBias initial valid
  have pz:=prelude_pz (floating q) q s c hasBias initial
  have acc:=IORecurrence.source_accumulation q s c hasBias vals initial cells selected plans valid hz i
  rw [wrapped] at acc
  have xr:=IOInputReads.rowScale s c hasBias vals initial cells cell.1
  have xw:=IOInputReads.columnScale s c hasBias vals initial cells cell.2
  dsimp only
  unfold value
  rw [acc,hx,hw,rn,pz,prelude_memory]
  simp only [Nat.mod_eq_of_lt hr,Nat.mod_eq_of_lt hc]
  rw [xr,xw]
  cases hasBias
  · rfl
  ·
    have xb:=IOInputReads.bias s c Bool.true vals initial cells cell.2 rfl
    simp only [Int.ofNat_lt,hc,ite_true,Int.toNat_natCast]
    rw [xb]
    simp [Mathematical.splitValue,IOLayout.data,cell]

#axiomsClean source_value
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOValue
