import bench.tritonbench_g.int8_dequant_matmul.Proofs.IOStores
import bench.tritonbench_g.int8_dequant_matmul.Proofs.QuoteDescriptors

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOSourceResult
open VeriTile Triton FP.Structural Quote
open IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

/-- Exact one-cell observation of a complete quoted CTA. It depends only on
readonly inputs and the selected current C cell, and preserves other owners. -/
theorem cell (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool)
    (vals : (IOLayout.memory F H s c hasBias).Inputs) (initial : State (Sum F H))
    (cells : IOInputReads.DataCells s c hasBias vals initial)
    (selected : Selection s c) (plans : q.plans=selectedPlan s c selected)
    (valid : initial.pids 0<Geometry.count (geometry s c)) (hz : initial.pids 1<c.S)
    (divisor : F) (hasDivisor : initial.regs .fp32 [] "divfactor"=some (fun _=>.inl divisor))
    (final : State (Sum F H))
    (ran : execute (floating q) q s.M s.N (source s c hasBias) initial=some final)
    (x : MatrixCell s) :
    final.mem "C" (x.1.val*s.N+x.2.val)=
      if initial.pids 0=(IOPrograms.owner s c x).val then
        Cell.mk .fp16 (.inr (if c.S=1 then
          Mathematical.splitValue q.ops s c selected divisor hasBias (IOLayout.data q.ops s vals) x
            ⟨initial.pids 1,hz⟩ else q.ops.hadd
          (asH q.ops ((initial.mem "C" (x.1.val*s.N+x.2.val)).read .fp16))
          (Mathematical.splitValue q.ops s c selected divisor hasBias (IOLayout.data q.ops s vals) x
            ⟨initial.pids 1,hz⟩)))
      else initial.mem "C" (x.1.val*s.N+x.2.val) := by
  obtain ⟨actual,execution,result⟩:=fullResult q s c hasBias initial divisor hasDivisor
  have same : actual=final := Option.some.inj (execution.symm.trans ran)
  subst final
  rw [result.memory]
  by_cases owns : initial.pids 0=(IOPrograms.owner s c x).val
  · rw [if_pos owns]
    have owned:=IOGeometry.owner_owns s c x
    rw [←owns] at owned
    obtain ⟨i,rows,columns⟩:=owned
    change row s c (initial.pids 0) i.1=x.1.val at rows
    change col s c (initial.pids 0) i.2.1=x.2.val at columns
    have hr : row s c (initial.pids 0) i.1<s.M := rows ▸ x.1.isLt
    have hc : col s c (initial.pids 0) i.2.1<s.N := columns ▸ x.2.isLt
    have enabled : Quote.active s c (preludeExecution (floating q) q s c hasBias initial).val i :=
      (IOStoreGeometry.active_iff (floating q) q s c hasBias initial valid i).mpr ⟨hr,hc⟩
    have address:=IOStoreGeometry.offset_eq (floating q) q s c hasBias initial valid i
    rw [rows,columns] at address
    rw [←address,IOStores.stores_at q s c hasBias initial valid divisor i enabled]
    have coordinate : (⟨row s c (initial.pids 0) i.1,hr⟩,⟨col s c (initial.pids 0) i.2.1,hc⟩)=x :=
      Prod.ext (Fin.ext rows) (Fin.ext columns)
    rw [IOValue.source_value q s c hasBias vals initial cells selected plans valid hz divisor i hr hc,
      coordinate,address]
  · rw [if_neg owns]
    apply IOStores.stores_miss
    exact Or.inr (fun i enabled=>IOStoreGeometry.offset_miss (floating q) q s c hasBias initial valid x owns i enabled)

theorem frame (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool)
    (vals : (IOLayout.memory F H s c hasBias).Inputs) (initial : State (Sum F H))
    (valid : initial.pids 0<Geometry.count (geometry s c))
    (divisor : F) (hasDivisor : initial.regs .fp32 [] "divfactor"=some (fun _=>.inl divisor))
    (final : State (Sum F H))
    (ran : execute (floating q) q s.M s.N (source s c hasBias) initial=some final) :
    (IOLayout.memory F H s c hasBias).Frame vals initial final := by
  obtain ⟨actual,execution,result⟩:=fullResult q s c hasBias initial divisor hasDivisor
  have same : actual=final := Option.some.inj (execution.symm.trans ran)
  subst final
  exact IOStores.stores_frame q s c hasBias vals initial valid divisor actual result.memory

#axiomsClean cell
#axiomsClean frame
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOSourceResult
