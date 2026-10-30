import bench.tritonbench_g.parallel_retention_attention.Proofs.PublicSourceMemory
import bench.tritonbench_g.parallel_retention_attention.Proofs.PublicObservedValues

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.PublicIO
open VeriTile.Triton FP.Structural Kernel Kernel.Native
noncomputable section
set_option maxHeartbeats 2000000
variable {α : Type}

/-- The actual combined backward shell produces all three channels and has
one complete frame; the intermediate DQ store remains part of execution. -/
theorem backward_implements (a : Scalar α) (p : DotTrees) (s : Shape) (tile : Tile)
    (valid : ProgramDomain s 64 tile) (keyPositive : 0<s.BK) (valuePositive : 0<s.BV) :
    (backwardIO a p s tile).Implements (backwardFormula a p s tile) := by
  intro xs _ bounds bounded state binding
  obtain ⟨final,ran,memory⟩ := backward_run s tile valid state
  let start := (backwardInvocation α s tile).initial state
  let q := FPQuery.values algebra dots (backwardConfig s tile) start
  let kv := FPStages.backwardValues algebra dots (backwardConfig s tile) start
  let qmem := storeTile "dq" ((tile.bh+s.B*s.H*tile.value)*(s.T*s.DK)) s.T s.DK
    (tile.chunk*64) (tile.key*s.BK) start q
  let kmem := storeTile "dk" ((tile.bh+s.B*s.H*tile.value)*(s.T*s.DK)) s.T s.DK
    (tile.chunk*64) (tile.key*s.BK) qmem kv.key
  let vmem := storeTile "dv" ((tile.bh+s.B*s.H*tile.key)*(s.T*s.DV)) s.T s.DV
    (tile.chunk*64) (tile.value*s.BV) kmem kv.value
  change final.mem=vmem.mem at memory
  refine ⟨backward_schedule_safe a p s tile valid xs bounds bounded state,final,ran,?_,?_⟩
  · intro o i active
    fin_cases o
    · change mapCell (evaluate a p) (final.mem "dq" (keyAddress s tile i)) =
        FP.Structural.Cell.mk .real (backwardFormula a p s tile xs (0 : Fin 3) i)
      change activeTile s.T s.DK (tile.chunk*64) (tile.key*s.BK) i at active
      have cell : final.mem "dq" (keyAddress s tile i)=FP.Structural.Cell.mk .real (q i) := by
        calc
          _ = vmem.mem "dq" (keyAddress s tile i) := congrFun (congrFun memory "dq") _
          _ = kmem.mem "dq" (keyAddress s tile i) := storeTile_frame "dv" "dq" _ _ _ _ _ _ _ _ (by decide)
          _ = qmem.mem "dq" (keyAddress s tile i) := storeTile_frame "dk" "dq" _ _ _ _ _ _ _ _ (by decide)
          _ = _ := storeTile_read "dq" _ _ _ _ _ _ _ i active.1 active.2
      rw [cell]
      change FP.Structural.Cell.mk .real (evaluate a p _) = _
      rw [query_observed a p s tile valid xs state binding i]
    · change mapCell (evaluate a p) (final.mem "dk" (keyAddress s tile i)) =
        FP.Structural.Cell.mk .real (backwardFormula a p s tile xs (1 : Fin 3) i)
      change activeTile s.T s.DK (tile.chunk*64) (tile.key*s.BK) i at active
      have cell : final.mem "dk" (keyAddress s tile i)=FP.Structural.Cell.mk .real (kv.key i) := by
        calc
          _ = vmem.mem "dk" (keyAddress s tile i) := congrFun (congrFun memory "dk") _
          _ = kmem.mem "dk" (keyAddress s tile i) := storeTile_frame "dv" "dk" _ _ _ _ _ _ _ _ (by decide)
          _ = _ := storeTile_read "dk" _ _ _ _ _ _ _ i active.1 active.2
      rw [cell]
      change FP.Structural.Cell.mk .real (evaluate a p _) = _
      rw [key_observed a p s tile valid valuePositive xs state binding i]
    · change mapCell (evaluate a p) (final.mem "dv" (valueAddress s tile i)) =
        FP.Structural.Cell.mk .real (backwardFormula a p s tile xs (2 : Fin 3) i)
      change activeTile s.T s.DV (tile.chunk*64) (tile.value*s.BV) i at active
      have cell : final.mem "dv" (valueAddress s tile i)=FP.Structural.Cell.mk .real (kv.value i) := by
        rw [memory]
        exact storeTile_read "dv" _ _ _ _ _ _ _ i active.1 active.2
      rw [cell]
      change FP.Structural.Cell.mk .real (evaluate a p _) = _
      rw [value_observed a p s tile valid keyPositive xs state binding i]
  · intro r off outside
    calc
      _ = vmem.mem r off := congrFun (congrFun memory r) off
      _ = kmem.mem r off := storeTile_frame_full "dv" r _ _ _ _ _ _ _ off
        (fun i hi => outside.1 (2 : Fin 3) i hi)
      _ = qmem.mem r off := storeTile_frame_full "dk" r _ _ _ _ _ _ _ off
        (fun i hi => outside.1 (1 : Fin 3) i hi)
      _ = start.mem r off := storeTile_frame_full "dq" r _ _ _ _ _ _ _ off
        (fun i hi => outside.1 (0 : Fin 3) i hi)
      _ = _ := rfl


#axiomsClean backward_implements
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.PublicIO
