import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.KernelForwardResult
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.KernelQueryResult
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.KernelBackwardResult
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.KernelStore

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
  (BackwardConfig BackwardAccumulator backwardOutputMemory withDot)
open Native
noncomputable section
set_option maxHeartbeats 10000000

def forwardAddress (s : Shape) (tile : Tile) (i : TileIndex [128,s.BV]) : Nat :=
  tileOffset ((tile.bh+s.B*s.H*tile.key)*(s.T*s.DV)) s.DV (tile.chunk*128) (tile.value*s.BV) i
def keyAddress (s : Shape) (tile : Tile) (i : TileIndex [64,s.BK]) : Nat :=
  tileOffset ((tile.bh+s.B*s.H*tile.value)*(s.T*s.DK)) s.DK (tile.chunk*64) (tile.key*s.BK) i
def valueAddress (s : Shape) (tile : Tile) (i : TileIndex [64,s.BV]) : Nat :=
  tileOffset ((tile.bh+s.B*s.H*tile.key)*(s.T*s.DV)) s.DV (tile.chunk*64) (tile.value*s.BV) i

theorem forwardMemory_eq (s : Shape) (b : Bases) (m : Memory α) (tile : Tile)
    (valid : ProgramDomain s 128 tile) (values : TileIndex [128,s.BV] → Expr α) :
    FPStages.outputMemory algebra (config s 128) (initial s b m tile) values =
      storeTile "o" ((tile.bh+s.B*s.H*tile.key)*(s.T*s.DV)) s.T s.DV
        (tile.chunk*128) (tile.value*s.BV) (initial s b m tile) values := by
  have chunks := source_chunks s 128 b m tile valid
  have decoded := grid_decode s 128 tile valid
  unfold FPStages.outputMemory
  rw [chunks.1,chunks.2.2]
  change (TileShape.allIndices [128,s.BV]).foldl
    (fun t i=>if tile.chunk*128+i.1.val<s.T ∧ tile.value*s.BV+i.2.1.val<s.DV then
      t.write "o" ((tile.bh+s.B*s.H*(ProgramDomain.gridX s tile/blocks s.DV s.BV))*(s.T*s.DV)+
        (tile.chunk*128+i.1.val)*s.DV+(tile.value*s.BV+i.2.1.val)*1) (.mk .real (values i)) else t)
      (initial s b m tile) = _
  rw [decoded.2.1]
  simp only [storeTile,tileOffset,Nat.mul_one]

theorem queryMemory_eq (s : Shape) (tile : Tile) (state : State (Expr α))
    (values : TileIndex [64,s.BK] → Expr α) :
    FPQuery.outputMemory algebra (backwardConfig s tile) "dq" state values =
      storeTile "dq" ((tile.bh+s.B*s.H*tile.value)*(s.T*s.DK)) s.T s.DK
        (tile.chunk*64) (tile.key*s.BK) state values := by
  simp only [FPQuery.outputMemory,storeTile,tileOffset,backwardConfig,config,algebra,Nat.mul_one,id_eq]

theorem backwardMemory_eq (s : Shape) (tile : Tile) (state : State (Expr α))
    (values : BackwardAccumulator (Expr α) 64 s.BK s.BV) :
    backwardOutputMemory algebra (backwardConfig s tile) state values =
      storeTile "dv" ((tile.bh+s.B*s.H*tile.key)*(s.T*s.DV)) s.T s.DV
        (tile.chunk*64) (tile.value*s.BV)
        (storeTile "dk" ((tile.bh+s.B*s.H*tile.value)*(s.T*s.DK)) s.T s.DK
          (tile.chunk*64) (tile.key*s.BK) state values.key) values.value := by
  simp only [backwardOutputMemory,storeTile,tileOffset,backwardConfig,config,algebra,Nat.mul_one,id_eq]

theorem backwardValues_inputs (c : BackwardConfig) (s t : State (Expr α))
    (hq : s.mem c.q=t.mem c.q) (hk : s.mem c.k=t.mem c.k)
    (hv : s.mem c.v=t.mem c.v) (hg : s.mem c.do_=t.mem c.do_) :
    FPStages.backwardValues algebra dots c s=FPStages.backwardValues algebra dots c t := by
  have readEq (r : RegionName) (h : s.mem r=t.mem r)
      (base rows cols rs cs ro co height width : Nat) :
      matrixRead algebra s r base rows cols rs cs ro co height width=
        matrixRead algebra t r base rows cols rs cs ro co height width := by
    funext i
    simp only [matrixRead,h]
  have update : FPStages.backwardUpdate algebra dots c s=FPStages.backwardUpdate algebra dots c t := by
    funext diagonal start previous
    simp only [FPStages.backwardUpdate,FPStages.decayedKey,backwardKey,backwardValue,
      readEq c.q hq,readEq c.k hk,readEq c.v hv,readEq c.do_ hg]
  simp only [FPStages.backwardValues,update]

theorem queryMemory_inputs (s : Shape) (tile : Tile) (state : State (Expr α))
    (values : TileIndex [64,s.BK] → Expr α) (region : RegionName) (different : region≠"dq") :
    (FPQuery.outputMemory algebra (backwardConfig s tile) "dq" state values).mem region=state.mem region := by
  funext offset
  rw [queryMemory_eq]
  exact storeTile_frame _ _ _ _ _ _ _ _ _ _ different

theorem forward_exec (s : Shape) (b : Bases) (m : Memory α) (tile : Tile)
    (valid : ProgramDomain s 128 tile) :
    ∃ final, FP.Structural.exec (withDot algebra dots) (FPStages.forwardKernel (config s 128))
      (initial s b m tile)=some final ∧
      ∀ i : TileIndex [128,s.BV],tile.chunk*128+i.1.val<s.T → tile.value*s.BV+i.2.1.val<s.DV →
        final.mem "o" (forwardAddress s tile i)=
          .mk .real (FPStages.forwardValues algebra dots (config s 128) (initial s b m tile) i) := by
  obtain ⟨final,run,mem⟩ := FPStages.forward_fp32 algebra dots (config s 128) (initial s b m tile)
  refine ⟨final,run,?_⟩
  intro i hr hc
  rw [mem,forwardMemory_eq s b m tile valid]
  exact storeTile_read _ _ _ _ _ _ _ _ i hr hc

theorem backward_exec (s : Shape) (b : Bases) (m : Memory α) (tile : Tile) :
    ∃ final, FP.Structural.exec (withDot algebra dots) (FPBackward.source (backwardConfig s tile) "dq")
      (initial s b m tile)=some final ∧
      (∀ i : TileIndex [64,s.BK],tile.chunk*64+i.1.val<s.T → tile.key*s.BK+i.2.1.val<s.DK →
        final.mem "dq" (keyAddress s tile i)=
          .mk .real (FPQuery.values algebra dots (backwardConfig s tile) (initial s b m tile) i)) ∧
      (∀ i : TileIndex [64,s.BK],tile.chunk*64+i.1.val<s.T → tile.key*s.BK+i.2.1.val<s.DK →
        final.mem "dk" (keyAddress s tile i)=
          .mk .real ((FPStages.backwardValues algebra dots (backwardConfig s tile) (initial s b m tile)).key i)) ∧
      (∀ i : TileIndex [64,s.BV],tile.chunk*64+i.1.val<s.T → tile.value*s.BV+i.2.1.val<s.DV →
        final.mem "dv" (valueAddress s tile i)=
          .mk .real ((FPStages.backwardValues algebra dots (backwardConfig s tile) (initial s b m tile)).value i)) := by
  let before := initial s b m tile
  let afterQuery := FPQuery.outputMemory algebra (backwardConfig s tile) "dq" before
    (FPQuery.values algebra dots (backwardConfig s tile) before)
  have unchanged : FPStages.backwardValues algebra dots (backwardConfig s tile) afterQuery=
      FPStages.backwardValues algebra dots (backwardConfig s tile) before := by
    apply backwardValues_inputs
    all_goals apply queryMemory_inputs
    all_goals simp [backwardConfig,config]
  obtain ⟨final,run,mem⟩ := FPBackward.backward_fp32 algebra dots (backwardConfig s tile) "dq" before
  have resultMemory : final.mem=(backwardOutputMemory algebra (backwardConfig s tile) afterQuery
      (FPStages.backwardValues algebra dots (backwardConfig s tile) before)).mem := by
    change final.mem=(backwardOutputMemory algebra (backwardConfig s tile) afterQuery
      (FPStages.backwardValues algebra dots (backwardConfig s tile) afterQuery)).mem at mem
    rw [unchanged] at mem
    exact mem
  refine ⟨final,run,?_,?_,?_⟩
  · intro i hr hc
    rw [resultMemory,backwardMemory_eq]
    rw [storeTile_frame _ _ _ _ _ _ _ _ _ _ (by decide),
      storeTile_frame _ _ _ _ _ _ _ _ _ _ (by decide)]
    change (FPQuery.outputMemory algebra (backwardConfig s tile) "dq" before _).mem _ _ = _
    rw [queryMemory_eq]
    exact storeTile_read _ _ _ _ _ _ _ _ i hr hc
  · intro i hr hc
    rw [resultMemory,backwardMemory_eq,storeTile_frame _ _ _ _ _ _ _ _ _ _ (by decide)]
    exact storeTile_read _ _ _ _ _ _ _ _ i hr hc
  · intro i hr hc
    rw [resultMemory,backwardMemory_eq]
    exact storeTile_read _ _ _ _ _ _ _ _ i hr hc

#axiomsClean forward_exec
#axiomsClean backward_exec
#axiomsClean backwardValues_inputs
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
