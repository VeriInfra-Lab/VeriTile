import bench.tritonbench_g.lightning_attention.CurrentScanBase
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.Current
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem query_block_inputs (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head block : Nat) (carry : Matrix α p.E p.D) (m : Memory α) :
    Inputs (queryBlockMemory M D p head block carry m) m := by
  unfold queryBlockMemory
  split
  · exact Inputs.refl _
  · apply fold_invariant _ (fun mem => Inputs mem m) _ _ (Inputs.refl _)
    intro k hk mem hmem
    exact (queryUpdate_inputs M D p head _ carry mem).trans hmem

theorem query_block_region (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head block : Nat) (carry : Matrix α p.E p.D) (m : Memory α)
    (region : RegionName) (address : Nat) (hr : region≠"queryGradient") :
    queryBlockMemory M D p head block carry m region address=m region address := by
  unfold queryBlockMemory
  split
  · rfl
  · apply ChunkDeltaFwd.foldRange_observe _ (fun mem : Memory α => mem region address)
    intro k hk mem
    exact query_region_frame M D p head _ carry mem region address hr

theorem query_block_read (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head n : Nat) (carry : Matrix α p.E p.D) (m : Memory α)
    (s : Fin p.NC) (i : TileIndex [p.CB,p.D]) (ht : (n*p.NC+s.val)*p.CB+i.1.val<p.N) :
    queryBlockMemory M D p head n carry m "queryGradient"
        (head*p.N*p.D+((n*p.NC+s.val)*p.CB+i.1.val)*p.D+i.2.1.val)=
      if n=0 then m "queryGradient" (head*p.N*p.D+((n*p.NC+s.val)*p.CB+i.1.val)*p.D+i.2.1.val)
      else .mk .real (cast M (add M (dot D (subG M p head ((n*p.NC+s.val)*p.CB) m) carry i)
        ((m "queryGradient" (head*p.N*p.D+((n*p.NC+s.val)*p.CB+i.1.val)*p.D+i.2.1.val)).read .real))) := by
  unfold queryBlockMemory
  split
  · rfl
  · have h := fold_once
      (fun k mem => queryUpdate M D p head ((n*p.NC+k)*p.CB) carry mem)
      (fun mem : Memory α => mem "queryGradient" (head*p.N*p.D+((n*p.NC+s.val)*p.CB+i.1.val)*p.D+i.2.1.val))
      (fun mem => Inputs mem m)
      (fun cell : Cell α => .mk .real (cast M (add M (dot D (subG M p head ((n*p.NC+s.val)*p.CB) m) carry i) (cell.read .real))))
      p.NC s.val m (Inputs.refl _)
      (fun k hk mem hm => (queryUpdate_inputs M D p head _ carry mem).trans hm)
      (by
        intro mem hm
        dsimp only
        rw [queryUpdate_read M D p head _ carry mem i ht,subG_inputs M p head _ mem m hm])
      (by
        intro k hk hne mem hm
        apply query_time_frame
        intro lane he
        have hc := forward_sub_collision p.NC p.CB n k lane.val n s.val i.1.val hk s.isLt lane.isLt i.1.isLt he
        exact hne hc.2.1)
    simpa only [s.isLt,ite_true] using h

theorem query_other_block (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head k n : Nat) (carry : Matrix α p.E p.D) (m : Memory α)
    (s : Fin p.NC) (i : TileIndex [p.CB,p.D]) (hne : k≠n) :
    queryBlockMemory M D p head k carry m "queryGradient"
        (head*p.N*p.D+((n*p.NC+s.val)*p.CB+i.1.val)*p.D+i.2.1.val)=
      m "queryGradient" (head*p.N*p.D+((n*p.NC+s.val)*p.CB+i.1.val)*p.D+i.2.1.val) := by
  unfold queryBlockMemory
  split
  · rfl
  · apply ChunkDeltaFwd.foldRange_observe _ (fun mem : Memory α => mem "queryGradient" _)
    intro j hj mem
    apply query_time_frame
    intro lane he
    have hc := forward_sub_collision p.NC p.CB k j lane.val n s.val i.1.val hj s.isLt lane.isLt i.1.isLt he
    exact hne hc.1

theorem query_scan_inputs (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head : Nat) (input m : Memory α) : Inputs (queryScanMemory M D p head input m) m := by
  apply fold_invariant _ (fun mem => Inputs mem m) _ _ (Inputs.refl _)
  intro k hk mem hm
  exact (query_block_inputs M D p head _ _ mem).trans hm

theorem query_scan_region (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head : Nat) (input m : Memory α) (region : RegionName) (address : Nat)
    (hr : region≠"queryGradient") :
    queryScanMemory M D p head input m region address=m region address := by
  apply ChunkDeltaFwd.foldRange_observe _ (fun mem : Memory α => mem region address)
  intro k hk mem
  exact query_block_region M D p head _ _ mem region address hr

theorem query_scan_read (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head : Nat) (input m : Memory α) (n : Fin p.NT) (s : Fin p.NC)
    (i : TileIndex [p.CB,p.D]) (ht : (n.val*p.NC+s.val)*p.CB+i.1.val<p.N) :
    queryScanMemory M D p head input m "queryGradient"
        (head*p.N*p.D+((n.val*p.NC+s.val)*p.CB+i.1.val)*p.D+i.2.1.val)=
      if n.val=0 then m "queryGradient" (head*p.N*p.D+((n.val*p.NC+s.val)*p.CB+i.1.val)*p.D+i.2.1.val)
      else .mk .real (cast M (add M
        (dot D (subG M p head ((n.val*p.NC+s.val)*p.CB) m) (queryCarry M D p head input n.val) i)
        ((m "queryGradient" (head*p.N*p.D+((n.val*p.NC+s.val)*p.CB+i.1.val)*p.D+i.2.1.val)).read .real))) := by
  have h := fold_once
    (fun k mem => queryBlockMemory M D p head k (queryCarry M D p head input k) mem)
    (fun mem : Memory α => mem "queryGradient" (head*p.N*p.D+((n.val*p.NC+s.val)*p.CB+i.1.val)*p.D+i.2.1.val))
    (fun mem => Inputs mem m)
    (fun cell : Cell α => if n.val=0 then cell else .mk .real (cast M (add M
      (dot D (subG M p head ((n.val*p.NC+s.val)*p.CB) m) (queryCarry M D p head input n.val) i) (cell.read .real))))
    p.NT n.val m (Inputs.refl _)
    (fun k hk mem hm => (query_block_inputs M D p head k _ mem).trans hm)
    (by
      intro mem hm
      dsimp only
      rw [query_block_read M D p head n.val _ mem s i ht,subG_inputs M p head _ mem m hm])
    (by
      intro k hk hne mem hm
      exact query_other_block M D p head k n.val _ mem s i hne)
  simpa only [n.isLt,ite_true,queryScanMemory] using h

#axiomsClean query_scan_inputs
#axiomsClean query_scan_read
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.Current
