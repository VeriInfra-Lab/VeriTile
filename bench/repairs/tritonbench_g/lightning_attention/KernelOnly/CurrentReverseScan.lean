import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.CurrentScanBase
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.Current
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def reverseRow (p : Parameters) (n s : Nat) := p.NT*p.BT-(n*p.NC+s+1)*p.CB

theorem reverse_block_inputs (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head block : Nat) (carry : Matrix α p.D p.E) (m : Memory α) :
    Inputs (keyValueBlockMemory M D p head block carry m) m := by
  unfold keyValueBlockMemory
  split
  · exact Inputs.refl _
  · apply fold_invariant _ (fun mem => Inputs mem m) _ _ (Inputs.refl _)
    intro k hk mem hmem
    exact (keyValueUpdate_inputs M D p head _ carry mem).trans hmem

theorem reverse_block_region (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head block : Nat) (carry : Matrix α p.D p.E) (m : Memory α)
    (region : RegionName) (address : Nat) (hk : region≠"keyGradient") (hv : region≠"valueGradient") :
    keyValueBlockMemory M D p head block carry m region address=m region address := by
  unfold keyValueBlockMemory
  split
  · rfl
  · apply ChunkDeltaFwd.foldRange_observe _ (fun mem : Memory α => mem region address)
    intro k hk' mem
    exact keyValue_region_frame M D p head _ carry mem region address hk hv

theorem reverse_scan_inputs (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head : Nat) (input m : Memory α) : Inputs (keyValueScanMemory M D p head input m) m := by
  apply fold_invariant _ (fun mem => Inputs mem m) _ _ (Inputs.refl _)
  intro k hk mem hm
  exact (reverse_block_inputs M D p head _ _ mem).trans hm

theorem reverse_scan_region (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head : Nat) (input m : Memory α) (region : RegionName) (address : Nat)
    (hk : region≠"keyGradient") (hv : region≠"valueGradient") :
    keyValueScanMemory M D p head input m region address=m region address := by
  apply ChunkDeltaFwd.foldRange_observe _ (fun mem : Memory α => mem region address)
  intro k hk' mem
  exact reverse_block_region M D p head _ _ mem region address hk hv

theorem key_block_read (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (htile : p.NC*p.CB=p.BT) (head : Nat) (n : Fin p.NT) (carry : Matrix α p.D p.E) (m : Memory α)
    (s : Fin p.NC) (i : TileIndex [p.D,p.CB]) (ht : reverseRow p n.val s.val+i.2.1.val<p.N) :
    keyValueBlockMemory M D p head n.val carry m "keyGradient" (head*p.N*p.D+(reverseRow p n.val s.val+i.2.1.val)*p.D+i.1.val)=
      if n.val=0 then m "keyGradient" (head*p.N*p.D+(reverseRow p n.val s.val+i.2.1.val)*p.D+i.1.val)
      else .mk .real (cast M (add M (dot D carry (subVT M p head (reverseRow p n.val s.val) m) i) ((m "keyGradient" (head*p.N*p.D+(reverseRow p n.val s.val+i.2.1.val)*p.D+i.1.val)).read .real))) := by
  unfold keyValueBlockMemory
  split
  · rfl
  · have h := fold_once
      (fun k mem => keyValueUpdate M D p head (reverseRow p n.val k) carry mem)
      (fun mem : Memory α => mem "keyGradient" (head*p.N*p.D+(reverseRow p n.val s.val+i.2.1.val)*p.D+i.1.val)) (fun mem => Inputs mem m)
      (fun cell : Cell α => .mk .real (cast M (add M (dot D carry (subVT M p head (reverseRow p n.val s.val) m) i) (cell.read .real))))
      p.NC s.val m (Inputs.refl _)
      (fun k hk mem hm => (keyValueUpdate_inputs M D p head _ carry mem).trans hm)
      (by
        intro mem hm
        dsimp only
        rw [keyUpdate_read M D p head _ carry mem i ht,subVT_inputs M p head _ mem m hm])
      (by
        intro k hk hne mem hm
        apply key_time_frame
        intro lane he
        have hc := reverse_sub_collision p htile n.val k lane.val n.val s.val i.2.1.val
          n.isLt n.isLt hk s.isLt lane.isLt i.2.1.isLt he
        exact hne hc.2.1)
    simpa only [s.isLt,ite_true,reverseRow] using h

theorem key_other_block (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (htile : p.NC*p.CB=p.BT) (head : Nat) (k n : Fin p.NT) (carry : Matrix α p.D p.E) (m : Memory α)
    (s : Fin p.NC) (i : TileIndex [p.D,p.CB]) (hne : k.val≠n.val) :
    keyValueBlockMemory M D p head k.val carry m "keyGradient" (head*p.N*p.D+(reverseRow p n.val s.val+i.2.1.val)*p.D+i.1.val)=m "keyGradient" (head*p.N*p.D+(reverseRow p n.val s.val+i.2.1.val)*p.D+i.1.val) := by
  unfold keyValueBlockMemory
  split
  · rfl
  · apply ChunkDeltaFwd.foldRange_observe _ (fun mem : Memory α => mem "keyGradient" _)
    intro j hj mem
    apply key_time_frame
    intro lane he
    have hc := reverse_sub_collision p htile k.val j lane.val n.val s.val i.2.1.val
      k.isLt n.isLt hj s.isLt lane.isLt i.2.1.isLt he
    exact hne hc.1

theorem key_scan_read (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (htile : p.NC*p.CB=p.BT) (head : Nat) (input m : Memory α) (n : Fin p.NT) (s : Fin p.NC)
    (i : TileIndex [p.D,p.CB]) (ht : reverseRow p n.val s.val+i.2.1.val<p.N) :
    keyValueScanMemory M D p head input m "keyGradient" (head*p.N*p.D+(reverseRow p n.val s.val+i.2.1.val)*p.D+i.1.val)=
      if n.val=0 then m "keyGradient" (head*p.N*p.D+(reverseRow p n.val s.val+i.2.1.val)*p.D+i.1.val)
      else .mk .real (cast M (add M (dot D (keyValueCarry M D p head input n.val) (subVT M p head (reverseRow p n.val s.val) m) i)
        ((m "keyGradient" (head*p.N*p.D+(reverseRow p n.val s.val+i.2.1.val)*p.D+i.1.val)).read .real))) := by
  have h := fold_once
    (fun k mem => keyValueBlockMemory M D p head k (keyValueCarry M D p head input k) mem)
    (fun mem : Memory α => mem "keyGradient" (head*p.N*p.D+(reverseRow p n.val s.val+i.2.1.val)*p.D+i.1.val)) (fun mem => Inputs mem m)
    (fun cell : Cell α => if n.val=0 then cell else .mk .real (cast M (add M
      (dot D (keyValueCarry M D p head input n.val) (subVT M p head (reverseRow p n.val s.val) m) i) (cell.read .real))))
    p.NT n.val m (Inputs.refl _)
    (fun k hk mem hm => (reverse_block_inputs M D p head k _ mem).trans hm)
    (by
      intro mem hm
      dsimp only
      rw [key_block_read M D p htile head n _ mem s i ht,subVT_inputs M p head _ mem m hm])
    (by
      intro k hk hne mem hm
      exact key_other_block M D p htile head ⟨k,hk⟩ n _ mem s i hne)
  simpa only [n.isLt,ite_true,keyValueScanMemory] using h

theorem value_block_read (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (htile : p.NC*p.CB=p.BT) (head : Nat) (n : Fin p.NT) (carry : Matrix α p.D p.E) (m : Memory α)
    (s : Fin p.NC) (i : TileIndex [p.CB,p.E]) (ht : reverseRow p n.val s.val+i.1.val<p.N) :
    keyValueBlockMemory M D p head n.val carry m "valueGradient" (head*p.N*p.E+(reverseRow p n.val s.val+i.1.val)*p.E+i.2.1.val)=
      if n.val=0 then m "valueGradient" (head*p.N*p.E+(reverseRow p n.val s.val+i.1.val)*p.E+i.2.1.val)
      else .mk .real (cast M (add M (dot D (subK M p head (reverseRow p n.val s.val) m) carry i) ((m "valueGradient" (head*p.N*p.E+(reverseRow p n.val s.val+i.1.val)*p.E+i.2.1.val)).read .real))) := by
  unfold keyValueBlockMemory
  split
  · rfl
  · have h := fold_once
      (fun k mem => keyValueUpdate M D p head (reverseRow p n.val k) carry mem)
      (fun mem : Memory α => mem "valueGradient" (head*p.N*p.E+(reverseRow p n.val s.val+i.1.val)*p.E+i.2.1.val)) (fun mem => Inputs mem m)
      (fun cell : Cell α => .mk .real (cast M (add M (dot D (subK M p head (reverseRow p n.val s.val) m) carry i) (cell.read .real))))
      p.NC s.val m (Inputs.refl _)
      (fun k hk mem hm => (keyValueUpdate_inputs M D p head _ carry mem).trans hm)
      (by
        intro mem hm
        dsimp only
        rw [valueUpdate_read M D p head _ carry mem i ht,subK_inputs M p head _ mem m hm])
      (by
        intro k hk hne mem hm
        apply value_time_frame
        intro lane he
        have hc := reverse_sub_collision p htile n.val k lane.val n.val s.val i.1.val
          n.isLt n.isLt hk s.isLt lane.isLt i.1.isLt he
        exact hne hc.2.1)
    simpa only [s.isLt,ite_true,reverseRow] using h

theorem value_other_block (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (htile : p.NC*p.CB=p.BT) (head : Nat) (k n : Fin p.NT) (carry : Matrix α p.D p.E) (m : Memory α)
    (s : Fin p.NC) (i : TileIndex [p.CB,p.E]) (hne : k.val≠n.val) :
    keyValueBlockMemory M D p head k.val carry m "valueGradient" (head*p.N*p.E+(reverseRow p n.val s.val+i.1.val)*p.E+i.2.1.val)=m "valueGradient" (head*p.N*p.E+(reverseRow p n.val s.val+i.1.val)*p.E+i.2.1.val) := by
  unfold keyValueBlockMemory
  split
  · rfl
  · apply ChunkDeltaFwd.foldRange_observe _ (fun mem : Memory α => mem "valueGradient" _)
    intro j hj mem
    apply value_time_frame
    intro lane he
    have hc := reverse_sub_collision p htile k.val j lane.val n.val s.val i.1.val
      k.isLt n.isLt hj s.isLt lane.isLt i.1.isLt he
    exact hne hc.1

theorem value_scan_read (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (htile : p.NC*p.CB=p.BT) (head : Nat) (input m : Memory α) (n : Fin p.NT) (s : Fin p.NC)
    (i : TileIndex [p.CB,p.E]) (ht : reverseRow p n.val s.val+i.1.val<p.N) :
    keyValueScanMemory M D p head input m "valueGradient" (head*p.N*p.E+(reverseRow p n.val s.val+i.1.val)*p.E+i.2.1.val)=
      if n.val=0 then m "valueGradient" (head*p.N*p.E+(reverseRow p n.val s.val+i.1.val)*p.E+i.2.1.val)
      else .mk .real (cast M (add M (dot D (subK M p head (reverseRow p n.val s.val) m) (keyValueCarry M D p head input n.val) i)
        ((m "valueGradient" (head*p.N*p.E+(reverseRow p n.val s.val+i.1.val)*p.E+i.2.1.val)).read .real))) := by
  have h := fold_once
    (fun k mem => keyValueBlockMemory M D p head k (keyValueCarry M D p head input k) mem)
    (fun mem : Memory α => mem "valueGradient" (head*p.N*p.E+(reverseRow p n.val s.val+i.1.val)*p.E+i.2.1.val)) (fun mem => Inputs mem m)
    (fun cell : Cell α => if n.val=0 then cell else .mk .real (cast M (add M
      (dot D (subK M p head (reverseRow p n.val s.val) m) (keyValueCarry M D p head input n.val) i) (cell.read .real))))
    p.NT n.val m (Inputs.refl _)
    (fun k hk mem hm => (reverse_block_inputs M D p head k _ mem).trans hm)
    (by
      intro mem hm
      dsimp only
      rw [value_block_read M D p htile head n _ mem s i ht,subK_inputs M p head _ mem m hm])
    (by
      intro k hk hne mem hm
      exact value_other_block M D p htile head ⟨k,hk⟩ n _ mem s i hne)
  simpa only [n.isLt,ite_true,keyValueScanMemory] using h

#axiomsClean reverse_scan_inputs
#axiomsClean key_scan_read
#axiomsClean value_scan_read
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.Current
