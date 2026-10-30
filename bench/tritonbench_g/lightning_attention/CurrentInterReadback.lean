import bench.tritonbench_g.lightning_attention.CurrentQueryScan
import bench.tritonbench_g.lightning_attention.CurrentReverseScan
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.Current
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem inter_inputs (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head : Nat) (input m : Memory α) : Inputs (interMemory M D p head input m) m :=
  (reverse_scan_inputs M D p head input _).trans (query_scan_inputs M D p head input m)

/-- Complete current-memory inter scan readback. The preserved intra cell is
read at the one owning query update; all other subblocks and the reverse scan
preserve that cell. This theorem is polymorphic in the scalar semantics. -/
theorem inter_query_read (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head : Nat) (input m : Memory α) (n : Fin p.NT) (s : Fin p.NC)
    (i : TileIndex [p.CB,p.D]) (ht : (n.val*p.NC+s.val)*p.CB+i.1.val<p.N) :
    interMemory M D p head input m "queryGradient"
        (head*p.N*p.D+((n.val*p.NC+s.val)*p.CB+i.1.val)*p.D+i.2.1.val)=
      if n.val=0 then m "queryGradient" (head*p.N*p.D+((n.val*p.NC+s.val)*p.CB+i.1.val)*p.D+i.2.1.val)
      else .mk .real (cast M (add M
        (dot D (subG M p head ((n.val*p.NC+s.val)*p.CB) m) (queryCarry M D p head input n.val) i)
        ((m "queryGradient" (head*p.N*p.D+((n.val*p.NC+s.val)*p.CB+i.1.val)*p.D+i.2.1.val)).read .real))) := by
  unfold interMemory
  rw [reverse_scan_region M D p head input _ "queryGradient" _ (by decide) (by decide)]
  exact query_scan_read M D p head input m n s i ht

/-- The actual backward key scan adds its proved future carry to the current
intra-produced key cell; the preceding query scan preserves both operands. -/
theorem inter_key_read (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (htile : p.NC*p.CB=p.BT) (head : Nat) (input m : Memory α) (n : Fin p.NT) (s : Fin p.NC)
    (i : TileIndex [p.D,p.CB]) (ht : reverseRow p n.val s.val+i.2.1.val<p.N) :
    interMemory M D p head input m "keyGradient"
        (head*p.N*p.D+(reverseRow p n.val s.val+i.2.1.val)*p.D+i.1.val)=
      if n.val=0 then m "keyGradient" (head*p.N*p.D+(reverseRow p n.val s.val+i.2.1.val)*p.D+i.1.val)
      else .mk .real (cast M (add M
        (dot D (keyValueCarry M D p head input n.val) (subVT M p head (reverseRow p n.val s.val) m) i)
        ((m "keyGradient" (head*p.N*p.D+(reverseRow p n.val s.val+i.2.1.val)*p.D+i.1.val)).read .real))) := by
  unfold interMemory
  rw [key_scan_read M D p htile head input _ n s i ht]
  rw [subVT_inputs M p head _ _ m (query_scan_inputs M D p head input m)]
  rw [query_scan_region M D p head input m "keyGradient" _ (by decide)]

theorem inter_value_read (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (htile : p.NC*p.CB=p.BT) (head : Nat) (input m : Memory α) (n : Fin p.NT) (s : Fin p.NC)
    (i : TileIndex [p.CB,p.E]) (ht : reverseRow p n.val s.val+i.1.val<p.N) :
    interMemory M D p head input m "valueGradient"
        (head*p.N*p.E+(reverseRow p n.val s.val+i.1.val)*p.E+i.2.1.val)=
      if n.val=0 then m "valueGradient" (head*p.N*p.E+(reverseRow p n.val s.val+i.1.val)*p.E+i.2.1.val)
      else .mk .real (cast M (add M
        (dot D (subK M p head (reverseRow p n.val s.val) m) (keyValueCarry M D p head input n.val) i)
        ((m "valueGradient" (head*p.N*p.E+(reverseRow p n.val s.val+i.1.val)*p.E+i.2.1.val)).read .real))) := by
  unfold interMemory
  rw [value_scan_read M D p htile head input _ n s i ht]
  rw [subK_inputs M p head _ _ m (query_scan_inputs M D p head input m)]
  rw [query_scan_region M D p head input m "valueGradient" _ (by decide)]

#axiomsClean inter_inputs
#axiomsClean inter_query_read
#axiomsClean inter_key_read
#axiomsClean inter_value_read
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.Current
