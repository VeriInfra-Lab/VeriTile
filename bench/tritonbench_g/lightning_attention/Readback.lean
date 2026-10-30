import bench.tritonbench_g.lightning_attention.Frame
import bench.tritonbench_g.lightning_attention.CurrentReadback
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
open VeriTile.Bench.Repairs.ChunkDeltaFwd (scatter_readback foldRange_last)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

theorem writeOutput_read (M : Algebra α) (p : Parameters) (pid : PID) (block : Nat)
    (values : Matrix α p.BT p.BM) (m : Memory α) (i : TileIndex [p.BT,p.BM])
    (hi : fwdActive p block i) (hcolumns : ∀ j : Fin p.BM,pid.channel*p.BM+j.val<p.E) :
    writeOutput M p pid block values m "output" (fwdOffset p pid block i)=.mk .real (cast M (values i)) :=
  scatter_readback _ _ _ _ _ i hi (fun a b ha hb he =>
    (Geometry.fwd_collision p pid pid block block a b ha hb (hcolumns a.2.1) (hcolumns b.2.1) he).2.2)

theorem fwdMemory_read (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input m : Memory α) (block : Nat) (i : TileIndex [p.BT,p.BM])
    (hb : block<p.NT) (hi : fwdActive p block i)
    (hcolumns : ∀ j : Fin p.BM,pid.channel*p.BM+j.val<p.E) :
    fwdMemory M D p pid input m "output" (fwdOffset p pid block i)=
      .mk .real (cast M (fwdOutput M D p pid input block (fwdCarry M D p pid input block) i)) := by
  unfold fwdMemory
  apply foldRange_last _ (fun mem : Memory α => mem "output" (fwdOffset p pid block i)) p.NT block m _ hb
  · intro mem
    exact writeOutput_read _ _ _ _ _ _ _ hi hcolumns
  · intro other ho hb' mem
    apply scatter_frame
    right
    intro j _ hj he
    have eq := (Geometry.fwd_collision p pid pid block other i j hi hj (hcolumns i.2.1) (hcolumns j.2.1) he).2.1
    omega

theorem intra_query_read (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head block : Nat)
    (m : Memory α) (i : TileIndex [p.BT,p.D]) (hi : block*p.BT+i.1.val<p.N) :
    intraMemory M D p head block m "queryGradient" (head*p.N*p.D+(block*p.BT+i.1.val)*p.D+i.2.1.val)=
      .mk .real (cast M (intraDQ M D p head block m i)) := by
  unfold intraMemory
  rw [writeRows_frame _ _ _ _ _ _ _ _ _ _ _ (Or.inl (by decide)),
    writeTrans_frame _ _ _ _ _ _ _ _ _ _ _ (Or.inl (by decide))]
  exact writeRows_read _ _ _ _ _ _ _ _ _ i hi

theorem intra_key_read (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head block : Nat)
    (m : Memory α) (i : TileIndex [p.D,p.BT]) (hi : block*p.BT+i.2.1.val<p.N) :
    intraMemory M D p head block m "keyGradient" (head*p.N*p.D+(block*p.BT+i.2.1.val)*p.D+i.1.val)=
      .mk .real (cast M (intraDKT M D p head block m i)) := by
  unfold intraMemory
  rw [writeRows_frame _ _ _ _ _ _ _ _ _ _ _ (Or.inl (by decide))]
  exact writeTrans_read _ _ _ _ _ _ _ _ _ i hi

theorem intra_value_read (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head block : Nat)
    (m : Memory α) (i : TileIndex [p.BT,p.E]) (hi : block*p.BT+i.1.val<p.N) :
    intraMemory M D p head block m "valueGradient" (head*p.N*p.E+(block*p.BT+i.1.val)*p.E+i.2.1.val)=
      .mk .real (cast M (intraDV M D p head block m i)) :=
  writeRows_read _ _ _ _ _ _ _ _ _ i hi
#axiomsClean fwdMemory_read
#axiomsClean intra_query_read
#axiomsClean intra_key_read
#axiomsClean intra_value_read
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly
