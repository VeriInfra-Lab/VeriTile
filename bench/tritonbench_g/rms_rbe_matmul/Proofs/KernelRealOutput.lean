import bench.tritonbench_g.rms_rbe_matmul.Proofs.KernelReal

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.KernelReal
open VeriTile.Triton FP.Structural
open KernelSource (Parameters)
open KernelGraph (Input Tile)
noncomputable section
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

theorem phase_real (decode : Float8E5Bits → ℝ) (p : Parameters) (t : Tile)
    (i : TileIndex [16,64]) :
    KernelGraph.phase (operations decode) p t i=
      KernelMath.angle p.headDim p.start (t.row*16+i.1.val) (t.column*64+i.2.1.val) := by
  have heven : t.column*64+KernelGraph.evenLane i.2.1=
      (t.column*64+i.2.1.val)-(t.column*64+i.2.1.val)%2 := by
    change t.column*64+i.2.1.val/2*2=_
    omega
  simp only [KernelGraph.phase,operations,KernelMath.angle,KernelMath.theta,Int.cast_add,
    Int.cast_natCast,heven,id_eq]

/-- Mathematical RMS projection followed by optional adjacent-pair rotation.
All input representation casts are erased only in this exact-Real semantics. -/
theorem output_real (decode : Float8E5Bits → ℝ) (zero : decode (Float8E5Bits.ofInt 0)=0)
    (fp8 : Bool) (p : Parameters) (v : Input ℝ) (t : Tile)
    (i : TileIndex [16,64]) (hm : t.row*16+i.1.val<p.M) (hn : t.column*64+i.2.1.val<p.N)
    (evenN : p.N%2=0) :
    KernelGraph.output (operations decode) fp8 p v t i=
      KernelMath.output p.K p.headDim p.start p.rotary
        (logical decode fp8 v t.batch (t.row*16+i.1.val))
        (t.row*16+i.1.val) (t.column*64+i.2.1.val) := by
  have hev : t.column*64+KernelGraph.evenLane i.2.1=
      (t.column*64+i.2.1.val)-(t.column*64+i.2.1.val)%2 := by
    change t.column*64+i.2.1.val/2*2=_
    omega
  have hod : t.column*64+KernelGraph.oddLane i.2.1=
      (t.column*64+i.2.1.val)-(t.column*64+i.2.1.val)%2+1 := by
    change t.column*64+(i.2.1.val/2*2+1)=_
    omega
  have evenValid : t.column*64+KernelGraph.evenLane i.2.1<p.N := by rw [hev]; omega
  have oddValid : t.column*64+KernelGraph.oddLane i.2.1<p.N := by rw [hod]; omega
  have parity : i.2.1.val%2=(t.column*64+i.2.1.val)%2 := by omega
  cases hr : p.rotary
  · simpa only [KernelGraph.output,KernelMath.output,hr,Bool.false_eq_true,if_false,operations]
      using projection_real decode zero fp8 p v t i hm hn
  · have eu := projection_real decode zero fp8 p v t
      (i.1,KernelGraph.evenLane i.2.1,PUnit.unit) hm evenValid
    have ev := projection_real decode zero fp8 p v t
      (i.1,KernelGraph.oddLane i.2.1,PUnit.unit) hm oddValid
    simp only [KernelGraph.output,KernelMath.output,hr,if_true]
    rw [eu,ev,phase_real]
    simp only [operations,hev,hod,parity,id_eq]

#axiomsClean output_real
#specNonCircular KernelMath.output avoiding [FP.Structural.exec,KernelGraph.output]
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.KernelReal
