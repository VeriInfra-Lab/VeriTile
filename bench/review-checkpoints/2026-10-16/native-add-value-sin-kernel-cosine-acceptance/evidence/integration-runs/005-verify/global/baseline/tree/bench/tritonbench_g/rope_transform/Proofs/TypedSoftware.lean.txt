import bench.tritonbench_g.rope_transform.Proofs.TypedFrame
namespace VeriTile.Bench.TritonBenchG.RopeTransform.Typed
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def roundedLane [Inhabited (FP.Value .fp32)] (cfg : FP.Config)
    (p : Parameters) (s : State (FP.Value .fp32)) (r : RegionName)
    (stride : Nat) (second : Bool) {H W : Nat} (i : TileIndex [H,W]) : FP.Value .fp32 :=
  let x := readFloat .fp32 s.mem r (address (s.pids 0) stride p.headDim (p.headDim/2) second i)
  let y := readFloat .fp32 s.mem r (address (s.pids 0) stride p.headDim (p.headDim/2) (!second) i)
  let c := readFloat .fp32 s.mem p.cos (s.pids 0 % p.length * p.cosStride + i.2.1.val)
  let sn := readFloat .fp32 s.mem p.sin (s.pids 0 % p.length * p.sinStride + i.2.1.val)
  if second == p.backward then FP.sub cfg (FP.mul cfg x c) (FP.mul cfg y sn)
  else FP.add cfg (FP.mul cfg x c) (FP.mul cfg y sn)

def roundedMemory [Inhabited (FP.Value .fp32)] (cfg : FP.Config)
    (p : Parameters) (s : State (FP.Value .fp32)) : Memory (FP.Value .fp32) :=
  let write (mem : Memory (FP.Value .fp32)) (r : RegionName) (stride heads H : Nat) (second : Bool) :=
    scatter mem r (s.pids 0) stride heads p.headDim second H (p.paddedHeadDim/2)
      (fun i => .mk .fp32 (roundedLane cfg p s r stride second i));
  let q0 := write s.mem p.q p.qStride p.qHeads p.paddedQHeads Bool.false;
  let q1 := write q0 p.q p.qStride p.qHeads p.paddedQHeads Bool.true;
  let k0 := write q1 p.k p.kStride p.kHeads p.paddedKHeads Bool.false;
  write k0 p.k p.kStride p.kHeads p.paddedKHeads Bool.true

theorem lane_software [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (p : Parameters)
    (s : State (FP.Value .fp32)) (r : RegionName) (stride heads : Nat)
    (second : Bool) {H W : Nat} (i : TileIndex [H,W])
    (hi : active heads (p.headDim/2) i) :
    laneResult (FP.Software.scalarFP32 rest cfg) .fp32 .fp32 p s r stride heads second i =
      roundedLane cfg p s r stride second i := by
  have hcol : i.2.1.val < p.headDim / 2 := by
    have h : i.1.val < heads ∧ i.2.1.val < p.headDim / 2 := by simpa [active] using hi
    exact h.2
  cases second <;> cases hb : p.backward <;>
    simp [laneResult, component, coefficient, hi, hcol, roundedLane,
      FP.Software.scalarFP32, FP.Software.fp32Context, hb]

theorem scatter_congr_active {α : Type} (mem : Memory α) (r : RegionName)
    (pid stride heads hd : Nat) (second : Bool) (H W : Nat)
    (v w : TileIndex [H,W] → Cell α)
    (h : ∀ i, active heads (hd/2) i → v i = w i) :
    scatter mem r pid stride heads hd second H W v =
      scatter mem r pid stride heads hd second H W w := by
  unfold scatter
  congr 1
  funext current i
  by_cases hi : active heads (hd/2) i
  · simp only [hi, ↓reduceIte, h i hi]
  · simp only [hi, Bool.false_eq_true, ↓reduceIte]

theorem reference_software [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (p : Parameters)
    (s : State (FP.Value .fp32)) :
    referenceMemory (FP.Software.scalarFP32 rest cfg) .fp32 .fp32 p s = roundedMemory cfg p s := by
  have step (mem : Memory (FP.Value .fp32)) (r : RegionName) (stride heads H : Nat) (second : Bool) :
      scatter mem r (s.pids 0) stride heads p.headDim second H (p.paddedHeadDim/2)
        (fun i => .mk .fp32 (laneResult (FP.Software.scalarFP32 rest cfg) .fp32 .fp32 p s r stride heads second i)) =
      scatter mem r (s.pids 0) stride heads p.headDim second H (p.paddedHeadDim/2)
        (fun i => .mk .fp32 (roundedLane cfg p s r stride second i)) := by
    apply scatter_congr_active
    intro i hi
    rw [lane_software rest cfg p s r stride heads second i hi]
  dsimp only [referenceMemory, roundedMemory, FloatDType.toTileDType, ofFloat, id]
  simp only [step]

#axiomsClean lane_software
#axiomsClean scatter_congr_active
#axiomsClean reference_software
end
end VeriTile.Bench.TritonBenchG.RopeTransform.Typed
