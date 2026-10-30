import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.NativeKernel
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.InputBinding

namespace VeriTile.Bench.Repairs.LayerNormWelfold.NativeInputs
open VeriTile.Triton FP.Structural InputBinding
open VeriTile.Torch.FP32 (Word Tensor TensorLaws)
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.ofBits .fp32 0⟩

/-- The compiler sees physical leaf identity but cannot inspect its input word. -/
def graphRead (read : Nat → Word) (address : Nat) : NativeKernel.Term :=
  .leaf "physical" address (read address).bits

def logicalGraph (read : Nat → Word) (t : Tensor) (laws : TensorLaws t) (i : Nat) : NativeKernel.Term :=
  .leaf "physical" (t.storage.address (t.physicalIndex i))
    (VeriTile.Torch.FP32.logicalValue read t laws i).bits

def state (read : Nat → Word) (x w bias : Tensor) (scratch : State NativeKernel.Term) : State NativeKernel.Term :=
  { scratch with mem := fun q i =>
      if q = "X" then .mk .real (graphRead read (address x i))
      else if q = "W" then .mk .real (graphRead read (address w i))
      else if q = "Bias" then .mk .real (graphRead read (address bias i))
      else scratch.mem q i }

structure Observations (read : Nat → Word) (x w bias : Tensor)
    (lx : TensorLaws x) (lw : TensorLaws w) (lb : TensorLaws bias)
    (s : State NativeKernel.Term) (S D : Nat) : Prop where
  xValues : ∀ row, row < S → ∀ col, col < D →
    (s.mem "X" (col + D * row)).read .real = logicalGraph read x lx (col + D * row)
  wValues : ∀ col, col < D → (s.mem "W" col).read .real = logicalGraph read w lw col
  biasValues : ∀ col, col < D → (s.mem "Bias" col).read .real = logicalGraph read bias lb col
  xPointer : ∀ row, row < S → ∀ col, col < D →
    address x (col + D * row) = x.storage.address (x.physicalIndex (col + D * row)) ∧
    address x (col + D * row) < 2^64 ∧
    (x.dataPtr + BitVec.ofNat 64 (4 * (col + D * row))).toNat = address x (col + D * row)
  wPointer : ∀ col, col < D → address w col = w.storage.address (w.physicalIndex col) ∧ address w col < 2^64 ∧
    (w.dataPtr + BitVec.ofNat 64 (4 * col)).toNat = address w col
  biasPointer : ∀ col, col < D → address bias col = bias.storage.address (bias.physicalIndex col) ∧ address bias col < 2^64 ∧
    (bias.dataPtr + BitVec.ofNat 64 (4 * col)).toNat = address bias col

/-- A scratch value used only to instantiate the input-only observation lemma;
no kernel runs on this state and its cells cannot enter a result. -/
private def emptyRaw : State Word :=
  ⟨fun _ _ => .mk .nat 0, fun _ _ _ => none, fun _ => 0, fun _ => 0,
    fun d _ _ => defaultValue d⟩

theorem observed (read : Nat → Word) (x w bias : Tensor)
    (lx : TensorLaws x) (lw : TensorLaws w) (lb : TensorLaws bias)
    (mx mw mb : Admission.Metadata) (bx : Layout x mx) (bw : Layout w mw) (bb : Layout bias mb)
    (S D : Nat) (h : Admission.Contract mw mb mx S D) (scratch : State NativeKernel.Term) :
    Observations read x w bias lx lw lb (state read x w bias scratch) S D := by
  have obs := InputBinding.observed read id x w bias lx lw lb mx mw mb bx bw bb S D h emptyRaw
  refine ⟨?_, ?_, ?_, obs.xPointer, obs.wPointer, obs.biasPointer⟩
  · intro row hr col hc
    have word : read (address x (col + D * row)) = VeriTile.Torch.FP32.logicalValue read x lx (col + D * row) :=
      obs.xValues row hr col hc
    change graphRead read (address x (col + D * row)) = logicalGraph read x lx (col + D * row)
    simp only [graphRead, logicalGraph]
    rw [word, (obs.xPointer row hr col hc).1]
  · intro col hc
    have word : read (address w col) = VeriTile.Torch.FP32.logicalValue read w lw col := obs.wValues col hc
    change graphRead read (address w col) = logicalGraph read w lw col
    simp only [graphRead, logicalGraph]
    rw [word, (obs.wPointer col hc).1]
  · intro col hc
    have word : read (address bias col) = VeriTile.Torch.FP32.logicalValue read bias lb col := obs.biasValues col hc
    change graphRead read (address bias col) = logicalGraph read bias lb col
    simp only [graphRead, logicalGraph]
    rw [word, (obs.biasPointer col hc).1]

#axiomsClean observed
end
end VeriTile.Bench.Repairs.LayerNormWelfold.NativeInputs
