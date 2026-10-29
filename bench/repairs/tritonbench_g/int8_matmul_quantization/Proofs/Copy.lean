import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.QuantReference

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization
noncomputable section
set_option maxHeartbeats 5000000

def copyCode (source destination : α → Address) : List α → Code I F H Unit
  | [] => pure ()
  | x::xs => do
      let byte ← emit (.read8 (source x))
      let _ ← emit (.store8 (destination x) byte)
      copyCode source destination xs
theorem eval_copyCode (a : Ops I F H) (source destination : α → Address)
    (xs : List α) (m before : Memory F H) (same : ∀ x,m (source x)=before (source x))
    (separate : ∀ x y,source x≠destination y) :
    (copyCode source destination xs).eval a m=
      ((),writeAll destination (fun x => .inl (read8 before (source x))) xs m) := by
  induction xs generalizing m with
  | nil => rfl
  | cons x xs ih =>
    have load : read8 m (source x)=read8 before (source x) := by unfold read8 Int8DequantMatmul.read8; rw [same x]
    simp only [copyCode,bind,eval_bind,eval_emit,Event.eval,Prod.fst,Prod.snd,load]
    have next : ∀ y,write8 m (destination x) (read8 before (source x)) (source y)=before (source y) := by
      intro y
      rw [write8,if_neg (separate y x),same y]
    rw [ih _ next]
    rfl
def quantCells (G K : Nat) : List (QuantCell G K) :=
  (List.finRange G).flatMap (fun g => (List.finRange K).map (fun k => (g,k)))
theorem quantCells_present (x : QuantCell G K) : x∈quantCells G K := by
  simp [quantCells]
def copyRun (a : Ops I F H) (source destination : QuantBuffers) (G K : Nat) (m : Memory F H) :=
  ((copyCode source.dataAddress destination.dataAddress (quantCells G K)).eval a m).2
theorem copyRun_at (a : Ops I F H) (source destination : QuantBuffers)
    (separate : source.output≠destination.output) (m : Memory F H) (x : QuantCell G K) :
    copyRun a source destination G K m (destination.dataAddress x)=.inl (read8 m (source.dataAddress x)) := by
  have sep : ∀ x y : QuantCell G K,source.dataAddress x≠destination.dataAddress y := by
    intro x y eq
    exact separate (congrArg Prod.fst eq)
  unfold copyRun
  rw [eval_copyCode a _ _ _ m m (fun _ => rfl) sep]
  exact writeAll_at _ _ destination.data_injective _ _ x (quantCells_present x)
theorem copyRun_frame (a : Ops I F H) (source destination : QuantBuffers)
    (separate : source.output≠destination.output) (m : Memory F H) (p : Address)
    (outside : ∀ x : QuantCell G K,p≠destination.dataAddress x) : copyRun a source destination G K m p=m p := by
  have sep : ∀ x y : QuantCell G K,source.dataAddress x≠destination.dataAddress y := by
    intro x y eq
    exact separate (congrArg Prod.fst eq)
  unfold copyRun
  rw [eval_copyCode a _ _ _ m m (fun _ => rfl) sep]
  exact writeAll_frame _ _ _ m p (fun x _ => outside x)

/-- Axis 0 quantizes into row-major intermediate storage, then the actual
t().contiguous().t() transports each int8 byte into column-major storage.
Negative axis -2 does not enter this literal axis==0 branch. -/
def axis0Helper (a : Ops I F H) (temporary final : QuantBuffers)
    (chosen : QuantInvocation G K) (m : Memory F H) :=
  copyRun a temporary final G K (quantRun a temporary chosen m)
theorem axis0Helper_at (a : Ops I F H) (temporary final : QuantBuffers)
    (sep : temporary.Separate) (different : temporary.output≠final.output)
    (chosen : QuantInvocation G K) (m : Memory F H) (x : QuantCell G K) :
    axis0Helper a temporary final chosen m (final.dataAddress x)=
      .inl (quantValue a temporary chosen.plan m x) := by
  rw [axis0Helper,copyRun_at a temporary final different]
  have h := quantRun_at a temporary sep chosen m (.inl x)
  change quantRun a temporary chosen m (temporary.dataAddress x)=.inl _ at h
  simp only [read8,Int8DequantMatmul.read8,h]

#axiomsClean eval_copyCode
#axiomsClean copyRun_at
#axiomsClean copyRun_frame
#axiomsClean axis0Helper_at
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization
