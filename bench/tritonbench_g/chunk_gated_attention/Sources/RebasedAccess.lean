import bench.tritonbench_g.chunk_gated_attention.Sources.Address
import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealHAlgebra
import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealHStore

/-! Evaluation of the actual repaired pointer expressions and all local loads.
Every lane is active by construction. The original pure numerical tile
algebra and store readback/frame lemmas remain imported unchanged. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedAccess
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealHMath RealHTiles
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def matrix (r : RegionName) (base nr nc sr sc : Nat) : Tile .blockPtr [nr,nc] :=
  RealHPointers.matrixPointer r base nr nc nr nc sr sc 0 0

def vector (r : RegionName) (base n stride : Nat) : Tile .blockPtr [n] :=
  RealHPointers.vectorPointer r base n n stride 0

def keyPointer (p : Layout) (l : Launch) (r : RegionName) (c : Nat) :=
  matrix r (l.head*p.keyHead+l.key*p.BK*p.keyFeature+c*p.BT*p.keyTime)
    p.BK p.BT p.keyFeature p.keyTime

def valuePointer (p : Layout) (l : Launch) (r : RegionName) (c : Nat) :=
  matrix r (l.head*p.valueHead+c*p.BT*p.valueTime+l.value*p.BV*p.valueFeature)
    p.BT p.BV p.valueTime p.valueFeature

def statePointer (p : Layout) (l : Launch) (r : RegionName) :=
  matrix r (l.head*p.K*p.V+l.key*p.BK*p.V+l.value*p.BV) p.BK p.BV p.V 1

def historyPointer (p : Layout) (l : Launch) (r : RegionName) (c : Nat) :=
  matrix r (l.head*p.historyHead+c*p.K*p.V+l.key*p.BK*p.historyKey+l.value*p.BV*p.historyValue)
    p.BK p.BV p.historyKey p.historyValue

def endpointPointer (p : Layout) (l : Launch) (r : RegionName)
    (c head time stride block index : Nat) :=
  vector r (l.head*head+(c*p.BT+p.BT-1)*time+index*block*stride) block stride

/-- Direct physical reads of staged tensors; no unspecified-lane oracle. -/
def loaded (p : Layout) (l : Launch) (a : Inputs) (c : Nat) : Loaded p where
  key e t := a.key (keyAddress p l (keyIndex p l e.val) (timeIndex p c t.val))
  value t j := a.value (valueAddress p l (timeIndex p c t.val) (valueIndex p l j.val))
  keyGate e t := a.gate (keyAddress p l (keyIndex p l e.val) (timeIndex p c t.val))
  valueGate t j := a.gate (valueAddress p l (timeIndex p c t.val) (valueIndex p l j.val))
  keyEnd e := a.gate (l.head*p.keyHead+(c*p.BT+p.BT-1)*p.keyTime+(keyIndex p l e.val)*p.keyFeature)
  valueEnd j := a.gate (l.head*p.valueHead+(c*p.BT+p.BT-1)*p.valueTime+(valueIndex p l j.val)*p.valueFeature)

theorem matrix_load (r : RegionName) (u : BlockState) (name : RegName)
    (base nr nc sr sc : Nat)
    (hp : u.regs .blockPtr [nr,nc] name = some (matrix r base nr nc sr sc)) :
    evalOp (RealHProgram.load2 nr nc name) u =
      some (Tile.mat (fun (i : Fin nr) (j : Fin nc) => some (u.readMem r (base+i.val*sr+j.val*sc)))) := by
  rw [RealHPointers.load2_eval r u name base nr nc nr nc sr sc 0 0 hp]
  congr 1
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp only [Tile.mat, Nat.zero_add, i.isLt, j.isLt, and_self, if_true]
  rfl

theorem vector_load (r : RegionName) (u : BlockState) (name : RegName)
    (base n stride : Nat)
    (hp : u.regs .blockPtr [n] name = some (vector r base n stride)) :
    evalOp (RealHProgram.load1 n name) u =
      some (Tile.vec (fun (i : Fin n) => some (u.readMem r (base+i.val*stride)))) := by
  rw [RealHPointers.load1_eval r u name base n n stride 0 hp]
  congr 1
  apply Tile.ext
  rintro ⟨i,⟨⟩⟩
  simp only [Tile.vec, Nat.zero_add, i.isLt, if_true]
  rfl

theorem key_load (p : Layout) (l : Launch) (r : RegionName) (a : Nat → ℝ)
    (c : Nat) (name : RegName) (u : BlockState) (hm : ∀ off, u.readMem r off = a off)
    (hp : u.regs .blockPtr [p.BK,p.BT] name = some (keyPointer p l r c)) :
    evalOp (RealHProgram.load2 p.BK p.BT name) u = some (Tile.mat (fun e t =>
      some (a (keyAddress p l (keyIndex p l e.val) (timeIndex p c t.val))))) := by
  rw [matrix_load r u name _ _ _ _ _ hp]
  congr 1
  apply Tile.ext
  rintro ⟨e,t,⟨⟩⟩
  simp only [Tile.mat,hm,keyAddress,timeIndex,keyIndex]
  congr 2 <;> ring

theorem value_load (p : Layout) (l : Launch) (r : RegionName) (a : Nat → ℝ)
    (c : Nat) (name : RegName) (u : BlockState) (hm : ∀ off, u.readMem r off = a off)
    (hp : u.regs .blockPtr [p.BT,p.BV] name = some (valuePointer p l r c)) :
    evalOp (RealHProgram.load2 p.BT p.BV name) u = some (Tile.mat (fun t j =>
      some (a (valueAddress p l (timeIndex p c t.val) (valueIndex p l j.val))))) := by
  rw [matrix_load r u name _ _ _ _ _ hp]
  congr 1
  apply Tile.ext
  rintro ⟨t,j,⟨⟩⟩
  simp only [Tile.mat,hm,valueAddress,timeIndex,valueIndex]
  congr 2 <;> ring

theorem endpoint_load (p : Layout) (l : Launch) (r : RegionName) (a : Nat → ℝ)
    (c head time stride block index : Nat) (name : RegName) (u : BlockState)
    (hm : ∀ off, u.readMem r off = a off)
    (hp : u.regs .blockPtr [block] name = some (endpointPointer p l r c head time stride block index)) :
    evalOp (RealHProgram.load1 block name) u = some (Tile.vec (fun i : Fin block =>
      some (a (l.head*head+(c*p.BT+p.BT-1)*time+(index*block+i.val)*stride)))) := by
  rw [vector_load r u name _ _ _ hp]
  congr 1
  apply Tile.ext
  rintro ⟨i,⟨⟩⟩
  simp only [Tile.vec,hm]
  congr 2 <;> ring

structure Full (p : Layout) (l : Launch) (c : Nat) : Prop where
  key : l.key*p.BK+p.BK ≤ p.K
  value : l.value*p.BV+p.BV ≤ p.V
  time : c*p.BT+p.BT ≤ p.T
  positiveTime : 0 < p.BT
  keyTime : p.keyTime = p.K*p.keyFeature
  valueTime : p.valueTime = p.V*p.valueFeature

/-- The old mathematical helper is reused only after all its checked lanes
are proved active and its flattened endpoint agrees with the actual stride. -/
theorem loaded_eq (p : Layout) (l : Launch) (r : RealHProgram.Regions) (a : Inputs)
    (padding : RegionName → Nat → ℝ) (c : Nat) (h : Full p l c) :
    loaded p l a c = RealHTiles.loaded p l r a padding c := by
  have hk (e : Fin p.BK) : keyIndex p l e.val < p.K := by
    unfold keyIndex; have := h.key; omega
  have hv (j : Fin p.BV) : valueIndex p l j.val < p.V := by
    unfold valueIndex; have := h.value; omega
  have ht (t : Fin p.BT) : timeIndex p c t.val < p.T := by
    unfold timeIndex; have := h.time; omega
  have hend : c*p.BT+p.BT-1 < p.T := by
    have := h.time; have := h.positiveTime; omega
  have hke (e : Fin p.BK) : (c*p.BT+p.BT-1)*p.K+l.key*p.BK+e.val < p.T*p.K := by
    have := Address.flatten2_bounded hend (hk e)
    simpa [keyIndex,Nat.add_assoc] using this
  have hve (j : Fin p.BV) : (c*p.BT+p.BT-1)*p.V+l.value*p.BV+j.val < p.T*p.V := by
    have := Address.flatten2_bounded hend (hv j)
    simpa [valueIndex,Nat.add_assoc] using this
  unfold loaded RealHTiles.loaded
  congr 1
  · funext e t; simp [loaded,RealHTiles.loaded,keyData,hk e,ht t]
  · funext t j; simp [loaded,RealHTiles.loaded,valueData,ht t,hv j]
  · funext e t; simp [loaded,RealHTiles.loaded,keyData,hk e,ht t]
  · funext t j; simp [loaded,RealHTiles.loaded,valueData,ht t,hv j]
  · funext e
    simp only [loaded,RealHTiles.loaded,endpointData,hke e,if_true,h.keyTime,keyIndex]
    congr 1 <;> ring
  · funext j
    simp only [loaded,RealHTiles.loaded,endpointData,hve j,if_true,h.valueTime,valueIndex]
    congr 1 <;> ring

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedAccess
