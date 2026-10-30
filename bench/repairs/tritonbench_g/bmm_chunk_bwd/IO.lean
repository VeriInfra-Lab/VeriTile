import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.IOSource
import VeriTile.Triton.Memory.PreparedAlgorithmProgramIO
import VeriTile.Triton.Memory.LogicalIO

namespace VeriTile.Bench.Repairs.BmmChunkBwd.PublicIO
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
noncomputable section

def jobs (h : HostCall) (c : Tuning) := HostGrid.programs (HostContract.grid h c)
def pid (h : HostCall) (c : Tuning) (i : Fin (jobs h c).length) := (jobs h c).get i
def view (h : HostCall) (c : Tuning) := RealViews.io Bool.false c.BM c.BN c.BCS (HostContract.parameters h)
def inputShape (h : HostCall) (c : Tuning) (i : Fin 3) : TileShape :=
  (jobs h c).length :: RealIO.inputShape c.BM c.BN h.chunk i
def outputShape (h : HostCall) (c : Tuning) : TileShape := [(jobs h c).length,c.BM,c.BN]

def inputWindow (h : HostCall) (c : Tuning) (i : Fin 3) (j : TileIndex (inputShape h c i)) :=
  (view h c).inputWindow i (pid h c j.1).tile (pid h c j.1).batch (pid h c j.1).chunkHead j.2
def inputMask (h : HostCall) (c : Tuning) (i : Fin 3) (j : TileIndex (inputShape h c i)) : Prop :=
  (view h c).inputMask i (pid h c j.1).tile (pid h c j.1).batch (pid h c j.1).chunkHead j.2
def outputWindow (h : HostCall) (c : Tuning) (j : TileIndex (outputShape h c)) :=
  HostGrid.address c.BM c.BN (HostContract.parameters h) (pid h c j.1) j.2
def outputMask (h : HostCall) (c : Tuning) (j : TileIndex (outputShape h c)) : Prop :=
  HostGrid.active c.BM c.BN (HostContract.parameters h) (pid h c j.1) j.2

/-- Three finite FP32 input-access arrays (Dout, A, optional residual) and the
full strided output. Inputs may alias each other. Dout's row mask uses chunk
size, whereas residual/store use the shortened last-chunk length. There is no
global scratch. Real interpretation erases arithmetic precision explicitly. -/
def memory (h : HostCall) (c : Tuning) : AlgorithmMemoryIO where
  nIn := 3
  nOut := 1
  nScr := 0
  bufs := [h.Dout,h.A,h.Residual,h.Out]
  ity := fun _ => .float
  iarity := fun i => (TileShape.allIndices (inputShape h c i)).length
  ibuf := (view h c).inputRegion
  oty := fun _ => .float
  oarity := fun _ => (TileShape.allIndices (outputShape h c)).length
  obuf := fun _ => h.Out
  obuf_mem := by simp
  sarity := Fin.elim0
  sbuf := Fin.elim0
  iwin := fun i _ _ _ _ j => inputWindow h c i ((TileShape.allIndices _).get j)
  imask := fun i _ _ _ _ j => inputMask h c i ((TileShape.allIndices _).get j)
  owin := fun _ _ _ _ _ j => outputWindow h c ((TileShape.allIndices _).get j)
  omask := fun _ _ _ _ _ j => outputMask h c ((TileShape.allIndices _).get j)
  swin := fun i => Fin.elim0 i
  smask := fun i => Fin.elim0 i

def source (h : HostCall) (c : Tuning) : AlgorithmSource where
  kernel := (Kernels.fp32Source h c).eraseDType
  projection := by rw [source_real]; exact (view h c).projection

/-- Exact original launch ABI: all three program and grid dimensions, fresh
private registers and zero undef for each CTA; previous memory is carried. -/
def invocation (h : HostCall) (c : Tuning) (q : HostGrid.Pid) : PreparedAlgorithmInvocation where
  source := source h c
  launch := {
    pids := fun ax => if ax=0 then q.tile else if ax=1 then q.batch else if ax=2 then q.chunkHead else 0
    numPids := some (fun ax => if ax=0 then (HostContract.grid h c).tiles
      else if ax=1 then (HostContract.grid h c).batches
      else if ax=2 then (HostContract.grid h c).chunkHeads else 1) }
  clearRegisters := Bool.true
  zeroUndef := Bool.true

def realIO (h : HostCall) (c : Tuning) : PreparedAlgorithmProgramIO where
  toAlgorithmMemoryIO := memory h c
  schedule := fun _ => (jobs h c).map (invocation h c)

def inputs (h : HostCall) (c : Tuning) (vals : (realIO h c).Ctx)
    (q : Fin (jobs h c).length) : RealInputs.Inputs c.BM c.BN h.chunk where
  dout := fun i k => vals (0 : Fin 3) (tilePos (inputShape h c 0) (q,i,k,()))
  a := fun k j => vals (1 : Fin 3) (tilePos (inputShape h c 1) (q,k,j,()))
  residual := fun i j => vals (2 : Fin 3) (tilePos (inputShape h c 2) (q,i,j,()))

/-- Independent finite sum of Doutᵀ times A, plus the optional residual.
Neither source execution nor output-memory reads occur in this formula. -/
def mathematical (h : HostCall) (c : Tuning) (vals : (realIO h c).Ctx)
    (_ : Fin 1) (j : Fin (TileShape.allIndices (outputShape h c)).length) : ℝ :=
  let lane := (TileShape.allIndices (outputShape h c)).get j
  let xs := inputs h c vals lane.1
  (∑ k : Fin h.chunk,
    if k.val < min h.chunk (h.seqlen-(pid h c lane.1).chunkHead/h.groupCount*h.chunk)
    then xs.dout lane.2.1 k * xs.a k lane.2.2.1 else 0) +
    if h.hasResidual then xs.residual lane.2.1 lane.2.2.1 else 0

end
end VeriTile.Bench.Repairs.BmmChunkBwd.PublicIO

namespace VeriTile.Bench.Repairs.BmmChunkBwd.PublicIO
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
noncomputable section

structure LogicalInputs (h : HostCall) (c : Tuning) where
  dout : Fin (jobs h c).length → Fin c.BM → Fin h.chunk → ℝ
  a : Fin (jobs h c).length → Fin h.chunk → Fin c.BN → ℝ
  residual : Fin (jobs h c).length → Fin c.BM → Fin c.BN → ℝ

def logicalInputsEquiv (h : HostCall) (c : Tuning) : (realIO h c).Ctx ≃ LogicalInputs h c where
  toFun := fun vals => ⟨fun q i k => (inputs h c vals q).dout i k,
    fun q k j => (inputs h c vals q).a k j, fun q i j => (inputs h c vals q).residual i j⟩
  invFun := fun xs => Fin.cases
    (fun j => let t := (TileShape.allIndices (inputShape h c 0)).get j; xs.dout t.1 t.2.1 t.2.2.1)
    (Fin.cases
      (fun j => let t := (TileShape.allIndices (inputShape h c 1)).get j; xs.a t.1 t.2.1 t.2.2.1)
      (Fin.cases
        (fun j => let t := (TileShape.allIndices (inputShape h c 2)).get j; xs.residual t.1 t.2.1 t.2.2.1)
        (fun i => i.elim0)))
  left_inv := by
    intro vals
    funext i j
    fin_cases i
    all_goals change vals _ (tilePos _ ((TileShape.allIndices _).get j)) = vals _ j
    all_goals rw [tilePos_get]
  right_inv := by
    intro xs
    cases xs with
    | mk dout a residual =>
      change LogicalInputs.mk
        (fun q i k => dout ((TileShape.allIndices (inputShape h c 0)).get (tilePos (inputShape h c 0) (q,i,k,()))).1
          ((TileShape.allIndices (inputShape h c 0)).get (tilePos (inputShape h c 0) (q,i,k,()))).2.1
          ((TileShape.allIndices (inputShape h c 0)).get (tilePos (inputShape h c 0) (q,i,k,()))).2.2.1)
        (fun q k j => a ((TileShape.allIndices (inputShape h c 1)).get (tilePos (inputShape h c 1) (q,k,j,()))).1
          ((TileShape.allIndices (inputShape h c 1)).get (tilePos (inputShape h c 1) (q,k,j,()))).2.1
          ((TileShape.allIndices (inputShape h c 1)).get (tilePos (inputShape h c 1) (q,k,j,()))).2.2.1)
        (fun q i j => residual ((TileShape.allIndices (inputShape h c 2)).get (tilePos (inputShape h c 2) (q,i,j,()))).1
          ((TileShape.allIndices (inputShape h c 2)).get (tilePos (inputShape h c 2) (q,i,j,()))).2.1
          ((TileShape.allIndices (inputShape h c 2)).get (tilePos (inputShape h c 2) (q,i,j,()))).2.2.1) = _
      simp only [get_tilePos]

def logicalOutputsEquiv (h : HostCall) (c : Tuning) :
    ((o : Fin 1) → Fin (TileShape.allIndices (outputShape h c)).length → ℝ) ≃
      (Fin (jobs h c).length → Fin c.BM → Fin c.BN → ℝ) where
  toFun := fun vals q i j => vals 0 (tilePos (outputShape h c) (q,i,j,()))
  invFun := fun f _ j => let t := (TileShape.allIndices (outputShape h c)).get j; f t.1 t.2.1 t.2.2.1
  left_inv := by
    intro vals
    funext o j
    have z : o=0 := Fin.eq_zero o
    subst o
    change vals 0 (tilePos _ ((TileShape.allIndices _).get j)) = vals 0 j
    rw [tilePos_get]
  right_inv := by
    intro f
    funext q i j
    change f ((TileShape.allIndices (outputShape h c)).get (tilePos (outputShape h c) (q,i,j,()))).1
      ((TileShape.allIndices (outputShape h c)).get (tilePos (outputShape h c) (q,i,j,()))).2.1
      ((TileShape.allIndices (outputShape h c)).get (tilePos (outputShape h c) (q,i,j,()))).2.2.1 = _
    rw [get_tilePos]

def chunkLength (h : HostCall) (c : Tuning) (q : Fin (jobs h c).length) : Nat :=
  min h.chunk (h.seqlen-(pid h c q).chunkHead/h.groupCount*h.chunk)

def logicalIO (h : HostCall) (c : Tuning) :
    LogicalIO (realIO h c).Ctx
      ((o : Fin 1) → Fin (TileShape.allIndices (outputShape h c)).length → ℝ)
      (LogicalInputs h c) (Fin (jobs h c).length → Fin c.BM → Fin c.BN → ℝ)
      (realIO h c).Implements where
  inputs := logicalInputsEquiv h c
  outputs := logicalOutputsEquiv h c

end
end VeriTile.Bench.Repairs.BmmChunkBwd.PublicIO
