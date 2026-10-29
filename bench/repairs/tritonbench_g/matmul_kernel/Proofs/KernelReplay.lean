import bench.repairs.tritonbench_g.matmul_kernel.Proofs.KernelResult
namespace VeriTile.Bench.Repairs.MatmulKernel.KernelReplay
open VeriTile.Triton FP.Structural KernelTensor KernelEntries KernelSource
noncomputable section
set_option maxHeartbeats 5000000
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def programBytes (M : Algebra Word) (D : DotAcc Word) (p : Prepared) (s : State Word)
    (pid : Program) (heap : KernelStorage.Heap) : KernelStorage.Heap :=
  (TileShape.allIndices [p.config.BM,p.config.BN]).foldl (fun heap i=>
    if active p.config (atProgram s pid) i then KernelStorage.write heap
      (p.metadata.pointerC+4*cOffset p.config (atProgram s pid) i)
      (outputValue M D p.config (atProgram s pid) i) else heap) heap

/-- Replay the actual ordered source stores at their physical byte addresses.
The source state executes the complete shared kernel for each program; the
heap changes only through the explicit four-byte store operations. -/
def replay (M : Algebra Word) (D : DotAcc Word) (p : Prepared) :
    List Program → State Word → KernelStorage.Heap → Option (State Word × KernelStorage.Heap)
  | [],s,heap=>some (s,heap)
  | pid::tail,s,heap=>do
      let next←FP.Structural.exec (backend M D) (KernelSource.source p.config) (atProgram s pid)
      replay M D p tail next (programBytes M D p s pid heap)

theorem projection (M : Algebra Word) (D : DotAcc Word) (p : Prepared) (before : KernelStorage.Heap)
    (order : List Program) (s : State Word) :
    replay M D p order s (nativeHeap p before s)=
      (KernelSource.execute M D p.config order s).map (fun final=>(final,nativeHeap p before final)) := by
  induction order generalizing s with
  | nil=>rfl
  | cons pid tail ih=>
    obtain ⟨t,he,hm,_⟩:=KernelSource.correctness M D p.config (atProgram s pid)
      (tile_injective p.config p.valid _)
    have image : nativeHeap p before t=programBytes M D p s pid (nativeHeap p before s) := by
      unfold nativeHeap
      rw [hm]
      change nativeHeap p before (written M D p.config (atProgram s pid))=_
      rw [program_bytes]
      rfl
    simp only [replay,KernelSource.execute,he,Option.bind_some]
    change replay M D p tail t (programBytes M D p s pid (nativeHeap p before s))=_
    rw [←image,ih]

theorem native_replay (instructions : KernelNative.Instructions) (p : Prepared)
    (before : KernelStorage.Heap) (final : State Word)
    (run : KernelSource.launch KernelNative.scalar (KernelScalar.dotAcc (KernelNative.dot instructions p.config.opcode))
      p.config (initial p (KernelStorage.read before))=some final) :
    replay KernelNative.scalar (KernelScalar.dotAcc (KernelNative.dot instructions p.config.opcode)) p
      (programs p.config) (initial p (KernelStorage.read before)) before=some (final,nativeHeap p before final) := by
  have h:=projection KernelNative.scalar (KernelScalar.dotAcc (KernelNative.dot instructions p.config.opcode))
    p before (programs p.config) (initial p (KernelStorage.read before))
  rw [native_initial] at h
  change _=(KernelSource.launch _ _ _ _).map _ at h
  rw [run] at h
  exact h

#axiomsClean projection
#axiomsClean native_replay
end
end VeriTile.Bench.Repairs.MatmulKernel.KernelReplay
