import bench.repairs.tritonbench_g.matmul_kernel.Proofs.KernelReal
import bench.repairs.tritonbench_g.matmul_kernel.Proofs.KernelNative
import bench.repairs.tritonbench_g.matmul_kernel.Proofs.KernelTensor
import bench.repairs.tritonbench_g.matmul_kernel.Proofs.KernelStorage
namespace VeriTile.Bench.Repairs.MatmulKernel.KernelEntries
open VeriTile.Triton FP.Structural KernelSource KernelScalar KernelTensor
noncomputable section
set_option maxHeartbeats 300000
set_option linter.unusedSimpArgs false
set_option maxRecDepth 32000
abbrev Word := FP.Value .fp32
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

theorem aRows (p : Prepared) : p.metadata.config.A.rows=p.config.M := p.valid.aShape.1
theorem aCols (p : Prepared) : p.metadata.config.A.cols=p.config.K := p.valid.aShape.2
theorem bRows (p : Prepared) : p.metadata.config.B.rows=p.config.K := p.valid.bShape.1
theorem bCols (p : Prepared) : p.metadata.config.B.cols=p.config.N := p.valid.bShape.2
theorem cRows (p : Prepared) : p.metadata.config.C.rows=p.config.M := p.valid.cShape.1
theorem cCols (p : Prepared) : p.metadata.config.C.cols=p.config.N := p.valid.cShape.2

def base (p : Prepared) (region : RegionName) : Nat :=
  if region="A" then p.metadata.pointerA else if region="B" then p.metadata.pointerB else p.metadata.pointerC

def initial {α : Type} [Inhabited α] (p : Prepared) (memory : Nat → α) : State α where
  mem region index:=.mk .real (memory (base p region+4*index))
  regs _ _ _:=none
  pids _:=0
  numPids dim:=if dim=0 then ceil p.config.M p.config.BM else if dim=1 then ceil p.config.N p.config.BN else 1
  undef dtype _ _:=defaultValue dtype

def realTarget (p : Prepared) (memory : Nat → ℝ) (r n : Nat) :=
  KernelReal.target p.config.K (realPoint memory p.a) (realPoint memory p.b) r n

def nativeTarget (instructions : KernelNative.Instructions) (p : Prepared)
    (heap : KernelStorage.Heap) (r n : Nat) :=
  KernelNative.target instructions p.config
    (nativePoint (KernelStorage.read heap) p.a) (nativePoint (KernelStorage.read heap) p.b) r n

/-- Reader through the actual destination Tensor's physical index and storage
address. The source-relative word index is derived from that address. -/
def realResult (p : Prepared) (final : State ℝ) (r n : Nat) : ℝ :=
  (final.mem "C" ((p.c.tensor.storage.address (p.c.tensor.physicalIndex (r*p.config.N+n))-
    p.metadata.pointerC)/4)).read .real

def nativeHeap (p : Prepared) (before : KernelStorage.Heap) (final : State Word) : KernelStorage.Heap :=
  KernelStorage.image p.metadata.pointerC p.config.C.capacity before final.mem

def nativeResult (p : Prepared) (before : KernelStorage.Heap) (final : State Word) (r n : Nat) : Word :=
  nativePoint (KernelStorage.read (nativeHeap p before final)) p.c r n

theorem c_address (p : Prepared) (w : Work) (hw : p.config.Active w) :
    p.metadata.pointerC+4*p.config.C.addr w.1 w.2=
      p.c.tensor.storage.address (p.c.tensor.physicalIndex (w.1*p.config.N+w.2)) := by
  simpa only [cCols p,p.valid.cShape.2] using address p.c w.1 w.2
    (by simpa only [cRows p,p.valid.cShape.1] using hw.1) (by simpa only [cCols p,p.valid.cShape.2] using hw.2)

theorem real_inputs (p : Prepared) (memory : Nat → ℝ) (w : Work) (hw : p.config.Active w) :
    KernelReal.target p.config.K
      (fun r k=>KernelSource.read ((initial p memory).mem "A" (p.config.A.addr r k)))
      (fun k n=>KernelSource.read ((initial p memory).mem "B" (p.config.B.addr k n))) w.1 w.2=
      realTarget p memory w.1 w.2 := by
  unfold KernelReal.target realTarget
  apply Finset.sum_congr rfl
  intro k hk
  have bound:=Finset.mem_range.mp hk
  have ra:=real_read memory p.a w.1 k
    (by simpa only [aRows p,p.valid.aShape.1] using hw.1) (by simpa only [aCols p,p.valid.aShape.2] using bound)
  have rb:=real_read memory p.b k w.2
    (by simpa only [bRows p,p.valid.bShape.1] using bound) (by simpa only [bCols p,p.valid.bShape.2] using hw.2)
  simpa only [initial,base,KernelSource.read,Cell.read_same,ite_true,show ("B":RegionName)≠"A" by decide,ite_false] using congrArg₂ (·*·) ra rb

theorem native_inputs (instructions : KernelNative.Instructions) (p : Prepared)
    (heap : KernelStorage.Heap) (w : Work) (hw : p.config.Active w) :
    KernelNative.target instructions p.config
      (fun r k=>KernelSource.read ((initial p (KernelStorage.read heap)).mem "A" (p.config.A.addr r k)))
      (fun k n=>KernelSource.read ((initial p (KernelStorage.read heap)).mem "B" (p.config.B.addr k n))) w.1 w.2=
      nativeTarget instructions p heap w.1 w.2 := by
  have ra (k : Nat) (hk : k<p.config.K) :
      KernelSource.read ((initial p (KernelStorage.read heap)).mem "A" (p.config.A.addr w.1 k))=
        nativePoint (KernelStorage.read heap) p.a w.1 k := by
    simpa only [initial,base,KernelSource.read,Cell.read_same,ite_true] using native_read (KernelStorage.read heap) p.a w.1 k
      (by simpa only [aRows p,p.valid.aShape.1] using hw.1) (by simpa only [aCols p,p.valid.aShape.2] using hk)
  have rb (k : Nat) (hk : k<p.config.K) :
      KernelSource.read ((initial p (KernelStorage.read heap)).mem "B" (p.config.B.addr k w.2))=
        nativePoint (KernelStorage.read heap) p.b k w.2 := by
    simpa only [initial,base,KernelSource.read,Cell.read_same,ite_true,show ("B":RegionName)≠"A" by decide,ite_false] using
      native_read (KernelStorage.read heap) p.b k w.2
        (by simpa only [bRows p,p.valid.bShape.1] using hk) (by simpa only [bCols p,p.valid.bShape.2] using hw.2)
  unfold nativeTarget KernelNative.target
  apply congrArg KernelNative.widen
  apply congrArg KernelNative.narrow
  apply congrArg (fun f=>VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages.foldRange f 0
    ((p.config.K+p.config.BK-1)/p.config.BK) 1 (FP.Value.zero .fp32))
  funext q acc
  apply congrArg (fun f=>(List.range (p.config.BK/8)).foldl f acc)
  funext z j
  congr 1
  · funext e
    split_ifs with h1 h2
    · exact ra _ h2
    · rfl
    · rfl
  · funext e
    split_ifs with h1 h2
    · exact rb _ h2
    · rfl
    · rfl

theorem native_initial (p : Prepared) (before : KernelStorage.Heap) :
    nativeHeap p before (initial p (KernelStorage.read before))=before := by
  apply KernelStorage.image_initial
  intro i hi
  simp [initial,base]

/-- All active source stores are within the remaining span of the actual
output Tensor. The byte image commutes with the entire source store fold. -/
theorem program_bytes (M : Algebra Word) (D : DotAcc Word) (p : Prepared)
    (before : KernelStorage.Heap) (s : State Word) :
    nativeHeap p before (written M D p.config s)=
      (TileShape.allIndices [p.config.BM,p.config.BN]).foldl (fun heap i=>
        if active p.config s i then KernelStorage.write heap
          (p.metadata.pointerC+4*cOffset p.config s i) (outputValue M D p.config s i) else heap)
        (nativeHeap p before s) := by
  apply KernelStorage.fold_projection
  intro i hi active
  have bounds:=view_bound p.config.C p.valid.cValid (row p.config s i.1) (col p.config s i.2.1)
    (by simpa only [cRows p,p.valid.cShape.1] using active.1) (by simpa only [cCols p,p.valid.cShape.2] using active.2)
  simpa only [cOffset,View.addr,Nat.mul_comm] using bounds

theorem real_correct (p : Prepared) (memory : Nat → ℝ) :
    ∃ final,KernelSource.launch KernelReal.scalar (dotAcc KernelReal.dot) p.config (initial p memory)=some final ∧
      (∀ w,p.config.Active w → realResult p final w.1 w.2=realTarget p memory w.1 w.2) ∧
      (∀ (region : RegionName) offset,(region≠"C" ∨ ∀ w,p.config.Active w → offset≠p.config.C.addr w.1 w.2) →
        final.mem region offset=(initial p memory).mem region offset) := by
  obtain ⟨t,he,hr,hf⟩:=KernelReal.launch_correct p.config p.valid (initial p memory)
  refine ⟨t,he,?_,hf⟩
  intro w hw
  unfold realResult
  rw [←c_address p w hw]
  have offset : (p.metadata.pointerC+4*p.config.C.addr w.1 w.2-p.metadata.pointerC)/4=p.config.C.addr w.1 w.2 := by omega
  rw [offset,hr w hw,real_inputs p memory w hw]
  rfl

theorem native_correct (instructions : KernelNative.Instructions) (p : Prepared) (before : KernelStorage.Heap) :
    ∃ final,KernelSource.launch KernelNative.scalar (dotAcc (KernelNative.dot instructions p.config.opcode))
      p.config (initial p (KernelStorage.read before))=some final ∧
      (∀ w,p.config.Active w → ∀ byte : Fin 4,
        nativeHeap p before final
          (p.c.tensor.storage.address (p.c.tensor.physicalIndex (w.1*p.config.N+w.2))+byte.val)=
          KernelStorage.bytes (nativeTarget instructions p before w.1 w.2) byte) ∧
      (∀ q,(∀ w,p.config.Active w → ∀ byte : Fin 4,
        q≠p.c.tensor.storage.address (p.c.tensor.physicalIndex (w.1*p.config.N+w.2))+byte.val) →
        nativeHeap p before final q=before q) := by
  obtain ⟨t,he,hr,hf⟩:=KernelNative.launch_correct instructions p.config p.valid (initial p (KernelStorage.read before))
  refine ⟨t,he,?_,?_⟩
  · intro w hw byte
    rw [←c_address p w hw]
    unfold nativeHeap
    rw [KernelStorage.image_byte _ _ _ _ _
      (view_bound p.config.C p.valid.cValid w.1 w.2
        (by simpa only [cRows p,p.valid.cShape.1] using hw.1) (by simpa only [cCols p,p.valid.cShape.2] using hw.2))]
    rw [hr w hw,native_inputs instructions p before w hw]
    rfl
  · intro q outside
    by_cases inside : p.metadata.pointerC≤q ∧ q<p.metadata.pointerC+4*p.config.C.capacity
    · let i:=(q-p.metadata.pointerC)/4
      let byte:Fin 4:=⟨(q-p.metadata.pointerC)%4,Nat.mod_lt _ (by decide)⟩
      have hi:i<p.config.C.capacity := by dsimp [i];omega
      have eq:p.metadata.pointerC+4*i+byte.val=q := by dsimp [i,byte];omega
      have unchanged : t.mem "C" i=(initial p (KernelStorage.read before)).mem "C" i := by
        apply hf
        right
        intro w hw equal
        apply outside w hw byte
        rw [←c_address p w hw,←equal,eq]
      rw [←eq]
      change KernelStorage.image _ _ _ _ _=_
      rw [KernelStorage.image_byte _ _ _ _ i hi byte,unchanged]
      simp only [initial,base,show ("C":RegionName)≠"A" by decide,show ("C":RegionName)≠"B" by decide,
        ite_false,Cell.read_same,KernelStorage.bytes_read]
    · exact KernelStorage.image_frame _ _ _ _ q (by omega)

#axiomsClean real_inputs
#axiomsClean native_inputs
#axiomsClean program_bytes
#axiomsClean real_correct
#axiomsClean native_correct
end
end VeriTile.Bench.Repairs.MatmulKernel.KernelEntries
