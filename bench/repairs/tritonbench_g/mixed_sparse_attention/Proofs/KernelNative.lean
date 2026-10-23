import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.RawNative
import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.KernelCompiler

namespace VeriTile.Bench.Repairs.MixedSparseAttention.KernelNative
open VeriTile Triton FP.Structural
open Whole002 Source Whole Checked KernelGrid KernelTensor
open VeriTile.Bench.Repairs.LayernormFwdTriton
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
abbrev Word := BitVec 32
def encode (word : Word) : Native.Expr := fun _=>.ready (.bits word)
def initial (c : Config) (request : Request) (layout : Reify.Layout c) (read : Nat → Word) : State Native.Expr :=
  KernelTensor.initial c request layout read 0 encode BitVec.toInt
def runtime (request : Request) : Native.Runtime := ⟨request.scaleWord⟩
def observe (I : Native.CompareISA Word) (c : Config) (_ : RegionName) (i : Nat) (cell : Cell Native.Expr) : Word :=
  Native.readNative I c cell (i%c.BD)
theorem offset_dim (q : Raw.Coord c) : Raw.offset q%c.BD=Raw.dim q := by
  simp [Raw.offset,logicalAddress,Nat.add_mod,Nat.mul_mod,Nat.mod_eq_of_lt q.2.2.isLt,Raw.dim]
def target (I : Native.CompareISA Word) (c : Config) (hbm : 0<c.BM) (request : Request)
    (layout : Reify.Layout c) (read : Nat → Word) (q : Raw.Coord c) : Word :=
  (KernelNativeReference.answer (runtime request) I c
    (start (grid c) (owner c q) (initial c request layout read)) (rowLane c hbm q)).eval I.toISA
structure Result (I : Native.CompareISA Word) (c : Config) (hbm : 0<c.BM) (request : Request)
    (layout : Reify.Layout c) (read : Nat → Word) (final : State Native.Expr) : Prop where
  output : ∀ q : Raw.Coord c,live c (initial c request layout read) q →
    observe I c c.out (Raw.offset q) (final.mem c.out (Raw.offset q))=target I c hbm request layout read q
  frame : ∀ r i,¬Writes c (initial c request layout read) r i →
    final.mem r i=(initial c request layout read).mem r i
theorem source (I : Native.CompareISA Word) (c : Config) (hbm : 0<c.BM)
    (request : Request) (layout : Reify.Layout c) (read : Nat → Word)
    (accepted : Admission c request (observed c request read 0 BitVec.toInt))
    (work : List Program) (order : work.Perm (programs (grid c))) :
    ∃ final,runPrograms (fun _=>Native.algebra (runtime request)) (fun _=>Native.dotModel c.storage I)
      c (grid c) work (initial c request layout read)=some final ∧ Result I c hbm request layout read final := by
  let s:=initial c request layout read
  have hm:=Checked.metadata c _ s accepted.metadata (metadata_initial layout read 0 encode BitVec.toInt)
  have hbn : 0<c.BN := by rw [accepted.schema.selected.2.2];decide
  obtain ⟨final,he,output,frame⟩:=RawNative.kernel_native I (runtime request) c accepted.separated
    accepted.contiguous s hm.sequence hbm hbn work order
  refine ⟨final,he,⟨?_,frame⟩⟩
  intro q hq
  simpa only [observe,offset_dim,target] using output q hq

def runQuoted (quote : ComputeKernel) (rt : Native.Runtime) (I : Native.CompareISA Word)
    (c : Config) : List Program → State Native.Expr → Option (State Native.Expr)
  | [],s=>some s
  | a::rest,s=>do
      let t←FP.Structural.exec (backend (Native.algebra rt) (Native.dotModel c.storage I)) quote (start (grid c) a s)
      runQuoted quote rt I c rest t
theorem run_quote (rt : Native.Runtime) (I : Native.CompareISA Word) (c : Config)
    (work : List Program) (s : State Native.Expr) :
    runQuoted (Source.kernel c) rt I c work s=
      runPrograms (fun _=>Native.algebra rt) (fun _=>Native.dotModel c.storage I) c (grid c) work s := by
  induction work generalizing s with
  | nil=>rfl
  | cons a rest ih=>simp only [runQuoted,runPrograms,ih];rfl

theorem input_words (I : Native.CompareISA Word) (bits : ∀ w,I.bits w=w)
    (c : Config) (request : Request) (layout : Reify.Layout c) (b : Buffers request) (read : Nat → Word)
    (accepted : Admission c request (observed c request read 0 BitVec.toInt))
    (r : Role) (hr : r∈[Role.query,.key,.value]) (i : Nat) (hi : i<r.size c) (owner : Fin 4) :
    (Native.lower (((initial c request layout read).mem (r.region c) i).read .real owner)).eval I.toISA=
      (Torch.FP32.logicalValue (fun a=>⟨read a⟩) (b.tensor r) (fpLaws b accepted r (by
        simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
        rcases hr with rfl|rfl|rfl <;> rfl)) i).bits := by
  have member : r∈inputRoles := by
    simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with rfl|rfl|rfl <;> simp [inputRoles]
  have floating : r.floating=Bool.true := by
    simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with rfl|rfl|rfl <;> rfl
  have hn:=(b.observation r).negative.trans (accepted.schema.header r).2.2.2.2.2.1
  rw [show initial c request layout read=KernelTensor.initial c request layout read 0 encode BitVec.toInt from rfl,
    input_cell layout r member,if_pos (by rw [floating])]
  simp [encode,Native.lower,Native.Graph.eval,bits,readRole,hi,Torch.FP32.logicalValue,hn,
    logical_address b accepted r i hi]

theorem complete {Store Byte : Type} (api : Memory.API Word Store Byte)
    (platform : KernelCompiler.Platform) (c : Config) (request : Request) (layout : Reify.Layout c)
    (b : Buffers request) (binding : KernelCompiler.Binding platform c request)
    (conversion : KernelCompiler.ScaleConversion) (argument : KernelCompiler.ScaleArgument)
    (_launchScale : KernelCompiler.LaunchScale conversion argument request) (store : Store)
    (accepted : Admission c request (observed c request (api.read store) 0 BitVec.toInt))
    (work : List Program) (order : work.Perm (programs (grid c))) :
    let I:=platform.instructions binding.object
    let hbm : 0<c.BM := by rw [accepted.schema.selected.2.1];decide
    let s:=initial c request layout (api.read store)
    ∃ final,runQuoted binding.decoded.source (runtime request) I c work s=some final ∧
      Result I c hbm request layout (api.read store) final ∧
      Shared.TensorImage api (placement b accepted s) (observe I c) final store ∧
      (∀ q : Raw.Coord c,live c s q →
        (Torch.FP32.logicalValue (fun a=>⟨api.read (Shared.commit api (placement b accepted s)
          (observe I c) final store) a⟩) (b.tensor .output) (fpLaws b accepted .output rfl) (Raw.offset q)).bits=
          target I c hbm request layout (api.read store) q) := by
  dsimp only
  obtain ⟨final,he,hr⟩:=source (platform.instructions binding.object) c _ request layout (api.read store) accepted work order
  refine ⟨final,?_,hr,Shared.complete _ _ _ _ _,?_⟩
  · rw [binding.quote,run_quote];exact he
  · intro q hq
    have readback:=Shared.native_logical_output api (placement b accepted _)
      (observe (platform.instructions binding.object) c) final store c.out (Raw.offset q)
      ⟨rfl,q,hq,rfl⟩
    exact readback.trans (hr.output q hq)

#axiomsClean source
#axiomsClean input_words
#axiomsClean complete
#specNonCircular target avoiding [Source.finalValues,Source.update,FP.Structural.exec,runQuoted,Whole.runPrograms]
end
end VeriTile.Bench.Repairs.MixedSparseAttention.KernelNative
