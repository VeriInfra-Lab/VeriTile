import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.OutputBytes

namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.ByteExecution
open VeriTile.Triton FP.Structural
open Contract TensorBinding InputAdapter FiniteExecution FiniteIndex OutputBytes
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 6000000

abbrev Machine := State Word × Read
def mask (x : Launch) (s : State Word) (i : CIndex x) : Bool :=
  (row x.M x.N x.config (s.pids 0) i.1.val).slt (word x.M) &&
    (column x.M x.N x.config (s.pids 0) i.2.1.val).slt (word x.N)
def address (x : Launch) (s : State Word) (i : CIndex x) : Nat :=
  (cOffset x.M x.N x.config (s.pids 0) i.1.val i.2.1.val).toNat

theorem address_bound (x : Launch) (legal : ConfigContract x.config)
    (env : IndexEnvelope x.M x.N x.K x.config) (s : State Word)
    (hp : s.pids 0<count x.M x.N x.config) (i : CIndex x) (hm : mask x s i=Bool.true) :
    address x s i<x.M*x.N := by
  have coords := (offset_words x.M x.N x.K x.config legal env Bool.false (s.pids 0)
    i.1.val i.2.1.val 0 0 hp i.1.isLt i.2.1.isLt).2.2
  have bound := (offset_bounds x.M x.N x.K x.config legal env Bool.false (s.pids 0)
    i.1.val i.2.1.val 0 0 hp i.1.isLt i.2.1.isLt
    (Contract.positive_tiles x.config legal).2.2.1 (Nat.zero_le _)).2.2
  have rows := row_mask x legal env _ _ hp i.1.isLt
  have cols := column_mask x legal env _ _ hp i.2.1.isLt
  simp only [mask,Bool.and_eq_true,rows,cols] at hm
  simp only [address,coords,toNat (by omega : naturalRow x.M x.N x.config (s.pids 0) i.1.val*x.N+
      naturalColumn x.M x.N x.config (s.pids 0) i.2.1.val<2^32)]
  exact linear_bound _ _ _ _ hm.1 hm.2

def step (x : Launch) (c : TensorBinding.Tensor) (s : State Word) (values : CIndex x → Int)
    (t : Machine) (i : CIndex x) : Machine :=
  if mask x s i then
    (t.1.write x.c (address x s i) (.mk .int (values i)),
      store t.2 (c.metadata.dataPtr.toNat+4*address x s i) (BitVec.ofInt 32 (values i)))
  else t

def logicalStep (x : Launch) (s : State Word) (values : CIndex x → Int)
    (t : State Word) (i : CIndex x) : State Word :=
  if mask x s i then t.write x.c (address x s i) (.mk .int (values i)) else t

theorem fold_render (x : Launch) (c : TensorBinding.Tensor) (s : State Word)
    (values : CIndex x → Int) (legal : ConfigContract x.config)
    (env : IndexEnvelope x.M x.N x.K x.config) (hp : s.pids 0<count x.M x.N x.config)
    (hc : x.c=outputC) (size : x.M*x.N=c.metadata.numel)
    (read : Read) (items : List (CIndex x)) (current : State Word) :
    items.foldl (step x c s values) (current,render read c current)=
      let final := items.foldl (logicalStep x s values) current
      (final,render read c final) := by
  induction items generalizing current with
  | nil => rfl
  | cons i tail ih =>
    simp only [List.foldl_cons]
    by_cases hm : mask x s i=Bool.true
    · have hb : address x s i<c.metadata.numel := by rw [←size]; exact address_bound x legal env s hp i hm
      simp only [step,logicalStep,if_pos hm,hc,←render_write read c current _ hb]
      exact ih _
    · simp only [step,logicalStep,if_neg hm]
      exact ih current

/-- Same finite instruction loop and masks as IntegerInstructions.program;
every store also updates the common physical byte reader. -/
def program (x : Launch) (c : TensorBinding.Tensor) (wrapped : Bool)
    (t : Machine) (suffix : Suffix x) : Option Machine :=
  (IntegerInstructions.loop x wrapped t.1 (cdiv (word x.K) (word x.config.blockK)).toNat
    (word 0) (fun _ => 0)).map fun carry =>
    (TileShape.allIndices [x.config.blockM,x.config.blockN]).foldl (step x c t.1 (suffix carry)) t

theorem program_render (x : Launch) (c : TensorBinding.Tensor) (wrapped : Bool)
    (s : State Word) (suffix : Suffix x) (legal : ConfigContract x.config)
    (env : IndexEnvelope x.M x.N x.K x.config) (hp : s.pids 0<count x.M x.N x.config)
    (hc : x.c=outputC) (size : x.M*x.N=c.metadata.numel) (read : Read) :
    program x c wrapped (s,render read c s) suffix=
      (IntegerInstructions.program x wrapped s suffix).map (fun final => (final,render read c final)) := by
  simp only [program,IntegerInstructions.program,Option.map_map]
  congr 1
  funext carry
  exact fold_render x c s (suffix carry) legal env hp hc size read _ s

def one (x : Launch) (c : TensorBinding.Tensor) (wrapped : Bool)
    (suffix : State Word → Suffix x) (pid : Nat) (t : Machine) : Option Machine :=
  let s := GridExecution.atProgram t.1 pid
  program x c wrapped (s,t.2) (suffix s)
def logicalOne (x : Launch) (wrapped : Bool) (suffix : State Word → Suffix x)
    (pid : Nat) (s : State Word) :=
  let current := GridExecution.atProgram s pid
  IntegerInstructions.program x wrapped current (suffix current)
def sequence (one : Nat → Machine → Option Machine) : List Nat → Machine → Option Machine
  | [],s => some s
  | pid::tail,s => (one pid s).bind (sequence one tail)

theorem sequence_render (x : Launch) (c : TensorBinding.Tensor) (wrapped : Bool)
    (suffix : State Word → Suffix x) (legal : ConfigContract x.config)
    (env : IndexEnvelope x.M x.N x.K x.config) (hc : x.c=outputC)
    (size : x.M*x.N=c.metadata.numel) (read : Read)
    (items : List Nat) (valid : ∀ pid∈items,pid<count x.M x.N x.config) (s : State Word) :
    sequence (one x c wrapped suffix) items (s,render read c s)=
      (GridExecution.sequence (logicalOne x wrapped suffix) items s).map
        (fun final => (final,render read c final)) := by
  induction items generalizing s with
  | nil => rfl
  | cons pid tail ih =>
    have hp := valid pid (by simp)
    have ht : ∀ q∈tail,q<count x.M x.N x.config := fun q h => valid q (by simp [h])
    have head := program_render x c wrapped (GridExecution.atProgram s pid)
      (suffix (GridExecution.atProgram s pid)) legal env hp hc size read
    have render_pid : render read c (GridExecution.atProgram s pid)=render read c s := rfl
    rw [render_pid] at head
    simp only [sequence,one,GridExecution.sequence,logicalOne,head]
    cases he : IntegerInstructions.program x wrapped (GridExecution.atProgram s pid)
      (suffix (GridExecution.atProgram s pid)) with
    | none => rfl
    | some current => simpa only [Option.map_some,Option.bind_some] using ih ht current

#axiomsClean address_bound
#axiomsClean fold_render
#axiomsClean program_render
#axiomsClean sequence_render
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.ByteExecution
