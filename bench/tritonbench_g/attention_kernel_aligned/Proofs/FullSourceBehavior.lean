import bench.tritonbench_g.attention_kernel_aligned.Proofs.FullFp16Complete
namespace VeriTile.Bench.TritonBenchG.AttentionKernelAligned.FullFp16
open VeriTile.Triton FP.Structural
open Full
noncomputable section
variable {α : Type} [Inhabited α]

/-- The arbitrary-length specialization retains the original unchecked load. -/
theorem query_storage (p : Parameters) :
    ((prologue p).drop 11 |>.take 1) =
      [.assign .fp16 [p.blockM,p.head] "q" (.alg
        (.load .fp16 (.blockPtr (.ref .blockPtr [p.blockM,p.head] "Q_block_ptr") [])
          .none .unspecified))] := rfl

theorem key_value_storage (p : Parameters) :
    (body p).take 2 =
      [.assign .fp16 [p.head,p.blockN] "k" (.alg
        (.load .fp16 (.blockPtr (.ref .blockPtr [p.head,p.blockN] "K_block_ptr") [])
          .none .unspecified)),
       .assign .fp16 [p.blockN,p.head] "v" (.alg
        (.load .fp16 (.blockPtr (.ref .blockPtr [p.blockN,p.head] "V_block_ptr") [])
          .none .unspecified))] := rfl

theorem score_dot_source (p : Parameters) :
    ((body p).drop 3 |>.take 1) =
      [.assign .fp16 [p.blockM,p.blockN] "qk" (.alg
        (.add .fp16 Broadcast.nil.consSame.consSame (.ref .fp16 [p.blockM,p.blockN] "qk")
          (.dotTyped .fp16 .fp16 .fp16 (batch:=[]) (M:=p.blockM) (K:=p.head) (N:=p.blockN)
            (.ref .fp16 [p.blockM,p.head] "q")
            (.ref .fp16 [p.head,p.blockN] "k") {outDType:=some .fp16})))] := rfl

/-- No implicit clipping follows from the parent shape of a block pointer. -/
theorem unchecked_is_active (ptr : BlockPtr) (index : List Nat) :
    ptr.inBounds index [] = Bool.true := rfl

/-- Every address observed by the original per-program source. These include
all tail lanes and both unmasked bias loads. They are not rounded or padded. -/
def ReadAddress (p : Parameters) (s : State α) (r : RegionName) (o : Nat) : Prop :=
  (r = p.q ∧ ∃ i : QLane p, o = qOffset p s i) ∨
  (r = p.bias ∧ ∃ i : ScoreLane p, o = b1Offset p s i) ∨
  ∃ c, c < p.blocks ∧
    ((r = p.k ∧ ∃ i : KLane p, o = kOffset p s c i) ∨
     (r = p.v ∧ ∃ i : VLane p, o = vOffset p s c i) ∨
     (r = p.bias ∧ ∃ i : Fin p.blockM, o = b0Offset p s c i))

def WriteAddress (p : Parameters) (s : State α) (r : RegionName) (o : Nat) : Prop :=
  r = p.output ∧ ∃ i : QLane p, o = outputOffset p s i

/-- An explicit allocation-domain condition, kept separate from the total
memory value theorem. The original wrapper does not establish this condition. -/
def AddressDomain (p : Parameters) (s : State α) (extent : RegionName → Nat) : Prop :=
  (∀ r o, ReadAddress p s r o → o < extent r) ∧
  (∀ r o, WriteAddress p s r o → o < extent r)

/-- Any nonempty remainder produces a real final loop iteration and a lane
whose coordinate is exactly N_CTX, the first coordinate beyond its shape. -/
theorem partial_block_has_tail (length block : Nat) (hb : 0 < block)
    (ht : length % block ≠ 0) :
    length / block * block < length ∧ length % block < block ∧
      length / block * block + length % block = length := by
  have hsum := Nat.mod_add_div length block
  have hlt := Nat.mod_lt length hb
  rw [Nat.mul_comm block (length / block)] at hsum
  omega

theorem key_tail_is_visited (p : Parameters) (hBN : 0 < p.blockN)
    (ht : p.length % p.blockN ≠ 0) :
    p.length / p.blockN < p.blocks ∧
      p.length / p.blockN * p.blockN + p.length % p.blockN = p.length := by
  obtain ⟨hguard, _, he⟩ := partial_block_has_tail p.length p.blockN hBN ht
  exact ⟨(block_guard p _ hBN).mp hguard, he⟩

/-- Accepted wrapper layout with N_CTX=65, one batch/head, and 64x64 tiles.
Bias is contiguous with the required final dimension 128. -/
def tailParameters : Parameters where
  q := "Q"
  k := "K"
  v := "V"
  bias := "B0"
  output := "Out"
  scale := 1
  qHeadStride := 4160
  qRowStride := 64
  qColStride := 1
  kHeadStride := 4160
  kSeqStride := 64
  kDimStride := 1
  vHeadStride := 4160
  vSeqStride := 64
  vDimStride := 1
  outHeadStride := 4160
  outRowStride := 64
  outColStride := 1
  biasHeadStride := 8320
  biasRowStride := 128
  batches := 1
  heads := 1
  length := 65
  blockM := 64
  blockN := 64
  head := 64
  biasLast := 64
  biasNumel := 128

def tailState (s : State α) : State α :=
  { s with pids := fun ax => if ax = 0 then 1 else 0 }

/-- Each address equals the end of an exactly sized allocation, so all four
are out of bounds. Empty boundary-check lists still mark these lanes active. -/
theorem original_tail_accesses (s : State α) :
    tailParameters.blocks = 2 ∧
    (qPointer tailParameters (tailState s)).address [1,0] = 4160 ∧
    (kPointer tailParameters (tailState s) 1).address [0,1] = 4160 ∧
    (vPointer tailParameters (tailState s) 1).address [1,0] = 4160 ∧
    (outputPointer tailParameters (tailState s)).address [1,0] = 4160 ∧
    (qPointer tailParameters (tailState s)).inBounds [1,0] [] = Bool.true ∧
    (kPointer tailParameters (tailState s) 1).inBounds [0,1] [] = Bool.true ∧
    (vPointer tailParameters (tailState s) 1).inBounds [1,0] [] = Bool.true ∧
    (outputPointer tailParameters (tailState s)).inBounds [1,0] [] = Bool.true := by
  simp [tailParameters,Parameters.blocks,tailState,qPointer,kPointer,vPointer,outputPointer,
    headBase,kvBase,BlockPtr.inBounds]

/-- Ignored source arguments stay ignored: the new contract does not repair
the V/output head-base computation behind the original program's back. -/
theorem ignored_head_strides (p : Parameters) (vh oh : Nat) :
    source { p with vHeadStride := vh, outHeadStride := oh } = source p := rfl

#axiomsClean query_storage
#axiomsClean key_value_storage
#axiomsClean score_dot_source
#axiomsClean partial_block_has_tail
#axiomsClean key_tail_is_visited
#axiomsClean original_tail_accesses
#axiomsClean ignored_head_strides
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionKernelAligned.FullFp16
