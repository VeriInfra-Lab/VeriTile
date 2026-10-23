import bench.repairs.tritonbench_g.context_attn_llama.Proofs.NativeSites
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.ContextAttnLlama.Native
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ContextAttnLlama
noncomputable section
set_option maxHeartbeats 4000000

inductive DataRole where | query | key | value | output deriving DecidableEq

def DataRole.region (d : DataRole) (c : Host.Call) := match d with
  | .query => c.regions.query | .key => c.regions.key | .value => c.regions.value | .output => c.regions.out
def DataRole.row (d : DataRole) (c : Host.Call) := match d with
  | .query => c.layout.qRow | .key => c.layout.kRow | .value => c.layout.vRow | .output => c.layout.outRow
def DataRole.head (d : DataRole) (c : Host.Call) := match d with
  | .query => c.layout.qHead | .key => c.layout.kHead | .value => c.layout.vHead | .output => c.layout.outHead
def DataRole.dim (d : DataRole) (c : Host.Call) := match d with
  | .query => c.layout.qDim | .key => c.layout.kDim | .value => c.layout.vDim | .output => c.layout.outDim

def qOffset (c : Host.Call) (role : DataRole) (idx : GridIndex c.grid) (i d : Nat) :=
  (c.start (batchAt c idx)+Formula.queryRow c.layout (c.launchAt idx) i)*role.row c+
    Formula.head c.layout (c.launchAt idx)*role.head c+d*role.dim c

def kvOffset (c : Host.Call) (role : DataRole) (idx : GridIndex c.grid) (step j d : Nat) :=
  mathPage c idx step j*role.row c+(Formula.head c.layout (c.launchAt idx)/c.layout.kvGroup)*role.head c+d*role.dim c

def qOffsetBits (c : Host.Call) (w : Widths) (role : DataRole) (idx : GridIndex c.grid) (i d : Nat) :=
  queryBits c.layout (c.launchAt idx) w.start (c.start (batchAt c idx)) i d (role.row c) (role.head c) (role.dim c)
def kvOffsetBits (c : Host.Call) (w : Widths) (role : DataRole) (idx : GridIndex c.grid) (step j d : Nat) :=
  valueBits c.layout (c.launchAt idx) (pageBits c w idx step j) d (role.row c) (role.head c) (role.dim c)

def qEvent (c : Host.Call) (A : FlatAlloc) (w : Widths) (B : Bytes A)
    (role : DataRole) (idx : GridIndex c.grid) (i d : Nat) :=
  access (queryMaskBits c w idx i) (B.base (role.region c)) (B.size (role.region c)) (qOffsetBits c w role idx i d)
def kvEvent (c : Host.Call) (A : FlatAlloc) (w : Widths) (B : Bytes A)
    (role : DataRole) (idx : GridIndex c.grid) (step j d : Nat) :=
  access (keyMaskBits c w idx step j) (B.base (role.region c)) (B.size (role.region c)) (kvOffsetBits c w role idx step j d)
def pageEvent (c : Host.Call) (A : FlatAlloc) (w : Widths) (B : Bytes A)
    (idx : GridIndex c.grid) (step j : Nat) :=
  if keyMaskBits c w idx step j then some (addptr
    (addptr (word (B.base c.regions.pages)) (B.size c.regions.pages)
      (scalar c.layout.reqRow*metadata w.request (c.request (batchAt c idx))))
    (B.size c.regions.pages) (scalar c.layout.reqToken*keyBits step c.layout.BN j)).toNat else none

theorem page_event_staged (c : Host.Call) (A : FlatAlloc) (w : Widths) (B : Bytes A)
    (idx : GridIndex c.grid) (step j : Nat) :
    pageEvent c A w B idx step j =
      access (keyMaskBits c w idx step j) (B.base c.regions.pages) (B.size c.regions.pages)
        (pageOffsetBits c.layout (c.launchAt idx) w.request (c.request (batchAt c idx)) step j) := by
  simp only [pageEvent,staged_ptr,access,pageOffsetBits]

theorem q_event (c : Host.Call) (A : FlatAlloc) (s : BlockState) (valid : Host.Valid c A s)
    (w : Widths) (B : Bytes A) (abi : ABI c A w B) (role : DataRole)
    (hr : role = .query ∨ role = .output) (idx : GridIndex c.grid) (i : Fin c.layout.BM) (d : Fin c.layout.D) :
    qEvent c A w B role idx i d =
      if Formula.queryRow c.layout (c.launchAt idx) i < c.length (batchAt c idx) then
        some (B.base (role.region c)+B.size (role.region c)*qOffset c role idx i d) else none := by
  apply event_exact A B _ _ _ _ _ (queryMask_exact c A w B abi idx i)
  · exact query_modular c.layout (c.launchAt idx) w.start _ _ _ _ _ _
      (block_abi c A w B abi idx) (batchHead_abi c A w B abi idx) c.blockM abi.heads abi.group
      (abi.start _ (Host.batch_bound c idx)) i.isLt (dim_abi c d d.isLt)
  · intro ha
    rcases hr with rfl|rfl
    · exact valid.query _ (Host.batch_bound c idx) _ (Host.head_bound c idx) _ ha _ d.isLt
    · exact valid.out _ (Host.batch_bound c idx) _ (Host.head_bound c idx) _ ha _ d.isLt

theorem page_event (c : Host.Call) (A : FlatAlloc) (s : BlockState) (valid : Host.Valid c A s)
    (w : Widths) (B : Bytes A) (abi : ABI c A w B) (idx : GridIndex c.grid)
    (step : Nat) (j : Fin c.layout.BN) (ht : step < countAt c idx) :
    pageEvent c A w B idx step j =
      if step*c.layout.BN+j.val < endAt c idx then
        some (B.base c.regions.pages+B.size c.regions.pages*
          (c.layout.reqRow*c.request (batchAt c idx)+c.layout.reqToken*(step*c.layout.BN+j.val))) else none := by
  rw [page_event_staged]
  apply event_exact A B _ _ _ _ _ (keyMask_exact c A w B abi idx step j ht)
  · exact page_modular c.layout (c.launchAt idx) w.request _ _ _
      (by rw [c.blockN]; exact c.blockM) (abi.request _ (Host.batch_bound c idx)) j.isLt
  · intro ha
    apply valid.page_bound _ (Host.batch_bound c idx)
    exact lt_of_lt_of_le ha (end_le _ _ _ _ (by dsimp only [seqAt]; omega))

theorem kv_event (c : Host.Call) (A : FlatAlloc) (s : BlockState) (valid : Host.Valid c A s)
    (w : Widths) (B : Bytes A) (abi : ABI c A w B) (role : DataRole)
    (hr : role = .key ∨ role = .value) (idx : GridIndex c.grid)
    (step : Nat) (j : Fin c.layout.BN) (d : Fin c.layout.D) (ht : step < countAt c idx) :
    kvEvent c A w B role idx step j d =
      if step*c.layout.BN+j.val < endAt c idx then
        some (B.base (role.region c)+B.size (role.region c)*kvOffset c role idx step j d) else none := by
  apply event_exact A B _ _ _ _ _ (keyMask_exact c A w B abi idx step j ht)
  · unfold kvOffsetBits
    rw [pageBits_exact c A w B abi idx step j ht]
    exact value_modular c.layout (c.launchAt idx) _ _ _ _ _
      (batchHead_abi c A w B abi idx) abi.heads abi.group (dim_abi c d d.isLt)
  · intro ha
    have hfull : step*c.layout.BN+j.val < c.length (batchAt c idx)+c.prompt (batchAt c idx) :=
      lt_of_lt_of_le ha (end_le _ _ _ _ (by dsimp only [seqAt]; omega))
    unfold kvOffset mathPage
    rw [if_pos ha]
    rcases hr with rfl|rfl
    · exact valid.key _ (Host.batch_bound c idx) _ (Host.head_bound c idx) _ hfull _ d.isLt
    · exact valid.value _ (Host.batch_bound c idx) _ (Host.head_bound c idx) _ hfull _ d.isLt

/-- Unconditional metadata reads use the exact native batch quotient. -/
theorem metadata_event (c : Host.Call) (A : FlatAlloc) (s : BlockState) (valid : Host.Valid c A s)
    (w : Widths) (B : Bytes A) (abi : ABI c A w B) (idx : GridIndex c.grid) (rg : RegionName)
    (hr : rg ∈ [c.regions.start,c.regions.sequence,c.regions.prompt,c.regions.request]) :
    access Bool.true (B.base rg) (B.size rg) (batchBits c.layout (c.launchAt idx)) =
      some (B.base rg+B.size rg*batchAt c idx) := by
  have he := (launch_exact c.layout (c.launchAt idx) (batchHead_abi c A w B abi idx) abi.heads abi.group).1
  have hb : batchAt c idx < A.extent rg := by
    simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with rfl|rfl|rfl|rfl
    · exact lt_of_lt_of_le (Host.batch_bound c idx) valid.start_bound
    · exact lt_of_lt_of_le (Host.batch_bound c idx) valid.length_bound
    · exact lt_of_lt_of_le (Host.batch_bound c idx) valid.prompt_bound
    · exact lt_of_lt_of_le (Host.batch_bound c idx) valid.request_bound
  simpa using masked_pointer A B rg Bool.true _ _ he (fun _ => hb)

def readEvent {α : Type} (mem : Nat → α) (zero : α) : Option Nat → α
  | none => zero | some p => mem p

def storeEvent {α : Type} (mem : Nat → α) (v : α) : Option Nat → Nat → α
  | none => mem | some p => Function.update mem p v

/-- Equality of byte events preserves arbitrary loaded/stored bit patterns;
there is no finite-value or floating arithmetic assumption here. -/
theorem read_congr {α : Type} (mem : Nat → α) (zero : α) (a b : Option Nat) (h : a=b) :
    readEvent mem zero a = readEvent mem zero b := by rw [h]
theorem store_congr {α : Type} (mem : Nat → α) (v : α) (a b : Option Nat) (h : a=b) :
    storeEvent mem v a = storeEvent mem v b := by rw [h]

/-- This certificate is indexed by the actual source grid and source count.
It covers metadata, all padded masks, all data loads and stores, and the last
native loop test. Exact source-site identities and evaluator equations are
`source_sites` and `source_observations`; no caller-provided trace is assumed. -/
structure Correspondence (c : Host.Call) (A : FlatAlloc) (s : BlockState)
    (w : Widths) (B : Bytes A) : Prop where
  launch : ∀ idx : GridIndex c.grid,
    batchBits c.layout (c.launchAt idx) = word (batchAt c idx) ∧
    headBits c.layout (c.launchAt idx) = word (Formula.head c.layout (c.launchAt idx)) ∧
    kvBits c.layout (c.launchAt idx) = word (Formula.head c.layout (c.launchAt idx)/c.layout.kvGroup)
  count : ∀ idx : GridIndex c.grid, Formula.count c.layout (c.launchAt idx) (Program.inputs s c.regions) = countAt c idx
  loop : ∀ (idx : GridIndex c.grid) step, step ≤ countAt c idx →
    (counterBits step).toInt = (step : Int) ∧
    (countBits w.sequence w.prompt (c.launchAt idx).block c.layout.BM (seqAt c idx)
      (c.prompt (batchAt c idx))).toInt = (countAt c idx : Int) ∧
    ((counterBits step).slt (countBits w.sequence w.prompt (c.launchAt idx).block c.layout.BM (seqAt c idx)
      (c.prompt (batchAt c idx))) = Bool.true ↔ step < countAt c idx)
  query : ∀ role, role = .query ∨ role = .output → ∀ (idx : GridIndex c.grid)
    (i : Fin c.layout.BM) (d : Fin c.layout.D),
    qEvent c A w B role idx i d = if Formula.queryRow c.layout (c.launchAt idx) i < c.length (batchAt c idx) then
      some (B.base (role.region c)+B.size (role.region c)*qOffset c role idx i d) else none
  page : ∀ (idx : GridIndex c.grid) step (j : Fin c.layout.BN), step < countAt c idx →
    pageEvent c A w B idx step j = if step*c.layout.BN+j.val < endAt c idx then
      some (B.base c.regions.pages+B.size c.regions.pages*
        (c.layout.reqRow*c.request (batchAt c idx)+c.layout.reqToken*(step*c.layout.BN+j.val))) else none
  kv : ∀ role, role = .key ∨ role = .value → ∀ (idx : GridIndex c.grid) step
    (j : Fin c.layout.BN) (d : Fin c.layout.D), step < countAt c idx →
    kvEvent c A w B role idx step j d = if step*c.layout.BN+j.val < endAt c idx then
      some (B.base (role.region c)+B.size (role.region c)*kvOffset c role idx step j d) else none
  metadata : ∀ (idx : GridIndex c.grid) rg,
    rg ∈ [c.regions.start,c.regions.sequence,c.regions.prompt,c.regions.request] →
    access Bool.true (B.base rg) (B.size rg) (batchBits c.layout (c.launchAt idx)) =
      some (B.base rg+B.size rg*batchAt c idx)
  causal : ∀ (idx : GridIndex c.grid) step (i : Fin c.layout.BM) (j : Fin c.layout.BN), step < countAt c idx →
    ((keyBits step c.layout.BN j).ule (causalBits (c.launchAt idx).block c.layout.BM i w.prompt (c.prompt (batchAt c idx))) = Bool.true ↔
      step*c.layout.BN+j.val ≤ Formula.queryRow c.layout (c.launchAt idx) i+c.prompt (batchAt c idx))

theorem correspondence (c : Host.Call) (A : FlatAlloc) (s : BlockState) (valid : Host.Valid c A s)
    (w : Widths) (B : Bytes A) (abi : ABI c A w B) : Correspondence c A s w B := by
  refine ⟨?_,source_count c A s valid,?_,q_event c A s valid w B abi,
    page_event c A s valid w B abi,kv_event c A s valid w B abi,metadata_event c A s valid w B abi,?_⟩
  · intro idx
    exact launch_exact _ _ (batchHead_abi c A w B abi idx) abi.heads abi.group
  · intro idx step ht
    exact loop_control _ _ _ _ _ _ _ (block_abi c A w B abi idx) c.blockM
      (abi.sequence _ (Host.batch_bound c idx)) (abi.prompt _ (Host.batch_bound c idx)) (by dsimp only [seqAt]; omega) ht
  · intro idx step i j ht
    have h := causal_mask w.prompt (c.launchAt idx).block c.layout.BM (seqAt c idx)
      (c.prompt (batchAt c idx)) step i j (block_abi c A w B abi idx) c.blockM
      (metadata_lt w.sequence _ (abi.sequence _ (Host.batch_bound c idx)))
      (abi.prompt _ (Host.batch_bound c idx)) (by dsimp only [seqAt]; omega) ht i.isLt (by simpa [c.blockN] using j.isLt)
    simpa only [c.blockN,Formula.queryRow] using h

#print axioms correspondence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnLlama.Native
