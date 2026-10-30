import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.Memory
import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.Source

namespace VeriTile.Bench.Repairs.KvCacheFilling

def stateWidth (r : Request) (value : Bool) : Nat := if value then r.dv else r.dk

structure InputLayout (r : Request) (out : Layout r) where
  base : Bool → Nat
  size : Bool → Nat
  offset : Bool → Nat
  capacity : ∀ v, offset v + r.tokens*r.heads*stateWidth r v*4 ≤ size v
  separated : ∀ v role, activeRole r role →
    base v + size v ≤ out.base role ∨ out.base role + out.size role ≤ base v

def inputPtr (r : Request) {out : Layout r} (l : InputLayout r out) (value : Bool) (row col b : Nat) : Nat :=
  l.base value + l.offset value + (row*stateWidth r value+col)*4+b

def readInput {α : Type} (r : Request) {out : Layout r} (l : InputLayout r out)
    (decode : (Fin 4 → Byte) → α) (zero : α) (m : Address → Byte) (value : Bool) (row col : Nat) : α :=
  if row < r.tokens*r.heads ∧ col < stateWidth r value then
    decode (fun b => m (physical (inputPtr r l value row col b)))
  else zero

theorem inputPtr_bound (r : Request) {out : Layout r} (l : InputLayout r out)
    (value : Bool) (row col b : Nat) (hr : row < r.tokens*r.heads) (hc : col < stateWidth r value) (hb : b<4) :
    l.base value ≤ inputPtr r l value row col b ∧ inputPtr r l value row col b < l.base value+l.size value := by
  have hf := flat_bound row col _ _ hr hc
  have hb' := flat_bound (row*stateWidth r value+col) b _ 4 hf hb
  have hcap := l.capacity value
  unfold inputPtr
  omega

def rowByteStores {α : Type} (r : Request) (l : Layout r) (ops : Operations α Byte)
    (kp : Plan r.dk) (vp : Plan r.dv) (k v : Nat → Nat → α) (encode : α → Nat → Byte) (j : Job) : List (Store Byte) :=
  (taskStores r ops kp vp k v j).flatMap (expandStore r l encode)

theorem rowByteStores_bound {α : Type} (r : Request) (d : Domain r) (l : Layout r) (ops : Operations α Byte)
    (kp : Plan r.dk) (vp : Plan r.dv) (k v : Nat → Nat → α) (encode : α → Nat → Byte)
    (j : Job) (hj : j ∈ jobs r) (s : Store Byte) (hs : s ∈ rowByteStores r l ops kp vp k v encode j) :
    ∃ role, activeRole r role ∧ l.base role ≤ s.address.row ∧ s.address.row < l.base role+l.size role := by
  obtain ⟨t,ht,hs'⟩ := List.mem_flatMap.mp hs
  obtain ⟨b,hb,ha,_⟩ := (mem_expandStore ..).mp hs'
  have hat := ((mem_taskStores ..).mp ht).1
  have hinfo := (mem_rowAddresses r j t.address).mp hat
  have hbound := bytePtr_bound r l t.address b ⟨(row_address_bound r d j hj t.address hat).1,hinfo.2.1,hinfo.2.2,hb⟩
  exact ⟨t.address.role,hinfo.2.2,by simpa [ha,physical] using hbound.1,by simpa [ha,physical] using hbound.2⟩

theorem row_preserves_inputs {α : Type} (r : Request) (d : Domain r) (l : Layout r) (inputs : InputLayout r l)
    (ops : Operations α Byte) (kp : Plan r.dk) (vp : Plan r.dv) (k v : Nat → Nat → α)
    (encode : α → Nat → Byte) (decode : (Fin 4 → Byte) → α) (zero : α)
    (j : Job) (hj : j ∈ jobs r) (m : Address → Byte) :
    readInput r inputs decode zero (runStores (rowByteStores r l ops kp vp k v encode j) m) =
      readInput r inputs decode zero m := by
  funext value row col
  unfold readInput
  split_ifs with h
  · congr 1
    funext b
    apply runStores_frame
    intro s hs he
    obtain ⟨role,har,hlo,hhi⟩ := rowByteStores_bound r d l ops kp vp k v encode j hj s hs
    have hptr := inputPtr_bound r inputs value row col b h.1 h.2 b.isLt
    have hrow := congrArg Address.row he
    change s.address.row = inputPtr r inputs value row col b at hrow
    rcases inputs.separated value role har with hsep | hsep <;> omega
  · rfl

theorem runStores_append {ν : Type} (xs ys : List (Store ν)) (m : Address → ν) :
    runStores (xs++ys) m = runStores ys (runStores xs m) := by
  induction xs generalizing m with
  | nil => rfl
  | cons x xs ih => exact ih _

/-- Actual sequential program interpretation: each program loads the current
memory through contiguous input descriptors, computes both rows and stores
bytes. It does not assume the inputs remain unchanged. -/
def runJobs {α : Type} (r : Request) (l : Layout r) (inputs : InputLayout r l)
    (ops : Operations α Byte) (kp : Plan r.dk) (vp : Plan r.dv)
    (encode : α → Nat → Byte) (decode : (Fin 4 → Byte) → α) (zero : α) : List Job → (Address → Byte) → Address → Byte
  | [],m => m
  | j::js,m =>
    let k := readInput r inputs decode zero m false
    let v := readInput r inputs decode zero m true
    runJobs r l inputs ops kp vp encode decode zero js (runStores (rowByteStores r l ops kp vp k v encode j) m)

theorem runJobs_snapshot {α : Type} (r : Request) (d : Domain r) (l : Layout r) (inputs : InputLayout r l)
    (ops : Operations α Byte) (kp : Plan r.dk) (vp : Plan r.dv)
    (encode : α → Nat → Byte) (decode : (Fin 4 → Byte) → α) (zero : α)
    (js : List Job) (valid : ∀ j ∈ js, j ∈ jobs r) (m : Address → Byte) :
    runJobs r l inputs ops kp vp encode decode zero js m =
      runStores (js.flatMap (rowByteStores r l ops kp vp
        (readInput r inputs decode zero m false) (readInput r inputs decode zero m true) encode)) m := by
  induction js generalizing m with
  | nil => rfl
  | cons j js ih =>
    simp only [runJobs,List.flatMap_cons,runStores_append]
    rw [ih (by intro j' hj'; exact valid j' (by simp [hj']))]
    rw [row_preserves_inputs r d l inputs ops kp vp _ _ encode decode zero j (valid j (by simp)) m]

theorem full_execution {α : Type} (r : Request) (d : Domain r) (l : Layout r) (inputs : InputLayout r l)
    (ops : Operations α Byte) (kp : Plan r.dk) (vp : Plan r.dv)
    (encode : α → Nat → Byte) (decode : (Fin 4 → Byte) → α) (zero : α) (m : Address → Byte) :
    runJobs r l inputs ops kp vp encode decode zero (jobs r) m =
      runStores (byteStores r l ops kp vp
        (readInput r inputs decode zero m false) (readInput r inputs decode zero m true) encode) m := by
  rw [runJobs_snapshot r d l inputs ops kp vp encode decode zero (jobs r) (by intro j hj; exact hj) m]
  simp only [byteStores,stores,List.flatMap_assoc]
  rfl


def sourceTaskValue {α β : Type} (r : Request) (ops : Operations α β)
    (kp : Plan r.dk) (vp : Plan r.dv) (kb : SourceBinding kp r.policy r.ck) (vb : SourceBinding vp r.policy r.cv)
    (k v : Nat → Nat → α) (j : Job) : Address → Cell α β :=
  rowValue (sourceRowExec ops r.policy r.ck (k (sourceRow r j)) kb.reduction.low kb.reduction.high)
    (sourceRowExec ops r.policy r.cv (v (sourceRow r j)) vb.reduction.low vb.reduction.high)

theorem sourceTaskValue_eq {α β : Type} (r : Request) (ops : Operations α β)
    (kp : Plan r.dk) (vp : Plan r.dv) (kb : SourceBinding kp r.policy r.ck) (vb : SourceBinding vp r.policy r.cv)
    (k v : Nat → Nat → α) (j : Job) (a : Address) (ha : a ∈ rowAddresses r j) :
    sourceTaskValue r ops kp vp kb vb k v j a = taskValue r ops kp vp k v j a := by
  obtain ⟨_,hc,_⟩ := (mem_rowAddresses r j a).mp ha
  rcases a with ⟨role,row,col⟩
  cases role with
  | key => exact sourceRowExec_value ops _ _ _ kp kb col hc
  | value => exact sourceRowExec_value ops _ _ _ vp vb col hc
  | keyMeta =>
    have h := sourceRowExec_metadata ops _ _ (k (sourceRow r j)) kp kb
    simp only [sourceTaskValue,taskValue,rowValue,h.1,h.2]
  | valueMeta =>
    have h := sourceRowExec_metadata ops _ _ (v (sourceRow r j)) vp vb
    simp only [sourceTaskValue,taskValue,rowValue,h.1,h.2]

def sourceRowByteStores {α : Type} (r : Request) (l : Layout r) (ops : Operations α Byte)
    (kp : Plan r.dk) (vp : Plan r.dv) (kb : SourceBinding kp r.policy r.ck) (vb : SourceBinding vp r.policy r.cv)
    (k v : Nat → Nat → α) (encode : α → Nat → Byte) (j : Job) : List (Store Byte) :=
  ((rowAddresses r j).map fun a => (⟨a,sourceTaskValue r ops kp vp kb vb k v j a⟩ : Store (Cell α Byte))).flatMap
    (expandStore r l encode)

theorem sourceRowByteStores_eq {α : Type} (r : Request) (l : Layout r) (ops : Operations α Byte)
    (kp : Plan r.dk) (vp : Plan r.dv) (kb : SourceBinding kp r.policy r.ck) (vb : SourceBinding vp r.policy r.cv)
    (k v : Nat → Nat → α) (encode : α → Nat → Byte) (j : Job) :
    sourceRowByteStores r l ops kp vp kb vb k v encode j = rowByteStores r l ops kp vp k v encode j := by
  unfold sourceRowByteStores rowByteStores taskStores
  congr 1
  apply List.map_congr_left
  intro a ha
  rw [sourceTaskValue_eq r ops kp vp kb vb k v j a ha]

def runSourceJobs {α : Type} (r : Request) (l : Layout r) (inputs : InputLayout r l)
    (ops : Operations α Byte) (kp : Plan r.dk) (vp : Plan r.dv)
    (kb : SourceBinding kp r.policy r.ck) (vb : SourceBinding vp r.policy r.cv)
    (encode : α → Nat → Byte) (decode : (Fin 4 → Byte) → α) (zero : α) : List Job → (Address → Byte) → Address → Byte
  | [],m => m
  | j::js,m =>
    let k := readInput r inputs decode zero m false
    let v := readInput r inputs decode zero m true
    runSourceJobs r l inputs ops kp vp kb vb encode decode zero js
      (runStores (sourceRowByteStores r l ops kp vp kb vb k v encode j) m)

theorem source_execution {α : Type} (r : Request) (l : Layout r) (inputs : InputLayout r l)
    (ops : Operations α Byte) (kp : Plan r.dk) (vp : Plan r.dv)
    (kb : SourceBinding kp r.policy r.ck) (vb : SourceBinding vp r.policy r.cv)
    (encode : α → Nat → Byte) (decode : (Fin 4 → Byte) → α) (zero : α) (js : List Job) (m : Address → Byte) :
    runSourceJobs r l inputs ops kp vp kb vb encode decode zero js m = runJobs r l inputs ops kp vp encode decode zero js m := by
  induction js generalizing m with
  | nil => rfl
  | cons j js ih => simp only [runSourceJobs,runJobs,sourceRowByteStores_eq,ih]

/-- The API modifies caller-owned storage and returns Python None. This model
returns only the updated byte state and Unit, without replacing output handles. -/
def call {α : Type} (r : Request) (l : Layout r) (inputs : InputLayout r l)
    (ops : Operations α Byte) (kp : Plan r.dk) (vp : Plan r.dv)
    (kb : SourceBinding kp r.policy r.ck) (vb : SourceBinding vp r.policy r.cv)
    (encode : α → Nat → Byte) (decode : (Fin 4 → Byte) → α) (zero : α) (m : Address → Byte) : (Address → Byte) × Unit :=
  (runSourceJobs r l inputs ops kp vp kb vb encode decode zero (jobs r) m,())

theorem call_execution {α : Type} (r : Request) (d : Domain r) (l : Layout r) (inputs : InputLayout r l)
    (ops : Operations α Byte) (kp : Plan r.dk) (vp : Plan r.dv)
    (kb : SourceBinding kp r.policy r.ck) (vb : SourceBinding vp r.policy r.cv)
    (encode : α → Nat → Byte) (decode : (Fin 4 → Byte) → α) (zero : α) (m : Nat → Byte) :
    (call r l inputs ops kp vp kb vb encode decode zero (fun a => m a.row)).1 ∘ physical =
      executeBytes r l ops kp vp
        (readInput r inputs decode zero (fun a => m a.row) false)
        (readInput r inputs decode zero (fun a => m a.row) true) encode m := by
  unfold call
  rw [source_execution,full_execution r d]
  rfl

theorem call_returns_none {α : Type} (r : Request) (l : Layout r) (inputs : InputLayout r l)
    (ops : Operations α Byte) (kp : Plan r.dk) (vp : Plan r.dv)
    (kb : SourceBinding kp r.policy r.ck) (vb : SourceBinding vp r.policy r.cv)
    (encode : α → Nat → Byte) (decode : (Fin 4 → Byte) → α) (zero : α) (m : Address → Byte) :
    (call r l inputs ops kp vp kb vb encode decode zero m).2 = () := rfl


theorem empty_call {α : Type} (r : Request) (l : Layout r) (inputs : InputLayout r l)
    (ops : Operations α Byte) (kp : Plan r.dk) (vp : Plan r.dv)
    (kb : SourceBinding kp r.policy r.ck) (vb : SourceBinding vp r.policy r.cv)
    (encode : α → Nat → Byte) (decode : (Fin 4 → Byte) → α) (zero : α) (m : Address → Byte)
    (h : jobs r = []) : call r l inputs ops kp vp kb vb encode decode zero m = (m,()) := by
  simp [call,h,runSourceJobs]

end VeriTile.Bench.Repairs.KvCacheFilling
