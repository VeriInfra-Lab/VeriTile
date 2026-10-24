import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.HostIdentity

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.SavedAliasInhabited
open VeriTile.Triton Storage Host
noncomputable section

/-- Small positive consistency witnesses; the public proofs remain all-shape. -/
def call (rows : Fin 2) : Call := ⟨rows.val,1,1,by decide,by decide,by decide,Bool.false,1⟩
def code (r : RegionName) : Nat :=
  if r="x" then 0 else if r="w" then 1 else if r="r" then 2 else
    if r="y" then 3 else if r="dx" then 4 else 5

theorem small (rows : Fin 2) (r : RegionName) : cells (call rows) r≤1 := by
  have bound:=rows.isLt
  simp only [cells,call,Call.cells,Nat.mul_one]
  split_ifs <;> omega

def allocation (rows : Fin 2) (r : RegionName) : Native.Allocation (cells (call rows) r) where
  storageId := code r
  buffer := {
    base := code r*16
    extent := cells (call rows) r
    bytes := 4
    bytesPositive := by decide
    allocation := by
      have bound:=small rows r
      change cells (call rows) r*4<9223372036854775808
      omega
    placement := by
      have bound:=small rows r
      have rank : code r≤5 := by unfold code; split_ifs <;> omega
      omega }
  holds := le_rfl
  fp32Bytes := rfl

def placement (rows : Fin 2) (phase : Phase) : Placement (call rows) phase where
  allocation r _ := allocation rows r
  identities := by
    intro r hr s hs different
    cases phase with
    | forward =>
      simp only [buffers,List.mem_cons,List.not_mem_nil,or_false] at hr hs
      rcases hr with rfl|rfl|rfl|rfl <;> rcases hs with rfl|rfl|rfl|rfl <;> simp_all [allocation,code]
    | backward =>
      simp only [buffers,List.mem_cons,List.not_mem_nil,or_false] at hr hs
      rcases hr with rfl|rfl|rfl|rfl|rfl <;> rcases hs with rfl|rfl|rfl|rfl|rfl <;> simp_all [allocation,code]
  separate := by
    intro r hr s hs different
    have rb:=small rows r
    have sb:=small rows s
    cases phase with
    | forward =>
      simp only [buffers,List.mem_cons,List.not_mem_nil,or_false] at hr hs
      rcases hr with rfl|rfl|rfl|rfl <;> rcases hs with rfl|rfl|rfl|rfl <;> simp_all [allocation,code] <;> omega
    | backward =>
      simp only [buffers,List.mem_cons,List.not_mem_nil,or_false] at hr hs
      rcases hr with rfl|rfl|rfl|rfl|rfl <;> rcases hs with rfl|rfl|rfl|rfl|rfl <;> simp_all [allocation,code] <;> omega

/-- All saved allocations and Y are caller-visible; dx/dw are elsewhere. -/
def live : Caller := ⟨fun addr => addr<64,fun id => id<4⟩
def context {α : Type} (rows : Fin 2) (saved : CellHeap α) : Context α :=
  ⟨call rows,[rows.val,1],[1],0,saved,placement rows .forward⟩

def prepared {α : Type} (rows : Fin 2) (saved : CellHeap α) : BackwardPrepared (context rows saved) live where
  storage := placement rows .backward
  fresh := by
    intro addr active written
    rcases written with ⟨i,inside,location⟩
    have output : i.1.val="dx" ∨ i.1.val="dw" := by simpa [outputs] using i.1.property
    change addr<64 at active
    rcases output with chosen|chosen <;>
      simp [Storage.atIndex,address,placement,allocation,chosen,code] at location <;> omega
  freshIdentity := by
    intro r hr fresh
    have roles : r="dx" ∨ r="dw" := by simpa [newRole,outputs] using fresh
    rcases roles with rfl|rfl <;> norm_num [live,placement,allocation,code] <;> decide
  freshBacking := by
    intro r hr fresh addr lo hi active
    have roles : r="dx" ∨ r="dw" := by simpa [newRole,outputs] using fresh
    change addr<64 at active
    rcases roles with rfl|rfl <;> simp [placement,allocation,code] at lo <;> omega
  reuse := by intro r hr; rfl

theorem saved_visible {α : Type} (rows : Fin 2) (saved : CellHeap α)
    (r : RegionName) (role : savedRole r) :
    live.identities ((context rows saved).storage.allocation r (saved_forward r role)).storageId := by
  rcases role with rfl|rfl|rfl
  · change code ("x" : RegionName)<4
    decide
  · change code ("w" : RegionName)<4
    decide
  · change code ("r" : RegionName)<4
    decide

theorem preparation_with_visible_saved {α : Type} (rows : Fin 2) (saved : CellHeap α) :
    Nonempty (BackwardPrepared (context rows saved) live) ∧
    ∀ r (role : savedRole r),
      live.identities ((context rows saved).storage.allocation r (saved_forward r role)).storageId :=
  ⟨⟨prepared rows saved⟩,saved_visible rows saved⟩

def upstream {α : Type} (rows : Fin 2) (saved : CellHeap α) (role : RegionName) : Upstream α :=
  ⟨[cells (call rows) role],0,fun off => saved (role,off)⟩

theorem saved_x_valid {α : Type} (rows : Fin 2) (saved : CellHeap α) :
    backwardValidate (context rows saved) (upstream rows saved "x")=.ok () := by
  simp [backwardValidate,context,upstream,call,cells,Call.cells]
theorem saved_r_valid {α : Type} (rows : Fin 2) (saved : CellHeap α) :
    backwardValidate (context rows saved) (upstream rows saved "r")=.ok () := by
  simp [backwardValidate,context,upstream,call,cells,Call.cells]
theorem saved_w_valid {α : Type} (saved : CellHeap α) :
    backwardValidate (context 1 saved) (upstream 1 saved "w")=.ok () := by
  simp [backwardValidate,context,upstream,call,cells,Call.cells]

theorem zero_row_saved_x {α : Type} (saved : CellHeap α) :
    Nonempty (BackwardPrepared (context 0 saved) live) ∧
    backwardValidate (context 0 saved) (upstream 0 saved "x")=.ok () ∧
    ((context 0 saved).storage.allocation "x" (by simp [buffers])).buffer.extent=0 :=
  ⟨⟨prepared 0 saved⟩,saved_x_valid 0 saved,rfl⟩

/-- A changed current payload is kept verbatim, even if supplied logical values
would encode differently. No stale snapshot is written over saved storage. -/
theorem raw_payload_not_restored :
    (finish (prepared 1 (fun _ => (7 : Nat))).toPrepared (fun value _ => value)
      (fun _ => 239) (fun _ => 999) (.error Failure.launch : Except Failure Unit)).bytes 0=239 :=
  finish_frame _ _ _ _ _ 0 (by change 0<64; decide)

theorem version_error_preserves_current_entry {α β : Type}
    (weight input : Call → CellHeap α → Option (CellHeap α)) (zero : α)
    (ctx : Context α) (g : Upstream α) (caller : Caller) (encode : α → Fin 4 → β) (entry : Heap β)
    (prepare : Except Failure (BackwardPrepared ctx caller)) (w x z : Gate α) (resolve : Except Failure Unit) :
    Host.backward weight input zero ctx g caller encode entry prepare w x z resolve (.error .savedVersion)=
      early entry .savedVersion := by rfl

#axiomsClean preparation_with_visible_saved
#axiomsClean saved_x_valid
#axiomsClean saved_r_valid
#axiomsClean saved_w_valid
#axiomsClean zero_row_saved_x
#axiomsClean raw_payload_not_restored
#axiomsClean version_error_preserves_current_entry
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.SavedAliasInhabited
