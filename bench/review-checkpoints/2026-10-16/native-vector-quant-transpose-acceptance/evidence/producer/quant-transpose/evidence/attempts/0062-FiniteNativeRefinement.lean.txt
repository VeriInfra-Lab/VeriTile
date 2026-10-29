import bench.tritonbench_g.quant_transpose_kernel.Proofs.FiniteNativeHost

namespace QuantTransposeFinite
open VeriTile.Bench.TritonBenchG
set_option maxHeartbeats 12000000
set_option maxRecDepth 24000
noncomputable section
open Classical

theorem byte_of_split (d : Storage) (w : Word d) (k : Nat) :
    byteOf w k=ReluTritonKernel.ByteMemory.splitByte w k := by
  apply BitVec.eq_of_toNat_eq
  simp [byteOf,ReluTritonKernel.ByteMemory.splitByte,Nat.shiftRight_eq_div_pow,Nat.mul_comm]

theorem native_read_of_bytes (d : Storage) (m : ByteMemory) (address : Nat) (w : Word d)
    (h : ∀ k, k<bytes d → m (address+k)=some (byteOf w k)) :
    nativeReadWord d m address=some w := by
  have present : ∀ k, k<bytes d → (m (address+k)).isSome := by
    intro k hk
    simp [h k hk]
  unfold nativeReadWord
  rw [if_pos present]
  congr 1
  apply BitVec.eq_of_getLsbD_eq
  intro bit within
  rw [BitVec.getLsbD_cast,ReluTritonKernel.ByteMemory.load_bit]
  · rw [h (bit/8) (by have := byte_width d; omega)]
    simp only [Option.getD_some,byte_of_split,ReluTritonKernel.ByteMemory.splitByte,
      BitVec.getLsbD_extractLsb']
    have low : bit%8<8 := Nat.mod_lt _ (by decide)
    simp only [low,decide_true,Bool.true_and]
    congr 1
    omega
  · simpa [byte_width] using within

theorem native_host_input_word (b : Numerics) (d : Storage) (a : Input) (alloc : Allocations)
    (m : Memory) (x : String) (logical : List (Word d)) (xs : Nat → Word d) (p : PhysicalLayout)
    (D : Domain d a alloc m x logical xs) (P : PhysicalBinding d a alloc m x p)
    (maximum : Word d) (i : Nat) (hi : i<a.numel) :
    nativeReadWord d (hostPhysical d p maximum (b.reciprocal d maximum))
      (p.inputPointer.toNat+inputOffset (parameters a alloc x) i*bytes d)=some (xs i) := by
  apply native_read_of_bytes
  intro k hk
  rw [←host_physical_preparation d a alloc m x p D.fresh P maximum (b.reciprocal d maximum)]
  exact physical_input_load d a alloc m _ x p logical xs D P i k hi hk

theorem native_host_inverse_word (b : Numerics) (d : Storage) (p : PhysicalLayout) (maximum : Word d) :
    nativeReadWord (compute d) (hostPhysical d p maximum (b.reciprocal d maximum)) p.inverse.base=
      some (b.reciprocal d maximum) := by
  apply native_read_of_bytes
  intro k hk
  simp [hostPhysical,wordWrite,inRange,hk]

/-- Successful input loads preserve exactly their active source lane relation. -/
def LoadedAt (d : Storage) (a : Input) (p : PhysicalLayout) (snapshot : ByteMemory)
    (pid lane : Nat) (item : NativeLoaded d) : Prop :=
  item.lane=lane ∧ ∃ native,
    nativeLane a.M a.N a.sam a.san pid lane=some native ∧ native.mask=Bool.true ∧
    item.inputAddress=(nativePointer p.inputPointer.toNat (bytes d) native.inputOffset).toNat ∧
    item.outputAddress=(nativePointer p.output.base 1 native.outputOffset).toNat ∧
    nativeReadWord d snapshot item.inputAddress=some item.value

theorem native_load_tile_members (d : Storage) (a : Input) (alloc : Allocations)
    (p : PhysicalLayout) (snapshot : ByteMemory) (pid : Nat) (lanes : List Nat)
    (loaded : List (NativeLoaded d))
    (h : nativeLoadTile d a alloc p snapshot pid lanes=.ok loaded) :
    (∀ item∈loaded, ∃ lane∈lanes, LoadedAt d a p snapshot pid lane item) ∧
    (∀ lane∈lanes, ∀ native, nativeLane a.M a.N a.sam a.san pid lane=some native →
      native.mask=Bool.true → ∃ item∈loaded, LoadedAt d a p snapshot pid lane item) := by
  induction lanes generalizing loaded with
  | nil => cases h; simp
  | cons lane rest ih =>
    simp only [nativeLoadTile] at h
    cases hn : nativeLane a.M a.N a.sam a.san pid lane with
    | none => simp [hn,requireSome] at h
    | some native =>
      simp only [hn,requireSome,native_bind_ok] at h
      by_cases mask : native.mask=Bool.true
      · simp only [mask,ite_true] at h
        split at h
        · cases h
        · cases hv : nativeReadWord d snapshot
            (nativePointer p.inputPointer.toNat (bytes d) native.inputOffset).toNat with
          | none => simp [hv] at h
          | some value =>
            cases ht : nativeLoadTile d a alloc p snapshot pid rest with
            | error issue => simp [hv,ht] at h
            | ok tail =>
              simp only [hv,native_bind_ok,ht,native_pure,Except.ok.injEq] at h
              subst loaded
              have tailSpec := ih tail ht
              have headSpec : LoadedAt d a p snapshot pid lane
                  ⟨lane,(nativePointer p.inputPointer.toNat (bytes d) native.inputOffset).toNat,
                  (nativePointer p.output.base 1 native.outputOffset).toNat,value⟩ :=
                ⟨rfl,native,hn,mask,rfl,rfl,hv⟩
              constructor
              · intro item hm
                rcases List.mem_cons.1 hm with rfl|hm
                · exact ⟨lane,by simp,headSpec⟩
                · obtain ⟨l,hl,hs⟩ := tailSpec.1 item hm
                  exact ⟨l,by simp [hl],hs⟩
              · intro l hl n hln hnm
                rcases List.mem_cons.1 hl with rfl|hl
                · exact ⟨_,by simp,headSpec⟩
                · obtain ⟨item,hi,hs⟩ := tailSpec.2 l hl n hln hnm
                  exact ⟨item,by simp [hi],hs⟩
      · simp only [mask] at h
        have tailSpec := ih loaded h
        constructor
        · intro item hm
          obtain ⟨l,hl,hs⟩ := tailSpec.1 item hm
          exact ⟨l,by simp [hl],hs⟩
        · intro l hl n hln hnm
          rcases List.mem_cons.1 hl with rfl|hl
          · rw [hn] at hln
            cases hln
            exact (mask hnm).elim
          · exact tailSpec.2 l hl n hln hnm

def WrittenFrom (b : Numerics) (d : Storage) (pid : Nat) (inverse : Word (compute d))
    (item : NativeLoaded d) (w : NativeWrite) : Prop :=
  w.pid=pid ∧ w.lane=item.lane ∧ w.inputAddress=item.inputAddress ∧
  w.outputAddress=item.outputAddress ∧ result b d item.value inverse=some w.value

theorem native_tile_write_members (b : Numerics) (d : Storage) (alloc : Allocations)
    (p : PhysicalLayout) (pid : Nat) (inverse : Word (compute d))
    (loaded : List (NativeLoaded d)) (seen : List Nat) (writes : List NativeWrite) (owners : List Nat)
    (h : nativeTileWrites b d alloc p pid inverse loaded seen=.ok (writes,owners)) :
    (∀ w∈writes, ∃ item∈loaded, WrittenFrom b d pid inverse item w) ∧
    (∀ item∈loaded, ∃ w∈writes, WrittenFrom b d pid inverse item w) := by
  induction loaded generalizing seen writes owners with
  | nil => cases h; simp
  | cons item rest ih =>
    simp only [nativeTileWrites] at h
    cases hz : result b d item.value inverse with
    | none => simp [hz,requireSome] at h
    | some z =>
      simp only [hz,requireSome,native_bind_ok] at h
      split at h
      · cases h
      · split at h
        · cases h
        · cases ht : nativeTileWrites b d alloc p pid inverse rest (item.outputAddress::seen) with
          | error issue => simp [ht] at h
          | ok tail =>
            rcases tail with ⟨tail,finalOwners⟩
            simp [ht] at h
            rcases h with ⟨rfl,rfl⟩
            have tailSpec := ih _ _ _ ht
            have headSpec : WrittenFrom b d pid inverse item
                ⟨pid,item.lane,item.inputAddress,item.outputAddress,z⟩ := ⟨rfl,rfl,rfl,rfl,hz⟩
            constructor
            · intro w hw
              rcases List.mem_cons.1 hw with rfl|hw
              · exact ⟨item,by simp,headSpec⟩
              · obtain ⟨v,hv,hs⟩ := tailSpec.1 w hw
                exact ⟨v,by simp [hv],hs⟩
            · intro v hv
              rcases List.mem_cons.1 hv with rfl|hv
              · exact ⟨_,by simp,headSpec⟩
              · obtain ⟨w,hw,hs⟩ := tailSpec.2 v hv
                exact ⟨w,by simp [hw],hs⟩

theorem native_host_inverse_load (b : Numerics) (d : Storage) (a : Input) (alloc : Allocations)
    (m : Memory) (x : String) (p : PhysicalLayout) (P : PhysicalBinding d a alloc m x p)
    (maximum : Word d) (pid : Nat) :
    nativeLoadInverse d alloc p (hostPhysical d p maximum (b.reciprocal d maximum)) pid=
      .ok (b.reciprocal d maximum) := by
  have bound : InRegion p.inverse p.inverse.base (bytes (compute d)) := by
    have hc := P.inverseCapacity
    have hv := P.inverseValid
    unfold InRegion ByteRegion.Valid at *
    omega
  simp [nativeLoadInverse,bound,P.inverseKernelRead,native_host_inverse_word,requireSome]

theorem loaded_at_refines (b : Numerics) (d : Storage) (a : Input) (alloc : Allocations)
    (m : Memory) (x : String) (logical : List (Word d)) (xs : Nat → Word d) (p : PhysicalLayout)
    (D : Domain d a alloc m x logical xs) (P : PhysicalBinding d a alloc m x p)
    (refines : ArithmeticRefines a d p) (maximum : Word d) (pid lane : Nat)
    (hp : pid<gridM a.M*gridN a.N) (hl : lane<128*128) (item : NativeLoaded d)
    (atLane : LoadedAt d a p (hostPhysical d p maximum (b.reciprocal d maximum)) pid lane item) :
    active (parameters a alloc x) pid lane ∧
    item.value=xs (offset (parameters a alloc x) pid lane) ∧
    item.outputAddress=p.output.base+offset (parameters a alloc x) pid lane := by
  obtain ⟨native,hn,hm,addresses⟩ := refines pid lane hp hl
  obtain ⟨_,actual,ha,mask,hin,hout,read⟩ := atLane
  rw [hn] at ha
  cases ha
  have act : active (parameters a alloc x) pid lane := by
    rw [hm] at mask
    exact of_decide_eq_true mask
  obtain ⟨hi,he⟩ := active_index (parameters a alloc x) pid lane act
  have hi' : offset (parameters a alloc x) pid lane<a.numel := by
    simpa [parameters,Parameters.n,←D.matrixCount] using hi
  obtain ⟨hin',hout'⟩ := addresses mask
  have ptr : item.inputAddress=p.inputPointer.toNat+
      inputOffset (parameters a alloc x) (offset (parameters a alloc x) pid lane)*bytes d := by
    rw [hin,hin',he]
    rfl
  have input := native_host_input_word b d a alloc m x logical xs p D P maximum _ hi'
  rw [ptr,input] at read
  refine ⟨act,(Option.some.inj read).symm,?_⟩
  rw [hout,hout']
  simp [offset,parameters,Nat.add_comm]

def LogicalWrite (b : Numerics) (d : Storage) (a : Input) (p : PhysicalLayout)
    (xs : Nat → Word d) (inverse : Word (compute d)) (w : NativeWrite) : Prop :=
  ∃ i, i<a.numel ∧ w.outputAddress=p.output.base+i ∧ result b d (xs i) inverse=some w.value

def LogicalWrites (b : Numerics) (d : Storage) (a : Input) (p : PhysicalLayout)
    (xs : Nat → Word d) (inverse : Word (compute d)) (indices : List Nat) (writes : List NativeWrite) : Prop :=
  (∀ w∈writes, LogicalWrite b d a p xs inverse w) ∧
  (∀ i∈indices, ∃ w∈writes, w.outputAddress=p.output.base+i)

theorem native_program_logical (b : Numerics) (d : Storage) (a : Input) (alloc : Allocations)
    (m : Memory) (x : String) (logical : List (Word d)) (xs : Nat → Word d) (p : PhysicalLayout)
    (D : Domain d a alloc m x logical xs) (P : PhysicalBinding d a alloc m x p)
    (refines : ArithmeticRefines a d p) (maximum : Word d) (pid : Nat)
    (hp : pid<gridM a.M*gridN a.N) (seen : List Nat) (writes : List NativeWrite) (owners : List Nat)
    (h : nativeProgramPlan b d a alloc p (hostPhysical d p maximum (b.reciprocal d maximum))
      pid seen=.ok (writes,owners)) :
    LogicalWrites b d a p xs (b.reciprocal d maximum)
      (laneIndices (parameters a alloc x) pid (List.range (128*128))) writes := by
  unfold nativeProgramPlan at h
  rw [native_host_inverse_load b d a alloc m x p P maximum pid] at h
  cases hl : nativeLoadTile d a alloc p (hostPhysical d p maximum (b.reciprocal d maximum))
      pid (List.range (128*128)) with
  | error issue => simp [hl] at h
  | ok loaded =>
    simp only [hl,native_bind_ok] at h
    have loadSpec := native_load_tile_members d a alloc p _ pid _ loaded hl
    have writeSpec := native_tile_write_members b d alloc p pid _ loaded seen writes owners h
    constructor
    · intro w hw
      obtain ⟨item,hm,hs⟩ := writeSpec.1 w hw
      obtain ⟨lane,hin,hat⟩ := loadSpec.1 item hm
      obtain ⟨act,value,address⟩ := loaded_at_refines b d a alloc m x logical xs p D P
        refines maximum pid lane hp (List.mem_range.1 hin) item hat
      have hi := (active_index (parameters a alloc x) pid lane act).1
      refine ⟨offset (parameters a alloc x) pid lane,?_,hs.2.2.2.1.trans address,?_⟩
      · simpa [parameters,Parameters.n,←D.matrixCount] using hi
      · rw [←value]; exact hs.2.2.2.2
    · intro i hi
      obtain ⟨lane,hlane,he⟩ := List.mem_filterMap.1 hi
      split at he
      · rename_i act
        cases he
        obtain ⟨native,hn,hm,_⟩ := refines pid lane hp (List.mem_range.1 hlane)
        have mask : native.mask=Bool.true := by
          rw [hm]
          exact decide_eq_true act
        obtain ⟨item,hin,hat⟩ := loadSpec.2 lane hlane native hn mask
        obtain ⟨w,hw,hs⟩ := writeSpec.2 item hin
        have address := (loaded_at_refines b d a alloc m x logical xs p D P
          refines maximum pid lane hp (List.mem_range.1 hlane) item hat).2.2
        exact ⟨w,hw,hs.2.2.2.1.trans address⟩
      · cases he

theorem logical_writes_append (b : Numerics) (d : Storage) (a : Input) (p : PhysicalLayout)
    (xs : Nat → Word d) (inverse : Word (compute d)) (leftI rightI : List Nat)
    (left right : List NativeWrite)
    (hl : LogicalWrites b d a p xs inverse leftI left)
    (hr : LogicalWrites b d a p xs inverse rightI right) :
    LogicalWrites b d a p xs inverse (leftI++rightI) (left++right) := by
  constructor
  · intro w hw
    rcases List.mem_append.1 hw with hw|hw
    · exact hl.1 w hw
    · exact hr.1 w hw
  · intro i hi
    rcases List.mem_append.1 hi with hi|hi
    · obtain ⟨w,hw,he⟩ := hl.2 i hi
      exact ⟨w,List.mem_append_left _ hw,he⟩
    · obtain ⟨w,hw,he⟩ := hr.2 i hi
      exact ⟨w,List.mem_append_right _ hw,he⟩

theorem native_grid_logical (b : Numerics) (d : Storage) (a : Input) (alloc : Allocations)
    (m : Memory) (x : String) (logical : List (Word d)) (xs : Nat → Word d) (p : PhysicalLayout)
    (D : Domain d a alloc m x logical xs) (P : PhysicalBinding d a alloc m x p)
    (refines : ArithmeticRefines a d p) (maximum : Word d) (pids : List Nat)
    (bound : ∀ pid∈pids, pid<gridM a.M*gridN a.N)
    (seen : List Nat) (writes : List NativeWrite) (owners : List Nat)
    (h : nativeGridPlan b d a alloc p (hostPhysical d p maximum (b.reciprocal d maximum))
      pids seen=.ok (writes,owners)) :
    LogicalWrites b d a p xs (b.reciprocal d maximum)
      (footprint (parameters a alloc x) pids) writes := by
  induction pids generalizing seen writes owners with
  | nil => cases h; constructor <;> simp [footprint]
  | cons pid rest ih =>
    simp only [nativeGridPlan] at h
    cases hh : nativeProgramPlan b d a alloc p (hostPhysical d p maximum (b.reciprocal d maximum)) pid seen with
    | error issue => simp [hh] at h
    | ok head =>
      rcases head with ⟨head,nextOwners⟩
      cases ht : nativeGridPlan b d a alloc p (hostPhysical d p maximum (b.reciprocal d maximum)) rest nextOwners with
      | error issue => simp [hh,ht] at h
      | ok tail =>
        rcases tail with ⟨tail,finalOwners⟩
        simp [hh,ht] at h
        rcases h with ⟨rfl,rfl⟩
        have hs := native_program_logical b d a alloc m x logical xs p D P refines maximum
          pid (bound pid (by simp)) seen head nextOwners hh
        have ts := ih (fun q hq => bound q (by simp [hq])) nextOwners tail finalOwners ht
        exact logical_writes_append b d a p xs _ _ _ head tail hs ts

theorem native_launch_logical (b : Numerics) (d : Storage) (a : Input) (alloc : Allocations)
    (m : Memory) (x : String) (logical : List (Word d)) (xs : Nat → Word d) (p : PhysicalLayout)
    (D : Domain d a alloc m x logical xs) (P : PhysicalBinding d a alloc m x p)
    (refines : ArithmeticRefines a d p) (maximum : Word d) (cfg : Config)
    (writes : List NativeWrite)
    (h : nativeLaunchPlan b d a alloc p (hostPhysical d p maximum (b.reciprocal d maximum)) cfg=.ok writes) :
    LogicalWrites b d a p xs (b.reciprocal d maximum) (List.range a.numel) writes := by
  unfold nativeLaunchPlan at h
  cases hg : nativeGridPlan b d a alloc p (hostPhysical d p maximum (b.reciprocal d maximum))
      (nativePids a cfg) [] with
  | error issue => simp [hg,Except.map] at h
  | ok plan =>
    simp only [hg,Except.map,Except.ok.injEq] at h
    subst writes
    have hs := native_grid_logical b d a alloc m x logical xs p D P refines maximum
      (nativePids a cfg) (fun pid hp => List.mem_range.1 hp) [] plan.1 plan.2 hg
    refine ⟨hs.1,?_⟩
    intro i hi
    apply hs.2
    apply footprint_complete (parameters a alloc x) cfg i
    simpa [parameters,Parameters.n,←D.matrixCount] using List.mem_range.1 hi

theorem native_trials_logical (b : Numerics) (d : Storage) (a : Input) (alloc : Allocations)
    (m : Memory) (x : String) (logical : List (Word d)) (xs : Nat → Word d) (p : PhysicalLayout)
    (D : Domain d a alloc m x logical xs) (P : PhysicalBinding d a alloc m x p)
    (refines : ArithmeticRefines a d p) (maximum : Word d) (configs : List Config)
    (writes : List NativeWrite)
    (h : nativeTrialsPlan b d a alloc p (hostPhysical d p maximum (b.reciprocal d maximum)) configs=.ok writes) :
    (∀ w∈writes, LogicalWrite b d a p xs (b.reciprocal d maximum) w) ∧
    (configs≠[] → ∀ i, i<a.numel → ∃ w∈writes, w.outputAddress=p.output.base+i) := by
  induction configs generalizing writes with
  | nil => cases h; simp
  | cons cfg rest ih =>
    simp only [nativeTrialsPlan] at h
    cases hh : nativeLaunchPlan b d a alloc p (hostPhysical d p maximum (b.reciprocal d maximum)) cfg with
    | error issue => simp [hh] at h
    | ok head =>
      cases ht : nativeTrialsPlan b d a alloc p (hostPhysical d p maximum (b.reciprocal d maximum)) rest with
      | error issue => simp [hh,ht] at h
      | ok tail =>
        simp [hh,ht] at h
        subst writes
        have hs := native_launch_logical b d a alloc m x logical xs p D P refines maximum cfg head hh
        have ts := ih tail ht
        constructor
        · intro w hw
          rcases List.mem_append.1 hw with hw|hw
          · exact hs.1 w hw
          · exact ts.1 w hw
        · intro _ i hi
          obtain ⟨w,hw,he⟩ := hs.2 i (List.mem_range.2 hi)
          exact ⟨w,List.mem_append_left _ hw,he⟩

/-- Conditional logical transpose formula for the actual returned byte
memory. The main source theorem does not require this arithmetic refinement. -/
theorem native_logical_transpose_refinement (b : Numerics) (d : Storage) (a : Input)
    (alloc : Allocations) (m : Memory) (x : String) (logical : List (Word d))
    (xs : Nat → Word d) (p : PhysicalLayout) (D : Domain d a alloc m x logical xs)
    (P : PhysicalBinding d a alloc m x p) (refines : ArithmeticRefines a d p)
    (maximum : Word d) (trials : List Config) (last : Config) (ret : NativeReturned)
    (R : NativeResult b d a alloc p maximum trials last ret) :
    ∀ i, i<a.numel → ret.memory (p.output.base+i)=result b d (xs i) (b.reciprocal d maximum) := by
  obtain ⟨_,writes,plan,_,pointwise,_⟩ := R
  have hs := native_trials_logical b d a alloc m x logical xs p D P refines maximum _ writes plan
  intro i hi
  rw [pointwise]
  unfold nativeByteResult
  cases hf : writes.reverse.find? (fun w => decide (w.outputAddress=p.output.base+i)) with
  | none =>
    obtain ⟨w,hw,he⟩ := hs.2 (by simp) i hi
    have hn := List.find?_eq_none.1 hf w (by simpa using hw)
    simp [he] at hn
  | some w =>
    have hw : w∈writes := by simpa using List.mem_of_find?_eq_some hf
    have he : w.outputAddress=p.output.base+i := by
      simpa using (List.find?_some (p := fun v : NativeWrite => decide (v.outputAddress=p.output.base+i)) hf)
    obtain ⟨j,_,hj,hvalue⟩ := hs.1 w hw
    have eq : j=i := by omega
    subst j
    exact hvalue.symm

/-- Both host-produced scalar words remain present in the final byte memory;
the returned maximum tensor aliases the original unsqueezed maximum storage. -/
theorem native_returned_scalar_bytes (b : Numerics) (d : Storage) (a : Input)
    (alloc : Allocations) (m : Memory) (x : String) (p : PhysicalLayout)
    (P : PhysicalBinding d a alloc m x p) (maximum : Word d) (trials : List Config)
    (last : Config) (ret : NativeReturned) (R : NativeResult b d a alloc p maximum trials last ret) :
    (∀ k, k<bytes d → ret.memory (p.maximum.base+k)=some (byteOf maximum k)) ∧
    (∀ k, k<bytes (compute d) → ret.memory (p.inverse.base+k)=some (byteOf (b.reciprocal d maximum) k)) := by
  obtain ⟨_,writes,_,_,_,frame⟩ := R
  have hm : InRegion p.maximum p.maximum.base (bytes d) := by
    have hc := P.maximumCapacity
    have hv := P.maximumValid
    unfold InRegion ByteRegion.Valid at *
    omega
  have hv : InRegion p.inverse p.inverse.base (bytes (compute d)) := by
    have hc := P.inverseCapacity
    have hv := P.inverseValid
    unfold InRegion ByteRegion.Valid at *
    omega
  constructor
  · intro k hk
    rw [frame _ (region_disjoint_address p.maximum p.output P.maximumOutput _ _ k hm hk)]
    have inside : p.maximum.Contains (p.maximum.base+k) := by
      have cap := P.maximumCapacity
      unfold ByteRegion.Contains
      omega
    have outside := disjoint_range p.maximum p.inverse _ _ P.maximumInverse P.inverseCapacity inside
    simp only [hostPhysical,wordWrite,if_neg outside]
    simp [inRange,hk]
  · intro k hk
    rw [frame _ (region_disjoint_address p.inverse p.output P.inverseOutput _ _ k hv hk)]
    simp [hostPhysical,wordWrite,inRange,hk]

#axiomsClean native_returned_scalar_bytes

#axiomsClean native_grid_logical
#axiomsClean native_trials_logical
#axiomsClean native_logical_transpose_refinement

#axiomsClean loaded_at_refines
#axiomsClean native_program_logical

#axiomsClean native_read_of_bytes
#axiomsClean native_host_input_word
#axiomsClean native_load_tile_members
#axiomsClean native_tile_write_members
#guard_msgs (drop info) in
#auditModuleAxioms
end
end QuantTransposeFinite
