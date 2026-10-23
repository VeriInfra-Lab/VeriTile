import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.Physical
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.PhysicalLayout
noncomputable section
open Native
set_option maxHeartbeats 10000000
abbrev Bytes := Region → Nat → Codec.Byte
def view (s : Shape) (f : Flags) (m : Bytes) : Physical.Cells :=
  fun a i => m a.region (4*Geometry.offset s f a+i.val)
/-- A real aligned FP32 store in an allocation, at the dense source offset. -/
def store (s : Shape) (f : Flags) (m : Bytes) (a : Addr) (value : Codec.Cell) : Bytes :=
  fun r byte => if r=a.region ∧ byte/4=Geometry.offset s f a then
    value ⟨byte%4,Nat.mod_lt _ (by decide)⟩ else m r byte

theorem view_store (s : Shape) (f : Flags) (m : Bytes) (a b : Addr)
    (ha : Geometry.Valid s f a) (hb : Geometry.Valid s f b) (value : Codec.Cell) :
    view s f (store s f m a value) b=if b=a then value else view s f m b := by
  have div (i : Fin 4) : (4*Geometry.offset s f b+i.val)/4=Geometry.offset s f b := by have hi:=i.isLt;omega
  have rem (i : Fin 4) : (4*Geometry.offset s f b+i.val)%4=i.val := by have hi:=i.isLt;omega
  by_cases he : b=a
  · subst b
    simp only [if_true]
    funext i
    simp only [view,store,div,true_and,ite_true]
    congr 1
    exact Fin.ext (rem i)
  · have hn : ¬(b.region=a.region ∧ Geometry.offset s f b=Geometry.offset s f a) := by
      rintro ⟨hr,ho⟩
      exact he (Geometry.offset_injective s f b a hb ha hr ho)
    funext i
    simp [view,store,div,hn,he]

def At (s : Shape) (f : Flags) (r : Region) (byte : Nat) (loc : Addr×Fin 4) :=
  Geometry.Valid s f loc.1 ∧ r=loc.1.region ∧ byte=4*Geometry.offset s f loc.1+loc.2.val
theorem at_unique (s : Shape) (f : Flags) (r : Region) (byte : Nat) (a b : Addr×Fin 4)
    (ha : At s f r byte a) (hb : At s f r byte b) : a=b := by
  obtain ⟨he,hi⟩:=Index.byte_address_injective s f a.1 b.1 ha.1 hb.1 (ha.2.1.symm.trans hb.2.1) a.2 b.2
    (ha.2.2.symm.trans hb.2.2)
  exact Prod.ext he hi
/-- Re-express the eager current-cell execution at allocation byte addresses.
Uniqueness follows from the proved dense pointer map, not an assumed alias
condition for scratch cells. Unwritten bytes, including canaries, are copied. -/
def publish (s : Shape) (f : Flags) (writes : Addr → Prop) (initial : Bytes) (result : Physical.Cells) : Bytes := by
  classical
  exact fun r byte => if h : ∃ loc,At s f r byte loc ∧ writes loc.1 then result h.choose.1 h.choose.2 else initial r byte

theorem publish_at (s : Shape) (f : Flags) (writes : Addr → Prop) (initial : Bytes) (result : Physical.Cells)
    (a : Addr) (i : Fin 4) (ha : Geometry.Valid s f a) (hw : writes a) :
    publish s f writes initial result a.region (4*Geometry.offset s f a+i.val)=result a i := by
  classical
  have he : ∃ loc,At s f a.region (4*Geometry.offset s f a+i.val) loc ∧ writes loc.1 :=
    ⟨(a,i),⟨ha,rfl,rfl⟩,hw⟩
  unfold publish
  rw [dif_pos he]
  have hu:=at_unique s f a.region (4*Geometry.offset s f a+i.val) he.choose (a,i) he.choose_spec.1 ⟨ha,rfl,rfl⟩
  rw [hu]

theorem publish_frame (s : Shape) (f : Flags) (writes : Addr → Prop) (initial : Bytes) (result : Physical.Cells)
    (r : Region) (byte : Nat) (outside : ∀ loc,At s f r byte loc → ¬writes loc.1) :
    publish s f writes initial result r byte=initial r byte := by
  unfold publish
  rw [dif_neg]
  rintro ⟨loc,ha,hw⟩
  exact outside loc ha hw

theorem view_publish (s : Shape) (f : Flags) (writes : Addr → Prop) (initial : Bytes) (result : Physical.Cells)
    (frame : ∀ a,¬writes a → result a=view s f initial a) (a : Addr) (ha : Geometry.Valid s f a) :
    view s f (publish s f writes initial result) a=result a := by
  classical
  funext i
  by_cases hw : writes a
  · exact publish_at s f writes initial result a i ha hw
  · rw [frame a hw]
    apply publish_frame
    intro loc hl
    have he:=at_unique s f a.region (4*Geometry.offset s f a+i.val) loc (a,i) hl ⟨ha,rfl,rfl⟩
    simpa [he] using hw

def forward (s : Shape) (f : Flags) (g : Trees) (scale : Word) (p : Instructions) (policy : Policy) (m : Bytes) :=
  publish s f (MemoryProof.ForwardCallWrites s f) m (Physical.forward s f g scale p policy (view s f m))
def backward (s : Shape) (f : Flags) (g : Trees) (scale : Word) (p : Instructions) (policy : Policy) (m : Bytes) :=
  publish s f (MemoryProof.BackwardWrites s f) m (Physical.backward s f g scale p policy (view s f m))

theorem forward_cell (s : Shape) (f : Flags) (g : Trees) (scale : Word) (p : Instructions) (policy : Policy)
    (m : Bytes) (a : Addr) (ha : Geometry.Valid s f a) :
    view s f (forward s f g scale p policy m) a=Physical.forward s f g scale p policy (view s f m) a :=
  view_publish s f _ m _ (Physical.forward_frame s f g scale p policy (view s f m)) a ha
theorem backward_cell (s : Shape) (f : Flags) (g : Trees) (scale : Word) (p : Instructions) (policy : Policy)
    (m : Bytes) (a : Addr) (ha : Geometry.Valid s f a) :
    view s f (backward s f g scale p policy m) a=Physical.backward s f g scale p policy (view s f m) a :=
  view_publish s f _ m _ (Physical.backward_frame s f g scale p policy (view s f m)) a ha

theorem forward_transport (s : Shape) (f : Flags) (g : Trees) (scale : Word) (p : Instructions)
    (hp : Obligations p) (policy : Policy) (m : Bytes) :
    forward s f g scale p policy m=forward s f g scale software policy m := by
  unfold forward
  rw [Physical.forward_transport s f g scale p hp policy]
theorem backward_transport (s : Shape) (f : Flags) (g : Trees) (scale : Word) (p : Instructions)
    (hp : Obligations p) (policy : Policy) (m : Bytes) :
    backward s f g scale p policy m=backward s f g scale software policy m := by
  unfold backward
  rw [Physical.backward_transport s f g scale p hp policy]

#axiomsClean view_store
#axiomsClean at_unique
#axiomsClean publish_frame
#axiomsClean forward_cell
#axiomsClean backward_cell
#axiomsClean forward_transport
#axiomsClean backward_transport
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.PhysicalLayout
