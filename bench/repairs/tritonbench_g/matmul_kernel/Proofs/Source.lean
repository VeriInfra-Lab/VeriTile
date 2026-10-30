import bench.repairs.tritonbench_g.matmul_kernel.Proofs.Memory
namespace VeriTile.Bench.Repairs.MatmulKernel
/-- Vector store statement: every lane value is computed from the program's
pre-store memory. The A/B operands of each program use current memory. -/
def scatter {α} (c : Config) (vals : Work → α) : List Work → Memory α → Memory α
  | [],m => m
  | w::ws,m => scatter c vals ws (fun r a => if r=.c ∧ a=c.C.addr w.1 w.2 then vals w else m r a)
def sourceProgram {α} (c : Config) (p : Ops α) (pid : Program) (m : Memory α) :=
  scatter c (value c p m) (programWork c pid) m

def sourcePrograms {α} (c : Config) (p : Ops α) : List Program → Memory α → Memory α
  | [],m => m
  | pid::pids,m => sourcePrograms c p pids (sourceProgram c p pid m)
def source {α} (c : Config) (p : Ops α) (m : Memory α) := sourcePrograms c p (programs c) m

theorem scatter_execute {α} (c : Config) (p : Ops α) (ws : List Work) (initial current : Memory α)
    (ha : current .a=initial .a) (hb : current .b=initial .b) :
    scatter c (value c p initial) ws current=execute c p ws current := by
  induction ws generalizing current with
  | nil => rfl
  | cons w ws ih =>
    have hv := value_inputs c p current initial ha hb w
    change scatter c (value c p initial) ws (fun r a => if r=.c ∧ a=c.C.addr w.1 w.2 then value c p initial w else current r a)=_
    rw [←hv]
    change scatter c (value c p initial) ws (step c p w current)=_
    rw [ih _ ((step_inputs c p w current).1.trans ha) ((step_inputs c p w current).2.trans hb)]
    rfl
theorem sourceProgram_execute {α} (c : Config) (p : Ops α) (pid : Program) (m : Memory α) :
    sourceProgram c p pid m=execute c p (programWork c pid) m :=
  scatter_execute c p _ m m rfl rfl
theorem sourcePrograms_execute {α} (c : Config) (p : Ops α) (pids : List Program) (m : Memory α) :
    sourcePrograms c p pids m=execute c p (pids.flatMap (programWork c)) m := by
  induction pids generalizing m with
  | nil => rfl
  | cons pid pids ih =>
    simp only [sourcePrograms,List.flatMap_cons,execute_append,sourceProgram_execute,ih]
theorem source_run {α} (c : Config) (p : Ops α) (m : Memory α) : source c p m=run c p m :=
  sourcePrograms_execute c p _ m
#axiomsClean scatter_execute
#axiomsClean source_run
end VeriTile.Bench.Repairs.MatmulKernel
