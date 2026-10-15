# Official comparator VM storage maintenance

This supplement records relocation of the live private VM's writable system disk from the bounded `/tmp` filesystem to independently checked space under `/root/veritile-comparator-storage-20261015`. It does not change the official comparator, its require_tools policy, source replay command, guest kernel, systemd restrictions, Landlock policy, source baseline, or proof inputs. It is not a 365-file success receipt.

The actual migration uses QEMU 6.2 drive-mirror with a full flat qcow2 target. The guest keeps running during the bulk mirror. When QMP reports ready, the host launcher and guest are paused, block-job-complete switches the active disk, and `qemu-img compare -U -f qcow2 -F qcow2` compares every virtual byte of the old backing chain against the new flat image. Only a successful compare permits resume. The same QEMU and guest boot continue. The detached original qcow2 is copied, both copies hashed, and the old path atomically becomes a symlink to the new active disk. No data belonging to another task is deleted.

`migration/receipt.json` records the real paused comparison, exact detached archive hash, pause duration and host process identity. `before.json` and `after.json` independently bind the guest boot identifier, long-lived official driver PID/start time, all 3,794 frozen F3 input hashes, exact comparator hash, original require_tools log hash and three read-only ISO mounts. `qmp-after.json` proves the same live VM uses the new flat disk and unchanged other block devices. The final outer receipt additionally rehashes the archived disk and records its immutable Debian backing image. The active disk cannot have one stable content hash while the VM runs; the meaningful content equality is the actual comparison while paused.

The internal migration receipt was written before printing its final success line. `main-receipt-postwrite-check.json` explicitly records the resulting terminal stdout hash change and rejects any other drift. The outer artifact map binds the final log bytes without altering that internal receipt.

The migration protocol was tested first on a private 8 MiB image; its real QMP transcript and compare exit are preserved. The exact-version QEMU sources in `migration/docs` substantiate flush behavior. They were obtained from the QEMU v6.2.0 sources:

- https://raw.githubusercontent.com/qemu/qemu/v6.2.0/softmmu/cpus.c (`do_vm_stop` drains and flushes block devices).
- https://raw.githubusercontent.com/qemu/qemu/v6.2.0/block/mirror.c (ready/final switch drain and flush).
- https://raw.githubusercontent.com/qemu/qemu/v6.2.0/qapi/block-core.json (drive-mirror and block-job-complete interface).

The general upstream procedure is https://www.qemu.org/docs/master/interop/live-block-operations.html. An attempted concurrent QMP client to raise mirror speed timed out before its greeting and sent no speed command; the migration retained its original 128 MiB/s bound.

Do not rerun `migration/migrate_live.py` on the current live disk: it deliberately requires the original pre-migration state. `migration/recovery-launch-command.json` is a recovery command description, explicitly not executed. Before any future restart, first wait for or preserve the official driver state and its logs. Existing `/tmp` launch paths still resolve through the symlink. The live space supervisor follows that symlink to the actual root backing filesystem; its 8 GiB low-water behavior is unchanged. Both `/tmp` and the new backing filesystem free-space values are captured separately in the outer receipt.

Independent review can rehash the outer artifact map, verify the QMP ready/stop/pivot/compare/cont transcript, verify the real compare exit/log, compare all before/after process/source/media fields, and rehash the detached archive and its base. Repeating the old/new whole-disk comparison after guest resume is invalid because new writes are expected. The full official replay remains the original F3 365 public files / 872 headlines; later workspace metadata is excluded.
