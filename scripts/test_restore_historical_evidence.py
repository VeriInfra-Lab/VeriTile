"""Archive recovery must preserve exact evidence and reject unsafe inputs."""
import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

from restore_historical_evidence import COMMIT, destination, missing_inputs, stage_tar, valid


NAME = 'bench/review-checkpoints/test/receipt.json'
DATA = b'{"status":"historical"}\n'


def binding(data):
    return {'bytes': len(data), 'git_blob': hashlib.sha1(
        f'blob {len(data)}\0'.encode() + data).hexdigest()}


def archive(members):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as tar:
        for name, data, kind in members:
            info = tarfile.TarInfo(f'VeriTile-{COMMIT}/{name}')
            info.type = kind
            info.size = len(data) if kind == tarfile.REGTYPE else 0
            if kind == tarfile.SYMTYPE:
                info.linkname = '/etc/passwd'
            tar.addfile(info, io.BytesIO(data) if info.isfile() else None)
    stream.seek(0)
    return stream


class HistoricalEvidence(unittest.TestCase):
    def test_exact_blob_restored_from_tar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage_tar(archive([(NAME, DATA, tarfile.REGTYPE)]), root, {NAME: binding(DATA)})
            self.assertEqual(DATA, (root / NAME).read_bytes())
            self.assertEqual({}, missing_inputs(root, {NAME: binding(DATA)}))

    def test_changed_same_size_blob_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'Git blob differs'):
                stage_tar(archive([(NAME, DATA.replace(b'historical', b'fabricated'), tarfile.REGTYPE)]),
                          Path(tmp), {NAME: binding(DATA)})

    def test_missing_duplicate_and_link_members_rejected(self):
        for members in ([], [(NAME, DATA, tarfile.REGTYPE)] * 2,
                        [(NAME, b'', tarfile.SYMTYPE)]):
            with self.subTest(members=members), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    stage_tar(archive(members), Path(tmp), {NAME: binding(DATA)})

    def test_non_manifest_paths_are_not_extracted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage_tar(archive([('../../outside', DATA, tarfile.REGTYPE),
                               (NAME, DATA, tarfile.REGTYPE)]), root, {NAME: binding(DATA)})
            self.assertEqual([root / NAME], [p for p in root.rglob('*') if p.is_file()])

    def test_manifest_cannot_escape_or_write_current_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for path in ('/absolute', 'bench/review-checkpoints/../../source',
                         'VeriTile/Spec.lean', 'bench//review-checkpoints/x'):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    destination(root, path)

    def test_symlink_escape_and_existing_changes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            root = Path(tmp)
            (root / 'bench').symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                destination(root, NAME)
            (root / 'bench').unlink()
            path = root / NAME
            path.parent.mkdir(parents=True)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'refusing to overwrite'):
                missing_inputs(root, {NAME: binding(DATA)})
            self.assertEqual(b'changed', path.read_bytes())
            self.assertFalse(valid(path, binding(DATA)))


if __name__ == '__main__':
    unittest.main()
