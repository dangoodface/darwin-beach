import json

from arena.upload import LocalTarget, R2Target, sync


def _tree(root):
    (root / "run1" / "genomes").mkdir(parents=True)
    (root / "run1" / "kernel").mkdir()
    (root / "run1" / "genomes" / "g1.bin").write_bytes(b"1" * 8)
    (root / "run1" / "kernel" / "k.wasm").write_bytes(b"\0asm")
    (root / "run1" / "latest.json").write_text('{"generation": 1}')
    (root / "index.json").write_text("[]")
    (root / "run1" / "latest.json.tmp-1").write_text("half written")


def test_local_mirror_only_sends_what_is_missing(tmp_path):
    src, dst = tmp_path / "out", tmp_path / "mirror"
    _tree(src)
    s1 = sync(src, LocalTarget(dst))
    assert sorted(s1.uploaded) == ["index.json", "run1/genomes/g1.bin", "run1/kernel/k.wasm", "run1/latest.json"]
    assert not list(dst.rglob("*tmp*"))
    # content first, indexes last
    assert s1.uploaded[-2:] == ["index.json", "run1/latest.json"]
    assert set(s1.uploaded[:2]) == {"run1/genomes/g1.bin", "run1/kernel/k.wasm"}
    s2 = sync(src, LocalTarget(dst))
    assert s2.uploaded == [] and s2.skipped == 4
    (src / "run1" / "latest.json").write_text('{"generation": 2}')
    (src / "run1" / "genomes" / "g2.bin").write_bytes(b"2" * 8)
    s3 = sync(src, LocalTarget(dst))
    assert sorted(s3.uploaded) == ["run1/genomes/g2.bin", "run1/latest.json"]
    assert json.loads((dst / "run1" / "latest.json").read_text())["generation"] == 2


def test_dry_run_writes_nothing(tmp_path):
    src, dst = tmp_path / "out", tmp_path / "mirror"
    _tree(src)
    s = sync(src, LocalTarget(dst), dry_run=True)
    assert len(s.uploaded) == 4 and not dst.exists()


class FakeS3:
    def __init__(self): self.objects, self.puts = {}, []
    def head_object(self, Bucket, Key):
        if Key not in self.objects: raise KeyError(Key)
    def put_object(self, Bucket, Key, Body, ContentType, CacheControl):
        self.objects[Key] = (Body, ContentType, CacheControl); self.puts.append(Key)


def test_r2_headers_state_and_head_check(tmp_path):
    src = tmp_path / "out"
    _tree(src)
    s3 = FakeS3()
    t = R2Target("b", s3, tmp_path / "state.json")
    sync(src, t)
    assert s3.objects["run1/genomes/g1.bin"][2].endswith("immutable")
    assert s3.objects["run1/latest.json"][1:] == ("application/json", "public, max-age=60")
    assert s3.objects["run1/kernel/k.wasm"][1] == "application/wasm"
    s3.puts.clear()
    sync(src, R2Target("b", s3, tmp_path / "state.json"))  # state survives a restart
    assert s3.puts == []
    # lost state: immutable files are found by head_object, indexes are re-sent once
    (tmp_path / "state.json").unlink()
    sync(src, R2Target("b", s3, tmp_path / "state.json"))
    assert sorted(s3.puts) == ["index.json", "run1/latest.json"]
