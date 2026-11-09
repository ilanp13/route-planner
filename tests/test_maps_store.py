import maps_store as ms


def test_parse_emails_normalizes_and_dedupes():
    assert ms.parse_emails(" A@x.com, b@y.com\na@X.com ;@school.org.il") == ["a@x.com", "b@y.com", "@school.org.il"]


def test_email_allowed_exact_domain_and_public():
    assert ms.email_allowed("a@x.com", ["a@x.com"])
    assert ms.email_allowed("Kid@School.org.il", ["@school.org.il"])
    assert not ms.email_allowed("a@evil-school.org.il", ["@school.org.il"])
    assert ms.email_allowed(None, ["*"])
    assert not ms.email_allowed(None, ["a@x.com"])
    assert not ms.email_allowed("", [])


def test_editors_can_view_any_map(tmp_path):
    store = ms.MapStore(tmp_path)
    mid = store.save({"x": 1}, title="t", owner="ed@x.com", viewers=[])
    meta = store.meta(mid)
    assert ms.can_view(meta, "ed@x.com", editors=[])        # owner
    assert ms.can_view(meta, "boss@x.com", editors=["boss@x.com"])
    assert not ms.can_view(meta, "parent@x.com", editors=[])


def test_store_roundtrip_and_update_viewers(tmp_path):
    store = ms.MapStore(tmp_path)
    mid = store.save({"payload": [1, 2]}, title="Yavne", owner="ed@x.com", viewers=["p@x.com"])
    assert store.load(mid) == {"payload": [1, 2]}
    store.update_meta(mid, viewers=["*"])
    assert store.meta(mid)["viewers"] == ["*"]
    assert [m["id"] for m in store.list()] == [mid]


def test_store_rejects_path_traversal_ids(tmp_path):
    store = ms.MapStore(tmp_path)
    assert store.meta("../../etc/passwd") is None
    assert store.load("../x") is None


def test_delete(tmp_path):
    store = ms.MapStore(tmp_path)
    mid = store.save({}, title="t", owner="o", viewers=[])
    store.delete(mid)
    assert store.meta(mid) is None
