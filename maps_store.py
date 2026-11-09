"""Saved maps + per-map access lists. File-based: one <id>.pkl (results) and
one <id>.json (metadata) per map under a data directory."""
from __future__ import annotations

import json
import pickle
import re
import secrets
import time
from pathlib import Path

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,32}$")
PUBLIC = "*"


def parse_emails(text: str | list | None) -> list[str]:
    if not text:
        return []
    items = text if isinstance(text, list) else re.split(r"[,\s;]+", text)
    out = []
    for e in items:
        e = str(e).strip().lower()
        if e and e not in out:
            out.append(e)
    return out


def email_allowed(email: str | None, allowed: list[str]) -> bool:
    if PUBLIC in allowed:
        return True
    if not email:
        return False
    email = email.strip().lower()
    for a in allowed:
        if a.startswith("@") and email.endswith(a):
            return True
        if email == a:
            return True
    return False


def can_view(meta: dict, email: str | None, editors: list[str]) -> bool:
    if email and (email.lower() == meta.get("owner", "").lower() or email_allowed(email, [e for e in editors if e != PUBLIC])):
        return True
    return email_allowed(email, meta.get("viewers", []))


class MapStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, map_id):
        if not map_id or not _ID_RE.match(map_id):
            return None
        return self.root / f"{map_id}.json", self.root / f"{map_id}.pkl"

    def save(self, results, title, owner, viewers) -> str:
        map_id = secrets.token_urlsafe(9)
        meta_p, data_p = self._paths(map_id)
        data_p.write_bytes(pickle.dumps(results))
        meta_p.write_text(json.dumps({"id": map_id, "title": title, "owner": owner,
                                      "viewers": parse_emails(viewers), "created": time.time()},
                                     ensure_ascii=False))
        return map_id

    def meta(self, map_id):
        p = self._paths(map_id)
        if not p or not p[0].exists():
            return None
        return json.loads(p[0].read_text())

    def load(self, map_id):
        p = self._paths(map_id)
        if not p or not p[1].exists():
            return None
        return pickle.loads(p[1].read_bytes())  # only files this app wrote

    def update_meta(self, map_id, **changes):
        meta = self.meta(map_id)
        if meta is None:
            return None
        if "viewers" in changes:
            changes["viewers"] = parse_emails(changes["viewers"])
        meta.update(changes)
        self._paths(map_id)[0].write_text(json.dumps(meta, ensure_ascii=False))
        return meta

    def delete(self, map_id):
        p = self._paths(map_id)
        if p:
            for f in p:
                f.unlink(missing_ok=True)

    def list(self):
        metas = [json.loads(f.read_text()) for f in self.root.glob("*.json")]
        return sorted(metas, key=lambda m: m.get("created", 0), reverse=True)
