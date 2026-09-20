"""临时写入租约端到端集成测试（每个用例使用独立临时 SQLite 文件）。

覆盖：获取 / 同持有者重复获取 / 续期 / 释放 / 查询，fencing token 严格递增与
过期接管，PATCH 与原子发布项的 lease_token 强制（409 lease_conflict 且不占用
request_id / release_id），过期后旧 token 永久失效，以及无租约写入的兼容性。
"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


def _create(client, content, doc_id="cfg"):
    r = client.post("/documents", json={"id": doc_id, "content": content})
    assert r.status_code == 201, r.text
    return r.json()


def _acquire(client, doc_id, holder, ttl=None):
    body = {"holder": holder}
    if ttl is not None:
        body["ttl_seconds"] = ttl
    return client.post(f"/documents/{doc_id}/lease", json=body)


def _renew(client, doc_id, holder, token, ttl=None):
    body = {"holder": holder, "lease_token": token}
    if ttl is not None:
        body["ttl_seconds"] = ttl
    return client.post(f"/documents/{doc_id}/lease/renew", json=body)


def _release_lease(client, doc_id, holder, token):
    return client.post(
        f"/documents/{doc_id}/lease/release",
        json={"holder": holder, "lease_token": token},
    )


def _patch(client, doc_id, base_revision, request_id, patch, lease_token=None):
    body = {"base_revision": base_revision, "request_id": request_id, "patch": patch}
    if lease_token is not None:
        body["lease_token"] = lease_token
    return client.patch(f"/documents/{doc_id}", json=body)


def _release(client, release_id, updates):
    return client.post("/releases", json={"release_id": release_id, "updates": updates})


def _item(doc_id, base_revision, patch, lease_token=None):
    body = {"document_id": doc_id, "base_revision": base_revision, "patch": patch}
    if lease_token is not None:
        body["lease_token"] = lease_token
    return body


# ---------------- 获取（acquire） ----------------


def test_acquire_creates_lease_with_initial_fencing_token(client):
    _create(client, {"a": 1})
    r = _acquire(client, "cfg", "deploy-bot", ttl=120)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["document_id"] == "cfg"
    assert body["holder"] == "deploy-bot"
    assert body["lease_token"] == 1
    assert body["ttl_seconds"] == 120
    assert body["new_lease"] is True
    assert body["acquired_at"] <= body["renewed_at"] <= body["expires_at"]


def test_acquire_uses_default_ttl_when_omitted(client):
    _create(client, {"a": 1})
    r = _acquire(client, "cfg", "deploy-bot")
    assert r.status_code == 201, r.text
    assert r.json()["ttl_seconds"] == settings.lease_default_ttl_seconds


def test_reacquire_by_same_holder_keeps_token_and_refreshes_expiry(client):
    _create(client, {"a": 1})
    first = _acquire(client, "cfg", "deploy-bot", ttl=60).json()
    r = _acquire(client, "cfg", "deploy-bot", ttl=300)
    assert r.status_code == 200, r.text
    second = r.json()
    assert second["new_lease"] is False
    assert second["lease_token"] == first["lease_token"]
    assert second["acquired_at"] == first["acquired_at"]
    assert second["expires_at"] >= first["expires_at"]
    assert second["ttl_seconds"] == 300


def test_acquire_by_other_holder_while_active_conflicts(client):
    _create(client, {"a": 1})
    _acquire(client, "cfg", "deploy-bot", ttl=60)
    r = _acquire(client, "cfg", "other-bot", ttl=60)
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["error"] == "lease_conflict"
    assert detail["document_id"] == "cfg"
    assert detail["holder"] == "deploy-bot"
    assert "expires_at" in detail
    # 冲突响应不泄露当前 fencing token
    assert "lease_token" not in detail


def test_acquire_on_missing_document_returns_404(client):
    r = _acquire(client, "ghost", "deploy-bot")
    assert r.status_code == 404
    assert r.json()["detail"]["error"] == "not_found"


# ---------------- 续期（renew） ----------------


def test_renew_keeps_token_and_extends_expiry(client):
    _create(client, {"a": 1})
    lease = _acquire(client, "cfg", "deploy-bot", ttl=60).json()
    r = _renew(client, "cfg", "deploy-bot", lease["lease_token"], ttl=600)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["lease_token"] == lease["lease_token"]
    assert body["acquired_at"] == lease["acquired_at"]
    assert body["renewed_at"] >= lease["renewed_at"]
    assert body["expires_at"] >= lease["expires_at"]
    assert body["ttl_seconds"] == 600


def test_renew_with_wrong_token_conflicts(client):
    _create(client, {"a": 1})
    lease = _acquire(client, "cfg", "deploy-bot").json()
    r = _renew(client, "cfg", "deploy-bot", lease["lease_token"] + 100)
    assert r.status_code == 409
    assert r.json()["detail"]["error"] == "lease_conflict"


def test_renew_with_wrong_holder_conflicts(client):
    _create(client, {"a": 1})
    lease = _acquire(client, "cfg", "deploy-bot").json()
    r = _renew(client, "cfg", "other-bot", lease["lease_token"])
    assert r.status_code == 409
    assert r.json()["detail"]["error"] == "lease_conflict"


def test_renew_without_lease_returns_404(client):
    _create(client, {"a": 1})
    r = _renew(client, "cfg", "deploy-bot", 1)
    assert r.status_code == 404
    assert r.json()["detail"]["error"] == "lease_not_found"


# ---------------- 释放（release） ----------------


def test_release_then_leaseless_writes_work_again(client):
    _create(client, {"a": 1})
    lease = _acquire(client, "cfg", "deploy-bot").json()
    token = lease["lease_token"]

    r = _release_lease(client, "cfg", "deploy-bot", token)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["released"] is True
    assert body["lease_token"] == token

    # 查询：已无有效租约
    assert client.get("/documents/cfg/lease").status_code == 404
    # 无租约写入恢复兼容
    r = _patch(client, "cfg", 1, "req-after-release", {"a": 2})
    assert r.status_code == 200, r.text
    # 重复释放：无有效租约
    r = _release_lease(client, "cfg", "deploy-bot", token)
    assert r.status_code == 404
    assert r.json()["detail"]["error"] == "lease_not_found"


def test_release_with_wrong_token_or_holder_conflicts(client):
    _create(client, {"a": 1})
    lease = _acquire(client, "cfg", "deploy-bot").json()
    token = lease["lease_token"]

    r = _release_lease(client, "cfg", "deploy-bot", token + 1)
    assert r.status_code == 409
    assert r.json()["detail"]["error"] == "lease_conflict"
    r = _release_lease(client, "cfg", "other-bot", token)
    assert r.status_code == 409
    # 租约仍然有效
    assert client.get("/documents/cfg/lease").status_code == 200


def test_fencing_token_strictly_increases_across_holders(client):
    _create(client, {"a": 1})
    first = _acquire(client, "cfg", "bot-a").json()
    assert first["lease_token"] == 1
    _release_lease(client, "cfg", "bot-a", 1)
    second = _acquire(client, "cfg", "bot-b").json()
    assert second["lease_token"] == 2
    _release_lease(client, "cfg", "bot-b", 2)
    third = _acquire(client, "cfg", "bot-a").json()
    assert third["lease_token"] == 3


# ---------------- 查询（query） ----------------


def test_query_current_lease(client):
    _create(client, {"a": 1})
    # 从未获取过
    r = client.get("/documents/cfg/lease")
    assert r.status_code == 404
    assert r.json()["detail"]["error"] == "lease_not_found"

    lease = _acquire(client, "cfg", "deploy-bot", ttl=90).json()
    r = client.get("/documents/cfg/lease")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["document_id"] == "cfg"
    assert body["holder"] == "deploy-bot"
    assert body["lease_token"] == lease["lease_token"]
    assert body["expires_at"] == lease["expires_at"]


def test_lease_endpoints_require_existing_document(client):
    assert client.get("/documents/ghost/lease").status_code == 404
    assert _renew(client, "ghost", "bot", 1).status_code == 404
    assert _release_lease(client, "ghost", "bot", 1).status_code == 404
    for r in (
        client.get("/documents/ghost/lease"),
        _renew(client, "ghost", "bot", 1),
        _release_lease(client, "ghost", "bot", 1),
    ):
        assert r.json()["detail"]["error"] == "not_found"


# ---------------- 过期与接管 ----------------


def test_expired_lease_lifecycle(client):
    _create(client, {"a": 1})
    lease = _acquire(client, "cfg", "bot-a", ttl=1).json()
    old_token = lease["lease_token"]
    time.sleep(1.1)

    # 过期后：续期失败（旧 token 永久失效）、查询为 404
    r = _renew(client, "cfg", "bot-a", old_token)
    assert r.status_code == 409
    assert r.json()["detail"]["error"] == "lease_conflict"
    assert client.get("/documents/cfg/lease").status_code == 404

    # 新持有者可立即接管，token 严格递增
    r = _acquire(client, "cfg", "bot-b", ttl=60)
    assert r.status_code == 201, r.text
    new_lease = r.json()
    assert new_lease["lease_token"] == old_token + 1
    assert new_lease["new_lease"] is True

    # 旧 token 写入被拒绝，新 token 放行
    r = _patch(client, "cfg", 1, "req-old", {"a": 2}, lease_token=old_token)
    assert r.status_code == 409
    assert r.json()["detail"]["error"] == "lease_conflict"
    r = _patch(client, "cfg", 1, "req-new", {"a": 2},
               lease_token=new_lease["lease_token"])
    assert r.status_code == 200, r.text
    assert r.json()["content"] == {"a": 2}


def test_stale_token_ignored_when_no_active_lease(client):
    _create(client, {"a": 1})
    lease = _acquire(client, "cfg", "bot-a", ttl=1).json()
    time.sleep(1.1)
    # 租约已过期且无人接管：写入不受限（携带过期 token 也被忽略，保持兼容）
    r = _patch(client, "cfg", 1, "req-stale", {"a": 2},
               lease_token=lease["lease_token"])
    assert r.status_code == 200, r.text
    r = _patch(client, "cfg", 2, "req-plain", {"a": 3})
    assert r.status_code == 200, r.text


# ---------------- PATCH 写入强制 ----------------


def test_patch_requires_matching_token_while_lease_active(client):
    _create(client, {"a": 1})
    lease = _acquire(client, "cfg", "deploy-bot").json()
    token = lease["lease_token"]

    # 未携带 token
    r = _patch(client, "cfg", 1, "req-1", {"a": 10})
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["error"] == "lease_conflict"
    assert detail["document_id"] == "cfg"
    assert detail["request_id"] == "req-1"
    assert detail["holder"] == "deploy-bot"
    assert "lease_token" not in detail

    # token 不匹配
    r = _patch(client, "cfg", 1, "req-2", {"a": 10}, lease_token=token + 1)
    assert r.status_code == 409
    assert r.json()["detail"]["error"] == "lease_conflict"

    # 文档未被修改
    doc = client.get("/documents/cfg").json()
    assert doc["revision"] == 1
    assert doc["content"] == {"a": 1}

    # 匹配 token 放行
    r = _patch(client, "cfg", 1, "req-3", {"a": 10}, lease_token=token)
    assert r.status_code == 200, r.text
    assert r.json()["revision"] == 2
    assert r.json()["content"] == {"a": 10}


def test_lease_conflict_does_not_consume_request_id(client):
    _create(client, {"a": 1})
    lease = _acquire(client, "cfg", "deploy-bot").json()

    r = _patch(client, "cfg", 1, "req-retry", {"a": 10})
    assert r.status_code == 409
    # 同一 request_id 修正后重试：不视为 request_id_conflict
    r = _patch(client, "cfg", 1, "req-retry", {"a": 10},
               lease_token=lease["lease_token"])
    assert r.status_code == 200, r.text
    assert r.json()["idempotent"] is False


def test_lease_check_precedes_guards(client):
    _create(client, {"a": 1, "mode": "on"})
    _acquire(client, "cfg", "deploy-bot")
    # guards 本也会失败，但租约检查更靠前
    r = client.patch(
        "/documents/cfg",
        json={
            "base_revision": 1,
            "request_id": "req-order",
            "patch": {"a": 2},
            "guards": [{"path": "/mode", "expected": "off"}],
        },
    )
    assert r.status_code == 409
    assert r.json()["detail"]["error"] == "lease_conflict"


def test_idempotent_replay_does_not_require_token(client):
    _create(client, {"a": 1})
    lease = _acquire(client, "cfg", "deploy-bot").json()
    r = _patch(client, "cfg", 1, "req-idem", {"a": 10},
               lease_token=lease["lease_token"])
    assert r.status_code == 200, r.text

    # 相同 request_id + 相同内容重放：即使不带 token 也直接返回原结果（不发生写入）
    r = _patch(client, "cfg", 1, "req-idem", {"a": 10})
    assert r.status_code == 200, r.text
    assert r.json()["idempotent"] is True
    assert r.json()["revision"] == 2


def test_leaseless_patch_stays_compatible(client):
    _create(client, {"a": 1})
    # 无租约：不带 token 与携带 token 的写入都放行（token 被忽略）
    r = _patch(client, "cfg", 1, "req-1", {"a": 2})
    assert r.status_code == 200, r.text
    r = _patch(client, "cfg", 2, "req-2", {"a": 3}, lease_token=999)
    assert r.status_code == 200, r.text
    assert r.json()["content"] == {"a": 3}


# ---------------- 原子发布项的 lease_token ----------------


def test_release_item_requires_matching_token(client):
    _create(client, {"a": 1}, "d1")
    _create(client, {"b": 1}, "d2")
    lease = _acquire(client, "d1", "deploy-bot").json()

    # d1 项未携带 token → 整批失败，lease_conflict 聚合进 failures
    r = _release(client, "rel-1", [
        _item("d1", 1, {"a": 10}),
        _item("d2", 1, {"b": 20}),
    ])
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "release_failed"
    assert detail["release_id"] == "rel-1"
    assert len(detail["failures"]) == 1
    failure = detail["failures"][0]
    assert failure["index"] == 0
    assert failure["document_id"] == "d1"
    assert failure["error"] == "lease_conflict"
    assert failure["status_code"] == 409
    assert failure["holder"] == "deploy-bot"

    # 整批回滚：两个文档都没有新修订
    assert client.get("/documents/d1").json()["revision"] == 1
    assert client.get("/documents/d2").json()["revision"] == 1

    # release_id 未被占用：修正后可用同一 ID 重试
    r = _release(client, "rel-1", [
        _item("d1", 1, {"a": 10}, lease_token=lease["lease_token"]),
        _item("d2", 1, {"b": 20}),
    ])
    assert r.status_code == 200, r.text
    assert r.json()["idempotent"] is False
    assert client.get("/documents/d1").json()["content"] == {"a": 10}
    assert client.get("/documents/d2").json()["content"] == {"b": 20}


def test_release_item_with_matching_token_succeeds(client):
    _create(client, {"a": 1}, "d1")
    _create(client, {"b": 1}, "d2")
    lease = _acquire(client, "d1", "deploy-bot").json()
    r = _release(client, "rel-1", [
        _item("d1", 1, {"a": 10}, lease_token=lease["lease_token"]),
        _item("d2", 1, {"b": 20}),
    ])
    assert r.status_code == 200, r.text
    results = {x["document_id"]: x for x in r.json()["results"]}
    assert results["d1"]["content"] == {"a": 10}
    assert results["d2"]["content"] == {"b": 20}


# ---------------- 模型层校验 ----------------


def test_lease_request_validation(client):
    _create(client, {"a": 1})
    # ttl_seconds 必须为正整数
    assert _acquire(client, "cfg", "bot", ttl=0).status_code == 422
    assert _acquire(client, "cfg", "bot", ttl=3601).status_code == 422
    # holder 必填
    assert client.post("/documents/cfg/lease", json={}).status_code == 422
    # renew / release 必须携带 lease_token
    assert client.post(
        "/documents/cfg/lease/renew", json={"holder": "bot"}
    ).status_code == 422
    assert client.post(
        "/documents/cfg/lease/release", json={"holder": "bot"}
    ).status_code == 422
    # PATCH 的 lease_token 必须为正整数
    r = _patch(client, "cfg", 1, "req-bad-token", {"a": 2}, lease_token=0)
    assert r.status_code == 422
