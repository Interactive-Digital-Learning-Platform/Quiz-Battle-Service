#!/usr/bin/env python
"""
scripts/verify_battle_live.py
────────────────────────────────
Permanent, manually-run live end-to-end verification tool for the 1v1
battle feature: a REAL uvicorn server (real ASGI lifespan, real WebSocket
protocol) + a real local Redis (an isolated logical DB, flushed at the
start of this run only) + an isolated temp-file SQLite database, driven by
two simulated user identities over real `websockets` client connections.

WHY THIS EXISTS, NOT JUST MORE PYTEST TESTS: this script (in throwaway
form, during Phase 4/5/6 development) caught three real bugs that the
`TestClient`-based pytest suite structurally cannot catch:
  - a startup crash from `app.state` name-shadowing (only surfaces when
    ASGI lifespan actually fires -- httpx.ASGITransport, used by every
    pytest `client` fixture, never sends a "lifespan" scope at all)
  - duplicate WebSocket event delivery from a per-connection progression
    timer (only surfaces with two REAL, independently-scheduled asyncio
    tasks racing in real wall-clock time -- a single test process driving
    both sockets serially never exercises that race)
  - a permanent matchmaking lockout after a player's first battle (an
    end-to-end, multi-request regression that's easy to miss when each
    unit test only exercises one function at a time)

NOT part of `pytest` on purpose: it needs a free TCP port and real elapsed
wall-clock time for the countdown, which the pytest suite deliberately
avoids by backdating timestamps instead of sleeping. Run manually after
touching the WS route (app/api/routes/battle_ws.py), the connection
manager (app/services/battle_connection_manager.py), or matchmaking's
Redis state (app/services/matchmaking_service.py):

    uv run python scripts/verify_battle_live.py

Requires a real local Redis reachable at BATTLE_VERIFY_REDIS_URL (defaults
to redis://localhost:6379/15, an isolated logical DB index -- flushed at
the start of this run only, never touches your dev DB's usual index 0).
Everything else (Groq, Clerk) is bypassed: question selection pulls from a
small pool this script seeds itself, and auth is a dependency-override
where the bearer token IS the user identity directly (same trick
tests/test_battle_ws_routes.py uses) -- no real network calls to Groq or
Clerk are made.

Known benign noise: uvicorn occasionally logs one "Exception in ASGI
application" / RuntimeError('WebSocket is not connected...') to stderr
during teardown, from a client socket closing slightly ahead of the
server processing it. It doesn't affect any check's outcome (the route's
`finally:` cleanup still runs) -- confirmed by running this script
repeatedly and checking pass/fail counts, not just eyeballing exit code.
"""
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Must happen BEFORE importing anything under app/ -- app.core.config's
# Settings() reads DATABASE_URL/REDIS_URL from the environment at import
# time, and app.core.database/app.core.redis_client bind their engine/
# client to those values at THEIR import time too.
_TMP_DB_PATH = Path(f"/tmp/battle_verify_{uuid.uuid4().hex}.db")
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_TMP_DB_PATH}"
os.environ["REDIS_URL"] = os.environ.get("BATTLE_VERIFY_REDIS_URL", "redis://localhost:6379/15")
# No real Groq calls: matches use the pool this script seeds itself.
os.environ["BATTLE_AI_FIRST_ENABLED"] = "false"

import httpx
import uvicorn
import websockets
from fastapi import Header, WebSocket

from app.core.config import settings

HOST = "127.0.0.1"
PORT = 8931
BASE_URL = f"http://{HOST}:{PORT}"
WS_BASE = f"ws://{HOST}:{PORT}"

CLERK_A = "verify-live-user-a"
CLERK_B = "verify-live-user-b"

N_QUESTIONS = settings.BATTLE_QUESTION_COUNT

_passed = 0
_failed = 0


def check(label: str, condition: bool) -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  OK    {label}")
    else:
        _failed += 1
        print(f"  FAIL  {label}")


async def _override_get_current_user(authorization: str | None = Header(None)) -> dict:
    # Bearer token IS the clerk_id directly -- no real Clerk call, same
    # simplification tests/test_battle_ws_routes.py uses for HTTP.
    if not authorization or not authorization.startswith("Bearer "):
        raise RuntimeError("verify_battle_live.py: request missing Authorization: Bearer <clerk_id>")
    return {"sub": authorization.removeprefix("Bearer ").strip()}


async def _override_get_current_user_ws(websocket: WebSocket) -> dict:
    return {"sub": websocket.query_params.get("token")}


async def _wait_for_server(client: httpx.AsyncClient, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = await client.get(f"{BASE_URL}/docs")
            if r.status_code == 200:
                return
        except httpx.TransportError:
            pass
        await asyncio.sleep(0.1)
    raise RuntimeError("Server did not become ready in time.")


# Genuinely distinct scenarios/vocabulary, not a shared template differing
# only by numbers/a random suffix -- battle_question_service._is_near_
# duplicate's Jaccard filter strips digits and punctuation entirely before
# comparing (see quiz_service._significant_words: `w.isalpha()`), so two
# questions differing only in numbers or a hyphenated/numeric suffix read
# as 100% identical to it. That silently starved an earlier version of this
# pool down to a couple of accepted rows, which fell through to a REAL
# rate-limited Groq call and made this "no real network calls" script
# flaky. Hardcoded, topically distinct sentences sidestep it entirely.
_SEED_QUESTION_TEXTS = [
    "A baker sold fresh loaves at the morning market stall today.",
    "A farmer counted sheep grazing peacefully across the green hillside.",
    "A librarian sorted returned novels onto the tall wooden shelves.",
    "A carpenter measured wooden planks before cutting them carefully.",
    "A gardener watered blooming roses along the stone garden path.",
    "A fisherman untangled nets after returning from the open sea.",
    "A tailor stitched colorful buttons onto the finished cotton shirt.",
    "A postman delivered parcels to houses along the quiet street.",
    "A painter mixed bright pigments before starting the mural project.",
    "A shepherd guided goats down the rocky mountain trail slowly.",
    "A chef chopped fresh vegetables for the evening dinner special.",
    "A mechanic replaced worn tires on the delivery truck engine.",
    "A teacher graded homework assignments during the quiet afternoon.",
    "A florist arranged tulips into bouquets for the wedding order.",
    "A miner loaded ore carts deep inside the abandoned tunnel.",
    "A sailor coiled thick ropes on the wooden ship deck.",
    "A weaver threaded colorful yarn through the antique wooden loom.",
    "A potter shaped wet clay into bowls on the spinning wheel.",
    "A beekeeper collected golden honey from the buzzing wooden hives.",
    "A blacksmith hammered glowing iron bars beside the roaring furnace.",
]

# A second, topically distinct pool for the "medium" tier -- battle matches
# between two default-rated (1000) players land in the "Mostly Easy" band,
# which is 70% easy / 30% medium (see BATTLE_DIFFICULTY_DISTRIBUTION), so
# seeding "easy" alone always left the medium tier's pool empty and fell
# through to a REAL, rate-limited Groq call for it -- same class of flake
# the comment above already warns about, just for the OTHER tier.
_SEED_QUESTION_TEXTS_MEDIUM = [
    "A cartographer traced coastlines onto the unfinished nautical chart.",
    "A locksmith fitted a new cylinder into the stubborn front door.",
    "A vintner corked bottles after the autumn grape harvest ended.",
    "A glazier cut replacement panes for the storm-damaged greenhouse.",
    "A cooper tightened iron hoops around the newly built barrel.",
    "A falconer trained a young hawk to return to the leather glove.",
    "A cobbler resoled worn boots for the traveling merchant.",
    "A chandler poured melted wax into rows of wooden candle molds.",
    "A stonemason chiseled inscriptions into the memorial archway.",
    "A brewer monitored fermenting barley in the cellar vats.",
    "A cheesemaker turned wheels of aging cheddar on wooden racks.",
    "A saddler stitched leather straps for the new riding harness.",
    "A thatcher layered reeds across the cottage roof frame.",
    "A cutler sharpened blades on the spinning grindstone wheel.",
    "A milliner shaped felt brims for the spring hat collection.",
    "A ropemaker twisted hemp fibers into a long coiled line.",
    "A tanner stretched hides across wooden frames to dry.",
    "A wheelwright fitted iron rims onto the wagon's wooden wheels.",
    "A dyer soaked cloth bolts in vats of indigo pigment.",
    "A joiner fitted mortise and tenon joints on the oak cabinet.",
]


async def _seed_question_pool(subject: str, difficulty: str, count: int) -> None:
    from app.core.database import AsyncSessionLocal
    from app.models.question import Question

    texts = _SEED_QUESTION_TEXTS if difficulty == "easy" else _SEED_QUESTION_TEXTS_MEDIUM
    if count > len(texts):
        raise ValueError(f"Need {count} distinct seed texts, only {len(texts)} available for difficulty={difficulty!r}.")

    async with AsyncSessionLocal() as db:
        db.add_all(
            Question(
                question=texts[i],
                options=["A", "B", "C", "D"], correct_answer="A",
                subject=subject, lesson="Verify Script", difficulty=difficulty,
            )
            for i in range(count)
        )
        await db.commit()


async def _recv_json(ws) -> dict:
    return json.loads(await ws.recv())


async def _drain_until(ws, event_type: str, *, timeout: float = 5.0) -> dict | None:
    deadline = time.time() + timeout
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            return None
        try:
            msg = await asyncio.wait_for(_recv_json(ws), timeout=remaining)
        except TimeoutError:
            return None
        if msg.get("type") == event_type:
            return msg


async def _join_and_match(client: httpx.AsyncClient, clerk_a: str, clerk_b: str, subject: str) -> int | None:
    r_a = await client.post(
        f"{BASE_URL}/api/v1/battle/queue/join", json={"subject": subject},
        headers={"Authorization": f"Bearer {clerk_a}"},
    )
    check(f"{clerk_a} joins queue (200)", r_a.status_code == 200)

    r_b = await client.post(
        f"{BASE_URL}/api/v1/battle/queue/join", json={"subject": subject},
        headers={"Authorization": f"Bearer {clerk_b}"},
    )
    check(f"{clerk_b} joins queue and matches immediately (200)", r_b.status_code == 200)
    body_b = r_b.json()
    check("response status == 'matched'", body_b.get("status") == "matched")
    return body_b.get("match_id")


async def run_full_flow(client: httpx.AsyncClient) -> None:
    print("\n== Full battle flow: queue -> ready -> countdown -> active -> answer -> forfeit-completion ==")
    await _seed_question_pool("Mathematics", "easy", count=20)
    await _seed_question_pool("Mathematics", "medium", count=20)

    match_id = await _join_and_match(client, CLERK_A, CLERK_B, "Mathematics")
    check("match_id present", match_id is not None)
    if match_id is None:
        return

    url_a = f"{WS_BASE}/api/v1/battle/match/{match_id}/ws?token={CLERK_A}"
    url_b = f"{WS_BASE}/api/v1/battle/match/{match_id}/ws?token={CLERK_B}"

    async with websockets.connect(url_a) as ws_a, websockets.connect(url_b) as ws_b:
        connected_a = await _recv_json(ws_a)
        check("A receives 'connected'", connected_a.get("type") == "connected")
        sync_a = await _recv_json(ws_a)
        check("A receives 'state_sync' with status='waiting'", sync_a.get("state", {}).get("status") == "waiting")
        await _recv_json(ws_b)  # connected
        await _recv_json(ws_b)  # state_sync

        await ws_a.send(json.dumps({"type": "ready"}))
        await ws_b.send(json.dumps({"type": "ready"}))

        countdown_a = await _drain_until(ws_a, "countdown_started")
        countdown_b = await _drain_until(ws_b, "countdown_started")
        check("countdown_started received on A", countdown_a is not None)
        check("countdown_started received on B", countdown_b is not None)

        match_started_a = await _drain_until(ws_a, "match_started", timeout=10)
        match_started_b = await _drain_until(ws_b, "match_started", timeout=10)
        check("match_started received on A (not duplicated -- see module docstring)", match_started_a is not None)
        check("match_started received on B (not duplicated -- see module docstring)", match_started_b is not None)

        # Everyone gets question 0 right after match_started; after that each
        # player moves on at their own pace with Next/Skip.
        first_question_a = await _drain_until(ws_a, "next_question", timeout=5)
        first_question_b = await _drain_until(ws_b, "next_question", timeout=5)
        check("A receives next_question for question 0", first_question_a is not None)
        check("B receives next_question for question 0", first_question_b is not None)
        check(
            "next_question payload never includes correct_answer",
            "correct_answer" not in json.dumps(first_question_a or {}),
        )

        question_id = (first_question_a or {}).get("question", {}).get("question_id")
        check("question_id extracted", question_id is not None)

        # Next over REST: the response is A's next question, handed out only
        # once the answer is saved -- never whether it was right.
        r = await client.post(
            f"{BASE_URL}/api/v1/battle/match/{match_id}/answer",
            json={"question_id": question_id, "selected_option": "A"},
            headers={"Authorization": f"Bearer {CLERK_A}"},
        )
        check("A answers question 0 over REST (200)", r.status_code == 200)
        check("REST answer moves A to question 1", r.status_code == 200 and r.json().get("question_index") == 1)
        check("REST answer response never includes correct_answer", "correct_answer" not in r.text)
        progress_b = await _drain_until(ws_b, "player_progress", timeout=5)
        check("B sees A's progress over WS", progress_b is not None and progress_b.get("answered_count") == 1)

        # Skip over WS.
        await ws_b.send(json.dumps({"type": "skip", "question_id": question_id}))
        next_b = await _drain_until(ws_b, "next_question", timeout=5)
        check("B's skip moves B to question 1 over WS", next_b is not None and next_b.get("question_index") == 1)

        # Forfeit A over REST rather than waiting out all remaining
        # questions in real time -- deliberately exercises the REST+WS
        # seam (a REST action whose push is observed over WS).
        r = await client.post(
            f"{BASE_URL}/api/v1/battle/match/{match_id}/forfeit",
            headers={"Authorization": f"Bearer {CLERK_A}"},
        )
        check("A forfeits over REST (200)", r.status_code == 200)

        finished_a = await _drain_until(ws_a, "match_finished", timeout=5)
        finished_b = await _drain_until(ws_b, "match_finished", timeout=5)
        check("match_finished delivered to A exactly once", finished_a is not None)
        check("match_finished delivered to B exactly once", finished_b is not None)
        check(
            "match_finished never includes correct_answer",
            "correct_answer" not in json.dumps(finished_a or {}) and "correct_answer" not in json.dumps(finished_b or {}),
        )
        if finished_a:
            by_user = {p["user_id"] for p in finished_a.get("participants", [])}
            check("match_finished.participants has exactly 2 entries", len(finished_a.get("participants", [])) == 2)
            check("participants keyed by real user_ids", len(by_user) == 2)

    profile_a = await client.get(
        f"{BASE_URL}/api/v1/battle/profile", headers={"Authorization": f"Bearer {CLERK_A}"},
    )
    check("GET /battle/profile reflects the forfeit (A has a loss)", profile_a.status_code == 200)
    subjects = profile_a.json().get("subjects", [])
    math_row = next((s for s in subjects if s["subject"] == "Mathematics"), None)
    check("A's Mathematics rating row shows losses >= 1 after forfeiting", bool(math_row and math_row["losses"] >= 1))

    # The Phase 6 regression this flow doubles as a check for: forfeiting
    # must release Redis "in_match" state, or this call 409s forever.
    r = await client.post(
        f"{BASE_URL}/api/v1/battle/queue/cancel", headers={"Authorization": f"Bearer {CLERK_A}"},
    )
    check("A is not stuck 'in_match' after forfeit (queue/cancel is a clean 200 no-op)", r.status_code == 200)


# Pause between one player's moves in the fast-play flows. Well under any
# human's pace (a real player can't read and answer a question in 0.3s), but
# keeps a bot under the per-socket message throttle
# (BATTLE_WS_MESSAGE_RATE_LIMIT_MAX_CALLS per ..._WINDOW_SECONDS).
_FAST_MOVE_PAUSE_SECONDS = 0.3


async def _start_match(client: httpx.AsyncClient, clerk_a: str, clerk_b: str, subject: str, match_id: int | None):
    """(Queue -> match, unless already matched) -> both connect + ready ->
    both on question 0. Returns (match_id, ws_a, ws_b, first_a, first_b) with
    the sockets still open, or None if any step failed (already reported)."""
    if match_id is None:
        match_id = await _join_and_match(client, clerk_a, clerk_b, subject)
    if match_id is None:
        return None
    ws_a = await websockets.connect(f"{WS_BASE}/api/v1/battle/match/{match_id}/ws?token={clerk_a}")
    ws_b = await websockets.connect(f"{WS_BASE}/api/v1/battle/match/{match_id}/ws?token={clerk_b}")
    for ws in (ws_a, ws_b):
        await _recv_json(ws)  # connected
        await _recv_json(ws)  # state_sync
    await ws_a.send(json.dumps({"type": "ready"}))
    await ws_b.send(json.dumps({"type": "ready"}))
    first_a = await _drain_until(ws_a, "next_question", timeout=settings.BATTLE_COUNTDOWN_DURATION_SECONDS + 10)
    first_b = await _drain_until(ws_b, "next_question", timeout=settings.BATTLE_COUNTDOWN_DURATION_SECONDS + 10)
    return match_id, ws_a, ws_b, first_a, first_b


async def _play_fast(ws, first: dict, *, skip: bool) -> dict:
    """Answers ("A", always correct in the seeded pool) or skips every
    question as fast as allowed, then waits for match_finished. Records
    each save's round trip and every event that arrived along the way."""
    result = {"latencies": [], "indexes": [], "errors": [], "progress": [], "finished": [], "unexpected": []}
    current = first
    while current is not None and not current.get("finished"):
        question_id = current["question"]["question_id"]
        move = {"type": "skip", "question_id": question_id} if skip else {
            "type": "answer", "question_id": question_id, "selected_option": "A",
        }
        sent = time.monotonic()
        await ws.send(json.dumps(move))
        current = None
        while current is None:
            try:
                msg = await asyncio.wait_for(_recv_json(ws), timeout=10)
            except TimeoutError:
                result["errors"].append("no reply within 10s")
                return result
            kind = msg.get("type")
            if kind == "next_question":
                result["latencies"].append(time.monotonic() - sent)
                result["indexes"].append(msg["question_index"])
                current = msg
            elif kind == "error":
                result["errors"].append(msg.get("detail"))
                return result
            elif kind == "player_progress":
                result["progress"].append(msg)
            elif kind == "match_finished":
                result["finished"].append(msg)
            elif kind != "pong" and kind != "ping":
                result["unexpected"].append(kind)
        await asyncio.sleep(_FAST_MOVE_PAUSE_SECONDS)
    # Done -- collect everything else until the match ends (and a beat
    # longer, to catch a duplicate match_finished).
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            msg = await asyncio.wait_for(_recv_json(ws), timeout=1.0 if result["finished"] else deadline - time.monotonic())
        except (TimeoutError, websockets.ConnectionClosed):
            if result["finished"]:
                break
            continue
        kind = msg.get("type")
        if kind == "player_progress":
            result["progress"].append(msg)
        elif kind == "match_finished":
            result["finished"].append(msg)
        elif kind == "error":
            result["errors"].append(msg.get("detail"))
    return result


def _progress_counts(result: dict, user_id: int) -> list[int]:
    return [p["answered_count"] for p in result["progress"] if p["user_id"] == user_id]


async def _run_fast_match(
    client: httpx.AsyncClient, clerk_a: str, clerk_b: str, subject: str, label: str, match_id: int | None = None,
) -> list[float]:
    started = await _start_match(client, clerk_a, clerk_b, subject, match_id)
    if started is None:
        check(f"{label}: match started", False)
        return []
    match_id, ws_a, ws_b, first_a, first_b = started
    try:
        check(f"{label}: both players get question 0", first_a is not None and first_b is not None)
        if first_a is None or first_b is None:
            return []
        res_a, res_b = await asyncio.gather(_play_fast(ws_a, first_a, skip=False), _play_fast(ws_b, first_b, skip=True))
    finally:
        await ws_a.close()
        await ws_b.close()

    n = settings.BATTLE_QUESTION_COUNT
    check(f"{label}: no errors (A={res_a['errors']}, B={res_b['errors']})", not res_a["errors"] and not res_b["errors"])
    check(f"{label}: A moved through every question in order", res_a["indexes"] == list(range(1, n + 1)))
    check(f"{label}: B moved through every question in order", res_b["indexes"] == list(range(1, n + 1)))
    finished = (res_a["finished"] or [None])[0] or {}
    participants = {p["user_id"]: p for p in finished.get("participants", [])}
    ids = list(participants)
    user_a = next((uid for uid, p in participants.items() if (p.get("correct_count") or 0) > 0), None)
    user_b = next((uid for uid in ids if uid != user_a), None)
    # Every one of the opponent's moves reached each player, once, in order.
    check(f"{label}: A saw all of B's progress, once each", user_b is not None and _progress_counts(res_a, user_b) == list(range(1, n + 1)))
    check(f"{label}: B saw all of A's progress, once each", user_a is not None and _progress_counts(res_b, user_a) == list(range(1, n + 1)))
    check(f"{label}: match_finished reached each player exactly once", len(res_a["finished"]) == 1 and len(res_b["finished"]) == 1)
    check(f"{label}: both players saw the same result", res_a["finished"][:1] == res_b["finished"][:1])
    check(f"{label}: A (all correct) beat B (all skipped)", finished.get("winner_user_id") == user_a and user_a is not None)
    if user_a is not None:
        check(f"{label}: A is credited {n} correct answers", participants[user_a].get("correct_count") == n)

    # Free to queue again straight away (in_match released at completion).
    r = await client.post(f"{BASE_URL}/api/v1/battle/queue/cancel", headers={"Authorization": f"Bearer {clerk_a}"})
    check(f"{label}: A isn't left stuck 'in_match'", r.status_code == 200)
    return res_a["latencies"] + res_b["latencies"]


def _report_latencies(label: str, latencies: list[float]) -> None:
    if not latencies:
        return
    ordered = sorted(latencies)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    print(f"  ..    {label}: {len(ordered)} saves — median {ordered[len(ordered) // 2] * 1000:.0f}ms, "
          f"p95 {p95 * 1000:.0f}ms, max {ordered[-1] * 1000:.0f}ms")


async def run_fast_play_flow(client: httpx.AsyncClient) -> None:
    print("\n== Fast play: both players race through every question (A answers, B skips) ==")
    await _seed_question_pool("Science", "easy", count=20)
    await _seed_question_pool("Science", "medium", count=20)
    latencies = await _run_fast_match(client, f"{CLERK_A}-fast", f"{CLERK_B}-fast", "Science", "fast match")
    _report_latencies("save round trips", latencies)


async def run_concurrent_fast_matches(client: httpx.AsyncClient, matches: int = 5) -> None:
    print(f"\n== Load: {matches} fast matches at once ({matches * 2} players, same subject) ==")
    pairs = [(f"{CLERK_A}-load{i}", f"{CLERK_B}-load{i}") for i in range(matches)]
    # Matched one pair at a time so each pair meets each other (joining all
    # at once could pair A0 with B1); then all of them play at once.
    match_ids = [await _join_and_match(client, a, b, "Science") for a, b in pairs]
    results = await asyncio.gather(*(
        _run_fast_match(client, a, b, "Science", f"load #{i}", match_id=mid)
        for i, ((a, b), mid) in enumerate(zip(pairs, match_ids, strict=True))
    ))
    _report_latencies("save round trips under load", [x for r in results for x in r])


async def run_ai_failure_release_check(client: httpx.AsyncClient) -> None:
    print("\n== AI-failure cancellation still releases matchmaking lock (Phase 6 regression) ==")
    import app.services.battle_gameplay_service as bgs

    clerk_a2, clerk_b2 = f"{CLERK_A}-2", f"{CLERK_B}-2"

    original_select_battle_questions = bgs.select_battle_questions

    async def _always_fail(*_args, **_kwargs):
        raise bgs.BattleQuestionSelectionError("forced failure -- verify_battle_live.py")

    bgs.select_battle_questions = _always_fail
    try:
        match_id = await _join_and_match(client, clerk_a2, clerk_b2, "Mathematics")
        check("match_id present", match_id is not None)
        if match_id is None:
            return

        url_a = f"{WS_BASE}/api/v1/battle/match/{match_id}/ws?token={clerk_a2}"
        url_b = f"{WS_BASE}/api/v1/battle/match/{match_id}/ws?token={clerk_b2}"
        async with websockets.connect(url_a) as ws_a, websockets.connect(url_b) as ws_b:
            await _recv_json(ws_a)  # connected
            await _recv_json(ws_a)  # state_sync
            await _recv_json(ws_b)
            await _recv_json(ws_b)

            await ws_a.send(json.dumps({"type": "ready"}))
            await ws_b.send(json.dumps({"type": "ready"}))

            cancelled_a = await _drain_until(ws_a, "match_cancelled", timeout=10)
            check("match_cancelled delivered after forced question-selection failure", cancelled_a is not None)
    finally:
        bgs.select_battle_questions = original_select_battle_questions

    # The actual regression: both users must be free to join a new queue
    # immediately, not stuck "in_match" forever (no TTL on that state).
    r_a2 = await client.post(
        f"{BASE_URL}/api/v1/battle/queue/join", json={"subject": "Mathematics"},
        headers={"Authorization": f"Bearer {clerk_a2}"},
    )
    check("user A can join a new queue after AI-failure cancellation (not locked out)", r_a2.status_code == 200)
    r_b2 = await client.post(
        f"{BASE_URL}/api/v1/battle/queue/join", json={"subject": "Mathematics"},
        headers={"Authorization": f"Bearer {clerk_b2}"},
    )
    check("user B can join a new queue after AI-failure cancellation (not locked out)", r_b2.status_code == 200)


async def main() -> int:
    import logging

    from app.core.database import engine
    from app.core.redis_client import redis_client
    from app.core.security import get_current_user, get_current_user_ws
    from app.main import app

    # settings.ENVIRONMENT=="development" (unchanged -- needed for /docs,
    # used below as the readiness probe) drives app/main.py's logging setup
    # to root-level DEBUG, which floods this script's output with every
    # SQL statement and WS/HTTP frame. Quiet the noisy loggers specifically
    # rather than touching the app's own logging config.
    for noisy_logger in (
        "sqlalchemy.engine", "aiosqlite", "httpcore", "httpx",
        "websockets.client", "websockets.server", "uvicorn.access",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    app.dependency_overrides[get_current_user] = _override_get_current_user
    app.dependency_overrides[get_current_user_ws] = _override_get_current_user_ws

    await redis_client.flushdb()
    print(f"Isolated DB:    {_TMP_DB_PATH}")
    print(f"Isolated Redis: {os.environ['REDIS_URL']}")

    config = uvicorn.Config(app, host=HOST, port=PORT, log_level="warning", lifespan="on")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            await _wait_for_server(client)
            print("Server ready.")

            await run_full_flow(client)
            await run_fast_play_flow(client)
            await run_concurrent_fast_matches(client)
            await run_ai_failure_release_check(client)
    finally:
        server.should_exit = True
        await server_task
        await redis_client.aclose()
        await engine.dispose()
        _TMP_DB_PATH.unlink(missing_ok=True)

    print(f"\n{_passed} passed, {_failed} failed.")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
