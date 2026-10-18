from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Quiz Battle Service"
    ENVIRONMENT: str = "development"

    # Must point at the SAME Postgres/Neon instance as Personalized-Quiz-
    # Service -- this service reads/writes its users/questions tables
    # directly via the thin mirror models in app/models/. See this repo's
    # README "Shared database" section.
    DATABASE_URL: str
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    # Log every SQL statement. Off by default even in development: on a
    # remote DB (Neon) the extra per-statement logging noticeably slows the
    # many small queries behind each battle Next/Skip.
    DB_ECHO: bool = False
    # Longest any single SQL statement may run before it's abandoned with a
    # TimeoutError -- a stalled connection then fails (and is retried by its
    # caller) instead of hanging that request or background loop forever.
    DB_COMMAND_TIMEOUT_SECONDS: float = 15.0

    # Diagnostic (app/core/loop_watchdog.py): warn when the event loop was
    # blocked this long -- i.e. this whole process stopped running.
    LOOP_WATCHDOG_ENABLED: bool = True
    LOOP_WATCHDOG_INTERVAL_SECONDS: float = 0.5
    LOOP_WATCHDOG_WARN_SECONDS: float = 1.0

    # Own Groq client (app/services/groq_service.py) -- generates fresh
    # questions to top up the shared question pool when it runs short for a
    # given (subject, difficulty), same key/model as Personalized-Quiz-
    # Service points at (a separate value is fine too, both just need a
    # valid Groq API key).
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama3-70b-8192"
    GROQ_MAX_RETRIES: int = 2
    GROQ_RETRY_BASE_DELAY_SECONDS: float = 1.0

    # Same Clerk project as Personalized-Quiz-Service -- both services
    # authenticate the same users.
    CLERK_JWKS_URL: str
    CLERK_ISSUER: str
    CLERK_AUDIENCE: str | None = None
    # Comma-separated origins allowed as a token's `azp` claim (e.g. "https://app.example.com"). Empty = no check.
    CLERK_AUTHORIZED_PARTIES: str = ""
    AUTH_BYPASS: bool = False

    # ── Competitive Battle System: rating/leaderboard ──────────────────────
    # Rating updates are triggered by real battle completion (see
    # battle_gameplay_service._finalize_match / forfeit_match).
    BATTLE_INITIAL_RATING: int = 0
    # Standard Elo K-factor. Flat for every player/subject -- no per-tier or
    # provisional-match K variation.
    BATTLE_ELO_K_FACTOR: float = 32.0
    # Floor so a long losing streak against much-higher-rated opponents
    # can't push rating negative. Purely defensive -- normal K=32 play
    # essentially never reaches it.
    BATTLE_MIN_RATING: int = 0
    # Anti-farming: only the first N completed matches between the same two
    # players within the window move either player's rating. Later ones still
    # record the result (wins/losses, history) but leave ratings unchanged --
    # stops someone queueing a second account into a quiet subject and
    # forfeiting it over and over to pump their main's rating.
    BATTLE_REPEAT_OPPONENT_RATED_LIMIT: int = 3
    BATTLE_REPEAT_OPPONENT_WINDOW_HOURS: int = 24

    # League thresholds -- minimum rating (inclusive) to be in that league.
    # Below BATTLE_LEAGUE_SILVER_MIN is Bronze. Keep these strictly
    # increasing; nothing enforces that automatically. League is always
    # computed from rating (app/services/league_engine.py), never stored.
    BATTLE_LEAGUE_SILVER_MIN: int = 1200
    BATTLE_LEAGUE_GOLD_MIN: int = 1500
    BATTLE_LEAGUE_PLATINUM_MIN: int = 1800
    BATTLE_LEAGUE_DIAMOND_MIN: int = 2100

    BATTLE_LEADERBOARD_DEFAULT_PAGE_SIZE: int = 20
    BATTLE_LEADERBOARD_MAX_PAGE_SIZE: int = 100

    # ── Competitive Battle System: matchmaking ─────────────────────────────
    # Redis-backed queues + rating-window matchmaking. Redis holds transient
    # queue/state only; Postgres (BattleMatch/BattleParticipant) is the
    # source of truth once two players are actually paired. See
    # app/services/matchmaking_service.py for the key schema and atomic-claim
    # design.
    REDIS_URL: str = "redis://localhost:6379/0"
    # A stalled Redis command/connect fails after this instead of hanging.
    # Pub/sub's blocking listen() is exempt (redis-py waits indefinitely
    # there regardless), so an idle match channel is never cut off.
    REDIS_SOCKET_TIMEOUT_SECONDS: float = 5.0
    REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS: float = 5.0

    # Rating-window widening by wait time -- a match requires
    # abs(rating_diff) <= min(window(A's wait), window(B's wait)). Tier 4
    # also doubles as the hard cap: get_rating_window() never returns more
    # than this, so there is no separate "max rating difference" setting to
    # keep in sync.
    BATTLE_MATCHMAKING_TIER1_WINDOW_RATING: int = 100
    BATTLE_MATCHMAKING_TIER1_MAX_WAIT_SECONDS: float = 5.0
    BATTLE_MATCHMAKING_TIER2_WINDOW_RATING: int = 150
    BATTLE_MATCHMAKING_TIER2_MAX_WAIT_SECONDS: float = 10.0
    BATTLE_MATCHMAKING_TIER3_WINDOW_RATING: int = 250
    BATTLE_MATCHMAKING_TIER3_MAX_WAIT_SECONDS: float = 20.0
    BATTLE_MATCHMAKING_TIER4_WINDOW_RATING: int = 400

    # How often the background sweep retries waiting users with their
    # newly-widened window.
    BATTLE_MATCHMAKING_SWEEP_INTERVAL_SECONDS: float = 2.0
    # Safety-net TTLs (self-healing, not the primary lifecycle mechanism) --
    # see app/services/matchmaking_service.py's module docstring.
    BATTLE_MATCHMAKING_CLAIM_TTL_SECONDS: int = 20
    BATTLE_MATCHMAKING_QUEUE_ENTRY_TTL_SECONDS: int = 900

    # Online-presence heartbeat TTL -- mark_online() refreshes this on every
    # matchmaking touchpoint (join/cancel/status poll); a client that closes
    # without an explicit mark_offline() call self-heals to "offline" once
    # this expires, same TTL-safety-net philosophy as the state hash above.
    BATTLE_PRESENCE_TTL_SECONDS: int = 60

    # ── Competitive Battle System: battle engine ───────────────────────────
    # Self-paced gameplay: both players get the SAME questions in the SAME
    # order, each moving on with Next/Skip at their own pace, no per-question
    # time limit. The whole match lasts BATTLE_MATCH_DURATION_SECONDS from
    # match.started_at, or ends earlier once both players have finished every
    # question (see battle_gameplay_service._ensure_match_progressed).
    BATTLE_QUESTION_COUNT: int = 10
    BATTLE_MATCH_DURATION_SECONDS: float = 300.0
    # How long after the visible clock hits 0:00 the server still accepts an
    # answer, on top of the player's own network allowance below. 0 = the
    # cutoff is 0:00 as THAT player's phone saw it: an answer sent before
    # their countdown ran out is accepted even though it arrives a round trip
    # later; anything later -- including a save still in progress -- is
    # refused ("Time's up") and the match ends without it.
    BATTLE_ANSWER_LATE_GRACE_SECONDS: float = 0.0
    # Network allowance: this player's fastest recently measured round trip
    # (server ping -> client pong over the match WebSocket), capped here so a
    # client that delays its pongs can never gain more than this. Used twice:
    #   - speed bonus: a question's hidden timer runs from when the server
    #     sent it to when the answer arrived, minus this, so time spent on the
    #     network isn't counted as thinking time;
    #   - the 0:00 cutoff: the phone's countdown runs about this far behind
    #     the server's, so an answer is accepted until 0:00 + this.
    # The match itself is finalized only once the largest possible allowance
    # has passed (see battle_gameplay_service.answers_close_at). 2s covers
    # slow mobile networks; the cost is that a deliberately slow-ponging
    # client could gain at most this much per answer.
    BATTLE_NETWORK_ALLOWANCE_MAX_SECONDS: float = 2.0
    # Round-trip measurement: first ping this long after a socket connects,
    # then every BATTLE_RTT_PING_INTERVAL_SECONDS; the allowance is the
    # fastest of the last BATTLE_RTT_SAMPLE_COUNT samples.
    BATTLE_RTT_FIRST_PING_DELAY_SECONDS: float = 1.0
    BATTLE_RTT_PING_INTERVAL_SECONDS: float = 5.0
    BATTLE_RTT_SAMPLE_COUNT: int = 5
    # grade has no persistence anywhere (not a Question column, prompt-text-
    # only for Groq).
    BATTLE_QUESTION_GRADE_DEFAULT: int = 10
    # Match activation (battle_question_service.py) never calls Groq live --
    # it only ever reads the DB pool, per (subject, difficulty) bucket, so a
    # slow/rate-limited/down Groq can no longer stall or fail match start.
    # A bucket read below BATTLE_POOL_MIN_SIZE kicks off a debounced,
    # fire-and-forget background top-up back up to BATTLE_POOL_TARGET_SIZE
    # (capped at BATTLE_POOL_TOPUP_MAX_BATCH questions per Groq call) --
    # mirrors Personalized-Quiz-Service's own POOL_MIN_SIZE/POOL_TARGET_SIZE/
    # POOL_TOPUP_MAX_BATCH (separate settings here since this is a different
    # service/process/config file, not a shared value).
    BATTLE_POOL_MIN_SIZE: int = 20
    BATTLE_POOL_TARGET_SIZE: int = 60
    BATTLE_POOL_TOPUP_MAX_BATCH: int = 20
    # Top up every allowed subject's thin buckets once at startup, so a
    # subject nobody has battled in yet already has questions (see
    # battle_question_service.warm_up_pools).
    BATTLE_POOL_WARMUP_ON_STARTUP: bool = True
    # Hard cap per (subject, difficulty) for "grow" top-ups -- the extra
    # batches requested once a match had to serve already-seen questions
    # because the pair had exhausted the never-seen pool. Keeps regular
    # players from memorising a fixed ~60-question set, while bounding the
    # Groq spend.
    BATTLE_POOL_MAX_SIZE: int = 300

    # ── AI-first question generation (battle_question_service) ────────────
    # When a match starts, ask the AI for fresh questions first -- checked
    # against everything either player has already been shown in a battle,
    # and against the pool -- and only fill whatever it couldn't provide in
    # time from the pool (never-seen first). Off = pool only, as before.
    BATTLE_AI_FIRST_ENABLED: bool = True
    # How long a match start waits for the AI before falling back to the
    # pool for the rest. Players see the "preparing" screen meanwhile.
    BATTLE_AI_GENERATION_TIMEOUT_SECONDS: float = 8.0
    # Ask for this much more than needed (rounded up, at least 1 per tier),
    # so questions dropped as repeats/near-duplicates still leave enough.
    BATTLE_AI_OVERGENERATE_RATIO: float = 0.4
    # Matches generating at once per process; any match beyond this just
    # uses the pool, so a burst of matches can't run into Groq's rate limit.
    BATTLE_AI_MAX_CONCURRENT_GENERATIONS: int = 4
    # How many of the two players' most recently seen questions are listed
    # in the AI prompt as "don't repeat these" (every seen question is still
    # filtered out afterwards either way).
    BATTLE_AI_AVOID_RECENT_QUESTIONS: int = 60
    # The lessons the two players have seen most, passed to the AI as
    # "prefer other lessons" -- spreads matches across the syllabus.
    BATTLE_AI_AVOID_TOP_LESSONS: int = 5

    # Base correct score -- wrong answer is always 0, never negative.
    BATTLE_SCORE_BASE_CORRECT: int = 100

    # Speed bonus tiers (response_time_seconds upper bound -> points),
    # capped so speed can never outweigh correctness.
    BATTLE_SPEED_BONUS_TIER1_MAX_SECONDS: float = 5.0
    BATTLE_SPEED_BONUS_TIER1_POINTS: int = 30
    BATTLE_SPEED_BONUS_TIER2_MAX_SECONDS: float = 10.0
    BATTLE_SPEED_BONUS_TIER2_POINTS: int = 20
    BATTLE_SPEED_BONUS_TIER3_MAX_SECONDS: float = 15.0
    BATTLE_SPEED_BONUS_TIER3_POINTS: int = 10
    BATTLE_SPEED_BONUS_TIER4_POINTS: int = 0

    # Streak bonus tiers (min consecutive-correct streak -> points), capped
    # at tier4 -- no infinite compounding. A wrong or skipped answer resets
    # the streak (see battle_gameplay_service._streak_before). Worked out as
    # each answer is saved but, like every score, only shown at the end.
    BATTLE_STREAK_BONUS_TIER1_MIN_STREAK: int = 2
    BATTLE_STREAK_BONUS_TIER1_POINTS: int = 5
    BATTLE_STREAK_BONUS_TIER2_MIN_STREAK: int = 3
    BATTLE_STREAK_BONUS_TIER2_POINTS: int = 10
    BATTLE_STREAK_BONUS_TIER3_MIN_STREAK: int = 4
    BATTLE_STREAK_BONUS_TIER3_POINTS: int = 15
    BATTLE_STREAK_BONUS_TIER4_MIN_STREAK: int = 5
    BATTLE_STREAK_BONUS_TIER4_POINTS: int = 20

    # ── Competitive Battle System: analytics (GET /battle/analytics) ───────
    # Rating-history points returned per subject (most recent N matches).
    BATTLE_ANALYTICS_RATING_HISTORY_LIMIT: int = 20
    # Results shown as the "recent form" strip (most recent N matches).
    BATTLE_ANALYTICS_RECENT_FORM_LIMIT: int = 10
    # A lesson needs at least this many battle answers before it's reported
    # as a strong/weak topic -- one or two answers say nothing.
    BATTLE_ANALYTICS_TOPIC_MIN_ANSWERS: int = 3

    # ── Competitive Battle System: realtime communication ──────────────────
    # WebSocket live layer over the REST engine -- REST endpoints keep
    # working standalone (POST .../ready replaces the removed
    # auto-activate-on-poll trigger). See app/api/routes/battle_ws.py and
    # app/services/battle_connection_manager.py.
    BATTLE_COUNTDOWN_DURATION_SECONDS: float = 5.0
    # Activation (mark_ready -> _activate_countdown) waits for a participant
    # to be BOTH ready AND have a live WS connection confirmed (see
    # mark_connected) before starting that match's question-timing clock --
    # otherwise a socket that's still mid-reconnect right at match start (see
    # BATTLE_DISCONNECT_NOTICE_DEBOUNCE_SECONDS's own note on queue.tsx's
    # socket closing and match-session.tsx's reopening fresh around here)
    # would silently eat into question 1's answerable window before the
    # client was even present to render it. This bounds how long to wait for
    # that confirmation once both sides ARE ready before just proceeding
    # anyway -- the fallback for a REST-only client (mark_connected never
    # fires for one that never opens a socket at all, by design still
    # supported -- see app/api/routes/battle_ws.py's module docstring) and
    # for the rare real connection that's delayed past this. Enforced by
    # run_stale_match_reap_once, not a live timer -- see its docstring.
    BATTLE_CONNECTION_READY_TIMEOUT_SECONDS: float = 8.0
    # Safety-net TTL for an abandoned ready hash (e.g. one side readies then
    # vanishes and the match is never cancelled/completed) -- not the
    # primary lifecycle mechanism.
    BATTLE_READY_TTL_SECONDS: int = 300
    # Must comfortably outlast the mobile client's own reconnect backoff
    # cycle (currently 1+2+4+8=15s cumulative before its last first-cycle
    # attempt even fires -- see RECONNECT_DELAYS_MS in
    # Mobile-Application/hooks/use-battle-match.tsx) plus handshake/DNS/Redis
    # latency on top. At 15.0 an ordinary drop that needed the full backoff
    # cycle to recover was racing this deadline with ~zero margin.
    BATTLE_DISCONNECT_GRACE_PERIOD_SECONDS: float = 30.0
    # Extra Redis-key TTL margin beyond the grace period itself, so the key
    # outlives the deadline it encodes long enough for a late grace-check
    # task to still read it.
    BATTLE_DISCONNECT_MARKER_BUFFER_SECONDS: float = 10.0
    # A client's socket for a match legitimately closes and reopens as a
    # brand-new connection during normal navigation -- e.g. queue.tsx's
    # socket closes and match-session.tsx's opens fresh a beat later once
    # both players ready up, well before the reopened one has a chance to
    # bump the connection generation. Without this delay, _notify_disconnect
    # (battle_ws.py) would win that race almost every time and publish a
    # spurious player_disconnected/player_reconnected pair to the OPPONENT
    # for a socket swap that was never a real drop. Waiting this long before
    # even checking the generation gives a same-device reconnect time to
    # land and supersede it first.
    BATTLE_DISCONNECT_NOTICE_DEBOUNCE_SECONDS: float = 3.0
    # How often a connected client's progression timer polls while a match
    # is still "waiting" for both sides to ready up.
    BATTLE_WS_WAITING_POLL_INTERVAL_SECONDS: float = 1.0
    # While "active", the progression timer re-checks the match at least this
    # often instead of sleeping straight through to the end of the match
    # clock -- one cheap read, and it notices a match that ended some other
    # way (both players finished, a forfeit) even if its stop signal is missed.
    BATTLE_WS_ACTIVE_POLL_INTERVAL_SECONDS: float = 5.0
    # Longest a single Next/Skip save may take over the WebSocket before it's
    # abandoned and reported as "Couldn't save your answer — retrying." --
    # the client resends it, and the save is idempotent (compare-and-swap on
    # current_question_index), so a stalled DB fails instead of hanging the
    # player's connection until the network stack gives up. Covers the save
    # up to its commit only. Sized for ~10 sequential queries to a distant DB
    # (each a full network round trip) -- keep it above the slowest normal save.
    BATTLE_ANSWER_SAVE_TIMEOUT_SECONDS: float = 20.0
    # Diagnostic: a Next/Skip save slower than this (or one that fails/times
    # out) logs how long each of its steps took -- shows which DB round trip
    # stalled. See battle_gameplay_service._SaveSteps.
    BATTLE_SLOW_SAVE_LOG_SECONDS: float = 3.0
    # Safety-net TTL for the per-(match,user) connection-generation counter
    # (see battle_gameplay_service.bump_connection_generation) -- generous,
    # well past any realistic match duration, not the primary cleanup path.
    BATTLE_CONNECTION_GENERATION_TTL_SECONDS: int = 3600
    # Delay before battle_connection_manager.py's supervisor restarts a
    # match's progression timer after it dies from an unhandled exception
    # (a transient DB/Redis error mid-tick, say) -- without this, one such
    # error would silently freeze that match's question progression forever
    # for anyone routed to this process, since the timer is only ever
    # started once, on the first local connection.
    BATTLE_PROGRESSION_TIMER_RESPAWN_DELAY_SECONDS: float = 2.0

    # ── Competitive Battle System: integration and hardening ───────────────
    # How long a match may sit at status="waiting" before the reaper treats
    # it as orphaned (the crash window between matchmaking's Postgres commit
    # and its Redis finalize step -- see matchmaking_service.py's module
    # docstring). Still well past BATTLE_MATCHMAKING_CLAIM_TTL_SECONDS (20s
    # default) so a merely-slow client is never mistaken for a crash -- 60s
    # is 3x that, plenty of margin, while keeping the "wrapping up" wait a
    # genuinely abandoned player sees short rather than a multi-minute one.
    BATTLE_ORPHANED_MATCH_REAP_AFTER_SECONDS: float = 60.0
    # How often the reaper loop itself runs -- lightweight queries (two
    # indexed status/timestamp scans), so checking frequently costs little
    # and directly bounds how long a match that just crossed a reap cutoff
    # above sits there unreaped before anyone waiting on it notices.
    BATTLE_STALE_MATCH_REAPER_INTERVAL_SECONDS: float = 15.0
    # How long a match may sit at status="countdown"/"active" with no client
    # ever revisiting it before the same reaper treats it as abandoned (both
    # players gone -- e.g. closed the app right after the match started).
    # _ensure_match_progressed only ever runs when triggered by a REST call
    # or an open WS connection, so a match nobody is connected to just sits
    # forever otherwise -- and its Redis "in_match" pointer (no TTL) with it,
    # locking the abandoning player out of queueing again until this reaper
    # runs. Set well past any real match's total duration
    # (BATTLE_MATCH_DURATION_SECONDS, 300s by default) rather than a
    # much larger arbitrary buffer: a match genuinely still being played
    # always transitions out of "active" via _ensure_match_progressed the
    # moment that budget elapses and anyone touches it, so this can never
    # reap a match someone is actually still playing -- it only shortens how
    # long an abandoned one blocks its players.
    BATTLE_ACTIVE_MATCH_REAP_AFTER_SECONDS: float = 600.0
    # An "active" match nobody is connected to anymore once its whole time
    # budget (BATTLE_MATCH_DURATION_SECONDS + BATTLE_ANSWER_LATE_GRACE_SECONDS)
    # plus this buffer has passed gets settled by the reaper: result and
    # rating changes applied as if someone had been there
    # to see it end -- so a winner whose opponent (and then they themselves)
    # closed the app still gets the win. Only a match where neither player
    # ever answered anything is cancelled as abandoned instead.
    BATTLE_FINISHED_MATCH_SETTLE_BUFFER_SECONDS: float = 30.0
    # How often to cross-check every "in_match" Redis pointer against its
    # match's actual Postgres status (matchmaking_service.py's
    # run_in_match_reconcile_once) -- self-heals a user permanently stuck
    # unable to queue if a match's post-commit release_match_state() call
    # ever failed (transient Redis error, etc.), since "in_match" has no TTL
    # by design and nothing else ever revisits an already-terminal match.
    # This is also the MOST common real trigger behind the mobile app's
    # "wrapping up" UI (OnlineList's FAB), since it's the exact case of
    # "Postgres already knows this match is over, Redis just hasn't caught
    # up yet" -- kept short (20s) so that wait stays brief rather than the
    # 2 minutes this used to allow.
    BATTLE_IN_MATCH_RECONCILE_INTERVAL_SECONDS: float = 20.0
    # Fixed-window (INCR+EXPIRE) rate limits -- see app/services/rate_limiter.py.
    BATTLE_QUEUE_JOIN_RATE_LIMIT_MAX_CALLS: int = 5
    BATTLE_QUEUE_JOIN_RATE_LIMIT_WINDOW_SECONDS: int = 10
    BATTLE_WS_CONNECT_RATE_LIMIT_MAX_CALLS: int = 10
    BATTLE_WS_CONNECT_RATE_LIMIT_WINDOW_SECONDS: int = 60
    # Per-socket inbound message cap (in-process sliding window). Each answer
    # message costs several DB queries; a real client sends a handful per
    # question, so anything above this is a buggy or abusive client.
    BATTLE_WS_MESSAGE_RATE_LIMIT_MAX_CALLS: int = 10
    BATTLE_WS_MESSAGE_RATE_LIMIT_WINDOW_SECONDS: float = 2.0
    # Comma-separated subjects players may queue for (exact names). Every queued subject gets its own rating row,
    # Redis queue and AI-generated questions written to the shared question pool, and the name is part of the AI
    # prompt -- so free text here would let a player steer what Groq writes into that pool. Defaults to the mobile
    # app's subject picker (constants/quizStyles.ts SUBJECT_ICONS); keep the two in sync. Empty = any subject.
    BATTLE_ALLOWED_SUBJECTS: str = (
        "Mathematics,Science,History,English,Geography,ICT,Health and Physical Education"
    )

    @property
    def battle_allowed_subjects(self) -> set[str]:
        return {s.strip() for s in self.BATTLE_ALLOWED_SUBJECTS.split(",") if s.strip()}

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


# Required fields (DATABASE_URL, GROQ_API_KEY, CLERK_JWKS_URL, CLERK_ISSUER)
# come from the environment / .env at runtime via pydantic-settings, not
# constructor arguments -- static analysis can't see that, hence the ignore.
settings = Settings()  # type: ignore[call-arg]
