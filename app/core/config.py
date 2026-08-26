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
    # Lockstep gameplay: both players see the SAME question at the SAME
    # time, derived purely from elapsed wall-clock time since
    # match.started_at (see app/services/battle_gameplay_service.py's
    # _current_question_order) -- BATTLE_QUESTION_TIME_LIMIT_SECONDS is each
    # question's fixed window; the match completes once
    # BATTLE_QUESTION_COUNT * BATTLE_QUESTION_TIME_LIMIT_SECONDS total
    # elapses, regardless of how many questions either player actually
    # answered (an unanswered question when its window closes just counts
    # as unanswered -- see _ensure_match_progressed).
    BATTLE_QUESTION_COUNT: int = 10
    BATTLE_QUESTION_TIME_LIMIT_SECONDS: float = 30.0
    # grade has no persistence anywhere (not a Question column, prompt-text-
    # only for Groq).
    BATTLE_QUESTION_GRADE_DEFAULT: int = 10
    BATTLE_QUESTION_GROQ_MAX_CALLS: int = 3

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
    # at tier4 -- no infinite compounding. Wrong/skipped answer resets the
    # streak (see battle_gameplay_service._streak_before).
    BATTLE_STREAK_BONUS_TIER1_MIN_STREAK: int = 2
    BATTLE_STREAK_BONUS_TIER1_POINTS: int = 5
    BATTLE_STREAK_BONUS_TIER2_MIN_STREAK: int = 3
    BATTLE_STREAK_BONUS_TIER2_POINTS: int = 10
    BATTLE_STREAK_BONUS_TIER3_MIN_STREAK: int = 4
    BATTLE_STREAK_BONUS_TIER3_POINTS: int = 15
    BATTLE_STREAK_BONUS_TIER4_MIN_STREAK: int = 5
    BATTLE_STREAK_BONUS_TIER4_POINTS: int = 20

    # ── Competitive Battle System: realtime communication ──────────────────
    # WebSocket live layer over the REST engine -- REST endpoints keep
    # working standalone (POST .../ready replaces the removed
    # auto-activate-on-poll trigger). See app/api/routes/battle_ws.py and
    # app/services/battle_connection_manager.py.
    BATTLE_COUNTDOWN_DURATION_SECONDS: float = 5.0
    # Safety-net TTL for an abandoned ready hash (e.g. one side readies then
    # vanishes and the match is never cancelled/completed) -- not the
    # primary lifecycle mechanism.
    BATTLE_READY_TTL_SECONDS: int = 300
    BATTLE_DISCONNECT_GRACE_PERIOD_SECONDS: float = 15.0
    # Extra Redis-key TTL margin beyond the grace period itself, so the key
    # outlives the deadline it encodes long enough for a late grace-check
    # task to still read it.
    BATTLE_DISCONNECT_MARKER_BUFFER_SECONDS: float = 10.0
    # How often a connected client's progression timer polls while a match
    # is still "waiting" for both sides to ready up.
    BATTLE_WS_WAITING_POLL_INTERVAL_SECONDS: float = 1.0
    # Safety-net TTL for the per-(match,user) connection-generation counter
    # (see battle_gameplay_service.bump_connection_generation) -- generous,
    # well past any realistic match duration, not the primary cleanup path.
    BATTLE_CONNECTION_GENERATION_TTL_SECONDS: int = 3600

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
    # runs. Set to 2x any real match's total duration (BATTLE_QUESTION_COUNT
    # * BATTLE_QUESTION_TIME_LIMIT_SECONDS is 300s by default) rather than a
    # much larger arbitrary buffer: a match genuinely still being played
    # always transitions out of "active" via _ensure_match_progressed the
    # moment that budget elapses and anyone touches it, so this can never
    # reap a match someone is actually still playing -- it only shortens how
    # long an abandoned one blocks its players.
    BATTLE_ACTIVE_MATCH_REAP_AFTER_SECONDS: float = 600.0
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
