import asyncio
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, WebSocket, WebSocketException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient, PyJWKClientError

from app.core.config import settings

_bearer_scheme = HTTPBearer(auto_error=False)
_jwks_client = PyJWKClient(settings.CLERK_JWKS_URL, cache_keys=True)


async def _verify_token(token: str) -> dict:
    # The actual Clerk-JWT-verification core -- only needs a raw token
    # string, so it's shared between the HTTP path (get_current_user, whose
    # token comes from an Authorization header via HTTPBearer) and the
    # WebSocket path (get_current_user_ws, whose token comes from a query
    # param instead, since native WS clients generally can't set custom
    # handshake headers). Raises the same HTTPExceptions either way; callers
    # translate them into whatever their transport needs (get_current_user_ws
    # turns them into a WebSocketException).
    try:
        # PyJWKClient fetches the JWKS with blocking urllib on a cache miss / key rotation; off the event loop
        # so one slow Clerk round-trip doesn't stall every live battle socket on this process.
        signing_key = await asyncio.to_thread(_jwks_client.get_signing_key_from_jwt, token)

        payload: dict = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            issuer=settings.CLERK_ISSUER,
            audience=settings.CLERK_AUDIENCE or None,
            options={
                "verify_exp": True,
                "verify_iat": True,
            },
        )

    except PyJWKClientError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not fetch Clerk public keys: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please sign in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Clerk session tokens carry `azp` (the origin that requested them). When CLERK_AUTHORIZED_PARTIES is set,
    # only tokens minted for those origins are accepted; native-app tokens may have no azp, which is allowed.
    allowed = {p.strip() for p in settings.CLERK_AUTHORIZED_PARTIES.split(",") if p.strip()}
    azp = payload.get("azp")
    if allowed and azp and azp not in allowed:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: unexpected authorized party.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload


def _dev_bypass_identity() -> dict:
    return {
        "sub": "dev-user",
        "email": "dev-user@example.com",
        "name": "Development User",
        "username": "dev-user",
        "dev_auth_bypass": True,
    }


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict:
    # Verifies the Clerk JWT against Clerk's public keys and returns the
    # decoded payload (at least `sub`, the Clerk user ID). In dev, if
    # AUTH_BYPASS is on, this skips verification entirely and hands back a
    # fake "dev-user" identity — lets you hit any endpoint locally without a
    # real Clerk session.
    if settings.ENVIRONMENT == "development" and settings.AUTH_BYPASS:
        return _dev_bypass_identity()

    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing or malformed. Provide: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return await _verify_token(credentials.credentials)


async def get_clerk_id(current_user: dict = Depends(get_current_user)) -> str:
    # The Clerk user ID (`sub`) every user-scoped route keys on.
    clerk_id: str = current_user.get("sub", "")
    if not clerk_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing the 'sub' (user ID) claim.",
        )
    return clerk_id


ClerkId = Annotated[str, Depends(get_clerk_id)]


async def get_current_user_ws(websocket: WebSocket) -> dict:
    # WebSocket counterpart of get_current_user. Native WS clients generally
    # can't set a custom Authorization header on the handshake, so the token
    # travels as a query param instead (?token=<jwt>). Must raise
    # WebSocketException, not HTTPException -- a WS route can't return a
    # normal HTTP error response once the handshake is underway; raising
    # this BEFORE the route body calls `await websocket.accept()` correctly
    # rejects the connection instead.
    if settings.ENVIRONMENT == "development" and settings.AUTH_BYPASS:
        return _dev_bypass_identity()

    token = websocket.query_params.get("token")
    if not token:
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION, reason="Missing 'token' query parameter."
        )

    try:
        return await _verify_token(token)
    except HTTPException as exc:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason=str(exc.detail))


async def require_admin_or_dev(current_user: dict = Depends(get_current_user)) -> dict:
    # Gate for internal/admin-only endpoints. There's no real role system
    # here, so you get in if you're either running in dev, or your Clerk ID
    # is in ADMIN_CLERK_IDS. Still runs get_current_user first, so this only
    # adds a stricter check on top — it never skips authentication.
    if settings.ENVIRONMENT == "development":
        return current_user

    admin_ids = {cid.strip() for cid in settings.ADMIN_CLERK_IDS.split(",") if cid.strip()}
    if current_user.get("sub") in admin_ids:
        return current_user

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="This endpoint is restricted to administrators.",
    )
