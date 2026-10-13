# Copied from Personalized-Quiz-Service's quiz_service.py (was
# get_or_create_user there). Keep in sync by hand if the upstream logic
# changes -- writes to the SAME `users` table Personalized-Quiz-Service
# owns, via this service's thin User mirror model.
import logging

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User

logger = logging.getLogger(__name__)


async def get_or_create_user(db: AsyncSession, clerk_id: str) -> User:
    stmt = select(User).where(User.clerk_id == clerk_id)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if user is not None:
        return user

    logger.info("First login — creating user record for clerk_id=%s", clerk_id)
    try:
        # Savepoint: a concurrent first request (REST call + WS connect, or the
        # quiz service) can insert the same clerk_id between our SELECT and
        # INSERT. Only this insert rolls back; the caller's transaction survives.
        async with db.begin_nested():
            user = User(clerk_id=clerk_id)
            db.add(user)
    except IntegrityError:
        logger.info("User for clerk_id=%s was created concurrently — reusing it", clerk_id)
        return (await db.execute(stmt)).scalar_one()

    await db.commit()
    await db.refresh(user)
    return user
