"""add battle participant finished_at

Revision ID: d529bd73ee22
Revises: 37f1511a38ce
Create Date: 2026-08-25 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd529bd73ee22'
down_revision: Union[str, Sequence[str], None] = '37f1511a38ce'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('battle_participants', sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('battle_participants', 'finished_at')
