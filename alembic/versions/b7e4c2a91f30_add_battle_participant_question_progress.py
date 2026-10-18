"""add battle participant question progress

Revision ID: b7e4c2a91f30
Revises: d529bd73ee22
Create Date: 2026-10-18 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b7e4c2a91f30'
down_revision: Union[str, Sequence[str], None] = 'd529bd73ee22'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'battle_participants',
        sa.Column('current_question_index', sa.Integer(), nullable=False, server_default='0'),
    )
    op.add_column(
        'battle_participants',
        sa.Column('current_question_presented_at', sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('battle_participants', 'current_question_presented_at')
    op.drop_column('battle_participants', 'current_question_index')
