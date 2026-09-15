"""empty message

Revision ID: 062a656ad412
Revises: 6f353d928e1f, a1b2c3d4e5f6
Create Date: 2026-09-15 15:16:18.340602

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '062a656ad412'
down_revision: Union[str, None] = ('6f353d928e1f', 'a1b2c3d4e5f6')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
