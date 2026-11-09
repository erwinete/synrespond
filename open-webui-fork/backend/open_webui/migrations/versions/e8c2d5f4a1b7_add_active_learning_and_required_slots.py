"""Active Learning: intent_feedback и обязательные слоты правил маршрутизации.

revision: e8c2d5f4a1b7
down_revision: e7a1f4c2b9d8
"""

import sqlalchemy as sa
from alembic import op

revision = 'e8c2d5f4a1b7'
down_revision = 'e7a1f4c2b9d8'
branch_labels = None
depends_on = None


def _table_exists(conn, name: str) -> bool:
    inspector = sa.inspect(conn)
    return name in inspector.get_table_names()


def _column_exists(conn, table: str, column: str) -> bool:
    inspector = sa.inspect(conn)
    return column in [col['name'] for col in inspector.get_columns(table)]


def upgrade() -> None:
    conn = op.get_bind()

    if not _column_exists(conn, 'intent_route', 'required_slots'):
        op.add_column('intent_route', sa.Column('required_slots', sa.JSON(), nullable=True))

    if not _table_exists(conn, 'intent_feedback'):
        op.create_table(
            'intent_feedback',
            sa.Column('id', sa.String(), primary_key=True, unique=True),
            sa.Column('bot_model_id', sa.String(), nullable=True, index=True),
            sa.Column('text', sa.Text(), nullable=False),
            sa.Column('predicted_intent', sa.Text(), nullable=True),
            sa.Column('confidence', sa.Float(), nullable=True),
            sa.Column('top_candidates', sa.JSON(), nullable=True),
            sa.Column('hint_intent', sa.Text(), nullable=True),
            sa.Column('hint_confidence', sa.Float(), nullable=True),
            sa.Column('labeled_intent', sa.Text(), nullable=True),
            sa.Column('status', sa.String(), nullable=False, server_default='pending'),
            sa.Column('source', sa.String(), nullable=False, server_default='gray_zone'),
            sa.Column('meta', sa.JSON(), nullable=True),
            sa.Column('created_at', sa.BigInteger(), nullable=False),
            sa.Column('updated_at', sa.BigInteger(), nullable=False),
        )
        op.create_index(
            'ix_intent_feedback_status_created', 'intent_feedback', ['status', 'created_at']
        )


def downgrade() -> None:
    op.drop_table('intent_feedback')
    op.drop_column('intent_route', 'required_slots')
