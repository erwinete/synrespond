"""Энкодеры-классификаторы, правила маршрутизации интентов и таблицы Ragas.

revision: e7a1f4c2b9d8
down_revision: e42c8f710a36
"""

import sqlalchemy as sa
from alembic import op

revision = 'e7a1f4c2b9d8'
down_revision = 'e42c8f710a36'
branch_labels = None
depends_on = None


def _table_exists(conn, name: str) -> bool:
    inspector = sa.inspect(conn)
    return name in inspector.get_table_names()


def upgrade() -> None:
    conn = op.get_bind()

    if not _table_exists(conn, 'encoder_model'):
        op.create_table(
            'encoder_model',
            sa.Column('id', sa.String(), primary_key=True, unique=True),
            sa.Column('name', sa.Text(), nullable=False),
            sa.Column('task', sa.Text(), nullable=False, server_default='sequence_classification'),
            sa.Column('endpoint_url', sa.Text(), nullable=True),
            sa.Column('model_path', sa.Text(), nullable=True),
            sa.Column('labels_json', sa.JSON(), nullable=True),
            sa.Column('thresholds', sa.JSON(), nullable=True),
            sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.text('1')),
            sa.Column('meta', sa.JSON(), nullable=True),
            sa.Column('user_id', sa.String(), nullable=True),
            sa.Column('created_at', sa.BigInteger(), nullable=False),
            sa.Column('updated_at', sa.BigInteger(), nullable=False),
        )
        op.create_index('ix_encoder_model_id_active', 'encoder_model', ['id', 'is_active'])

    if not _table_exists(conn, 'intent_route'):
        op.create_table(
            'intent_route',
            sa.Column('id', sa.String(), primary_key=True, unique=True),
            sa.Column('bot_model_id', sa.String(), nullable=False, index=True),
            sa.Column('intent_name', sa.Text(), nullable=False),
            sa.Column('examples', sa.JSON(), nullable=True),
            sa.Column('min_confidence', sa.Float(), nullable=False, server_default=sa.text('0.80')),
            sa.Column('tau_low', sa.Float(), nullable=False, server_default=sa.text('0.50')),
            sa.Column('action_type', sa.Text(), nullable=False, server_default='RAG_COLLECTION'),
            sa.Column('target_collection_ids', sa.JSON(), nullable=True),
            sa.Column('use_hipporag', sa.Boolean(), nullable=False, server_default=sa.text('0')),
            sa.Column('hybrid_weights', sa.JSON(), nullable=True),
            sa.Column('system_prompt_template', sa.Text(), nullable=True),
            sa.Column('direct_response', sa.Text(), nullable=True),
            sa.Column('webhook_config', sa.JSON(), nullable=True),
            sa.Column('priority', sa.BigInteger(), nullable=False, server_default=sa.text('0')),
            sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.text('1')),
            sa.Column('meta', sa.JSON(), nullable=True),
            sa.Column('created_at', sa.BigInteger(), nullable=False),
            sa.Column('updated_at', sa.BigInteger(), nullable=False),
        )
        op.create_index('ix_intent_route_model_intent', 'intent_route', ['bot_model_id', 'intent_name'])

    if not _table_exists(conn, 'ragas_dataset'):
        op.create_table(
            'ragas_dataset',
            sa.Column('id', sa.String(), primary_key=True, unique=True),
            sa.Column('name', sa.Text(), nullable=False),
            sa.Column('description', sa.Text(), nullable=True),
            sa.Column('items', sa.JSON(), nullable=False),
            sa.Column('meta', sa.JSON(), nullable=True),
            sa.Column('user_id', sa.String(), nullable=True),
            sa.Column('created_at', sa.BigInteger(), nullable=False),
            sa.Column('updated_at', sa.BigInteger(), nullable=False),
        )

    if not _table_exists(conn, 'ragas_run'):
        op.create_table(
            'ragas_run',
            sa.Column('id', sa.String(), primary_key=True, unique=True),
            sa.Column('name', sa.Text(), nullable=True),
            sa.Column('mode', sa.String(), nullable=False, server_default='batch'),
            sa.Column('dataset_id', sa.String(), nullable=True),
            sa.Column('bot_model_id', sa.String(), nullable=True),
            sa.Column('status', sa.String(), nullable=False, server_default='pending'),
            sa.Column('metrics', sa.JSON(), nullable=True),
            sa.Column('config', sa.JSON(), nullable=True),
            sa.Column('sample_count', sa.BigInteger(), nullable=False, server_default=sa.text('0')),
            sa.Column('error', sa.Text(), nullable=True),
            sa.Column('created_at', sa.BigInteger(), nullable=False),
            sa.Column('completed_at', sa.BigInteger(), nullable=True),
        )
        op.create_index('ix_ragas_run_status_created', 'ragas_run', ['status', 'created_at'])

    if not _table_exists(conn, 'ragas_sample'):
        op.create_table(
            'ragas_sample',
            sa.Column('id', sa.String(), primary_key=True, unique=True),
            sa.Column('run_id', sa.String(), nullable=False, index=True),
            sa.Column('question', sa.Text(), nullable=False),
            sa.Column('rewritten_query', sa.Text(), nullable=True),
            sa.Column('answer', sa.Text(), nullable=True),
            sa.Column('contexts', sa.JSON(), nullable=True),
            sa.Column('ground_truth', sa.Text(), nullable=True),
            sa.Column('scores', sa.JSON(), nullable=True),
            sa.Column('claims', sa.JSON(), nullable=True),
            sa.Column('retrieval_trace', sa.JSON(), nullable=True),
            sa.Column('flag', sa.Text(), nullable=True),
            sa.Column('bot_model_id', sa.String(), nullable=True),
            sa.Column('chat_id', sa.String(), nullable=True),
            sa.Column('message_id', sa.String(), nullable=True),
            sa.Column('created_at', sa.BigInteger(), nullable=False),
        )
        op.create_index('ix_ragas_sample_run_created', 'ragas_sample', ['run_id', 'created_at'])


def downgrade() -> None:
    op.drop_table('ragas_sample')
    op.drop_table('ragas_run')
    op.drop_table('ragas_dataset')
    op.drop_table('intent_route')
    op.drop_table('encoder_model')
