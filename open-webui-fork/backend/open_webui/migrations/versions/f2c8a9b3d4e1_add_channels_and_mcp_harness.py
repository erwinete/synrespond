"""Stage 4: allowed_channels для правил интентов и реестр MCP-харнесов.

revision: f2c8a9b3d4e1
down_revision: e8c2d5f4a1b7
"""

import sqlalchemy as sa
from alembic import op

revision = 'f2c8a9b3d4e1'
down_revision = 'e8c2d5f4a1b7'
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

    if not _column_exists(conn, 'intent_route', 'allowed_channels'):
        op.add_column('intent_route', sa.Column('allowed_channels', sa.JSON(), nullable=True))

    if not _column_exists(conn, 'intent_route', 'mcp_harness_config'):
        op.add_column('intent_route', sa.Column('mcp_harness_config', sa.JSON(), nullable=True))

    if not _table_exists(conn, 'mcp_harness'):
        op.create_table(
            'mcp_harness',
            sa.Column('id', sa.String(), primary_key=True, unique=True),
            sa.Column('name', sa.Text(), nullable=False),
            sa.Column('description_for_agents', sa.Text(), nullable=True),
            sa.Column('transport', sa.String(), nullable=False, server_default='streamable_http'),
            sa.Column('endpoint_or_cmd', sa.Text(), nullable=False),
            sa.Column('auth_token', sa.Text(), nullable=True),
            sa.Column('discovered_tools_schema', sa.JSON(), nullable=True),
            sa.Column('timeout_ms', sa.BigInteger(), nullable=False, server_default=sa.text('5000')),
            sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.text('1')),
            sa.Column('meta', sa.JSON(), nullable=True),
            sa.Column('created_at', sa.BigInteger(), nullable=False),
            sa.Column('updated_at', sa.BigInteger(), nullable=False),
        )


def downgrade() -> None:
    op.drop_table('mcp_harness')
    op.drop_column('intent_route', 'mcp_harness_config')
    op.drop_column('intent_route', 'allowed_channels')
