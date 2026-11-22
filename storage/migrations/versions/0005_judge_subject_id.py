"""Store the subject ID for individual rationale judge calls."""

import sqlalchemy as sa
from alembic import op


revision = "0005_judge_subject_id"
down_revision = "0004_activity_model"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = sa.inspect(op.get_bind()).get_columns("model_calls")
    if not any(column["name"] == "subject_id" for column in columns):
        op.add_column("model_calls", sa.Column("subject_id", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("model_calls") as batch:
        batch.drop_column("subject_id")
