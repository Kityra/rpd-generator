"""Add course-scoped adjacent profession boundaries."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB


revision = "0006_profession_boundaries"
down_revision = "0005_judge_subject_id"
branch_labels = None
depends_on = None


def upgrade() -> None:
    json_type = sa.JSON().with_variant(JSONB(), "postgresql")
    op.create_table(
        "profession_boundaries",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("course_id", sa.String(36), sa.ForeignKey("courses.id"), nullable=False),
        sa.Column("primary_profession", sa.String(300), nullable=False),
        sa.Column("adjacent_profession", sa.String(300), nullable=False),
        sa.Column("skill_jaccard_overlap", sa.Float(), nullable=False),
        *(sa.Column(name, json_type, nullable=False, server_default="[]") for name in (
            "overlap_tasks_json", "exclusive_to_primary_json", "exclusive_to_adjacent_json",
            "shared_core_json", "handoff_artifacts_json", "boundary_conflicts_json", "rejected_noise_turns_json",
            "primary_claims_json",
        )),
        sa.Column("invite_token_hash", sa.String(64), nullable=True, unique=True),
        sa.Column("invite_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_interview_id", sa.String(36), nullable=True),
        sa.Column("turn_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("interview_completed", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("verified_by_methodologist", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("course_id", "primary_profession", "adjacent_profession"),
    )
    for key in ("course_id", "primary_profession", "adjacent_profession"):
        op.create_index(f"ix_profession_boundaries_{key}", "profession_boundaries", [key])


def downgrade() -> None:
    op.drop_table("profession_boundaries")
