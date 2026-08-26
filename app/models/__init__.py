# Every model needs to be imported here, or SQLAlchemy won't know it exists
# when create_all()/alembic autogenerate runs -- add new models to this list too.
from app.models.ai_generation_event import AIGenerationEvent
from app.models.battle_match import (
	BattleAnswer,
	BattleMatch,
	BattleMatchQuestion,
	BattleParticipant,
)
from app.models.competitive_rating import CompetitiveSubjectRating
from app.models.question import Question
from app.models.user import User

__all__ = [
	"AIGenerationEvent",
	"BattleAnswer",
	"BattleMatch",
	"BattleMatchQuestion",
	"BattleParticipant",
	"CompetitiveSubjectRating",
	"Question",
	"User",
]
