# Copied from Personalized-Quiz-Service's quiz_service.py (was _CallBudget
# there). Keep in sync by hand if the upstream logic changes.
from dataclasses import dataclass


@dataclass
class CallBudget:
    # Shared, mutable cap on Groq calls across one question-selection
    # request's dedup-retry loop -- without this, a single request could
    # fire off far more Groq calls than it should ever cost.
    remaining: int

    def consume(self) -> bool:
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        return True
