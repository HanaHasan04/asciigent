"""Free-tier failure handling.

A 20-request daily allowance makes retry policy a correctness issue, not a
robustness nicety: blind retries burn the day in seconds.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asciigent.providers import (  # noqa: E402
    DailyQuotaExhausted,
    _is_per_day,
    _retry_delay,
)

PER_DAY = (
    "Error code: 429 - Quota exceeded for metric: generate_content_free_tier_"
    "requests, limit: 20, model: gemini-3.8-flash. Please retry in 40.0198s. "
    "'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'"
)
PER_MINUTE = (
    "Error code: 429 - Quota exceeded. 'quotaId': "
    "'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', "
    "'retryDelay': '31s'"
)


def test_per_day_quota_is_distinguished_from_per_minute():
    assert _is_per_day(PER_DAY)
    assert not _is_per_day(PER_MINUTE)


def test_retry_delay_is_read_from_either_form():
    assert _retry_delay(PER_DAY) == 40.0198
    assert _retry_delay(PER_MINUTE) == 31.0
    assert _retry_delay("no delay mentioned") is None


def test_daily_quota_is_its_own_exception_type():
    """Callers must be able to stop a sweep on it, not treat it as transient."""
    assert issubclass(DailyQuotaExhausted, RuntimeError)


def test_sdk_level_retries_are_disabled_for_openai_compat_providers():
    """The SDK retries 429s blindly; on 20/day that wastes the allowance."""
    import inspect

    from asciigent.providers import OpenAICompatProvider

    src = inspect.getsource(OpenAICompatProvider.__init__)
    assert "max_retries=0" in src
