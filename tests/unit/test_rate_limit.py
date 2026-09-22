import pytest


def test_token_bucket_rejects_requests_after_capacity_is_spent() -> None:
    from mini_stack.provider.rate_limit import TokenBucket

    bucket = TokenBucket(capacity=2, refill_per_second=0)

    assert bucket.allow(now=0.0)
    assert bucket.allow(now=0.0)
    assert not bucket.allow(now=0.0)


def test_token_bucket_refills_deterministically() -> None:
    from mini_stack.provider.rate_limit import TokenBucket

    bucket = TokenBucket(capacity=1, refill_per_second=2)
    assert bucket.allow(now=0.0)
    assert not bucket.allow(now=0.0)
    assert bucket.allow(now=0.5)
    with pytest.raises(ValueError, match="capacity"):
        TokenBucket(capacity=0, refill_per_second=1)
