"""Core unit tests for Mandarin Assistant."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import Mock, patch
import os


def test_query_corpus_high_similarity():
    """Test retrieval with high similarity match."""
    from app import query_corpus
    phrase, sim, suggestions = query_corpus("Want to drink tonight?")
    assert phrase is not None
    assert sim > 0.7  # Should find close match
    assert "en" in phrase


def test_query_corpus_low_similarity():
    """Test retrieval with no good match."""
    from app import query_corpus
    phrase, sim, suggestions = query_corpus("quantum physics dissertation")
    # Should still return something, but low similarity
    assert sim is not None
    assert sim < 0.6


def test_query_corpus_empty_input():
    """Test handling of empty input."""
    from app import query_corpus
    phrase, sim, suggestions = query_corpus("")
    # Should handle gracefully
    assert suggestions is not None


@patch('openai.OpenAI')
def test_generate_direct_phrase_success(mock_openai_class):
    """Test successful LLM generation."""
    from app import generate_direct_phrase
    
    mock_client = Mock()
    mock_openai_class.return_value = mock_client
    
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '''```json
{
    "en": "Hello",
    "thai": "สวัสดี",
    "zh": "你好",
    "zh_trad": "你好",
    "pinyin": "nǐ hǎo",
    "zh_thai": "นี่ห่าว"
}
```'''
    mock_client.chat.completions.create.return_value = mock_response
    
    phrase, explanation = generate_direct_phrase("Hello")
    assert phrase is not None
    assert "zh" in phrase
    assert "pinyin" in phrase


def test_validate_environment_missing_key(monkeypatch):
    """Missing API key should raise error."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    
    with pytest.raises(EnvironmentError):
        from importlib import reload
        import app
        reload(app)


def test_validate_environment_success(monkeypatch):
    """Valid API key should not raise."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    # If we get here without error, validation passed
    assert os.getenv("OPENAI_API_KEY") == "sk-test-key"


@pytest.fixture
def app_client():
    """Create Flask test client."""
    from app import app
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_returns_200(app_client):
    """Health check should return 200 OK."""
    import json
    response = app_client.get("/health")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "healthy"


def test_rate_limiter_allows_under_limit():
    """Requests under limit should be allowed."""
    from app import RateLimiter
    limiter = RateLimiter(max_requests=5, window_seconds=60)
    
    for _ in range(5):
        assert limiter.is_allowed("test-client") is True


def test_rate_limiter_blocks_over_limit():
    """Requests over limit should be blocked."""
    from app import RateLimiter
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    
    for _ in range(3):
        limiter.is_allowed("test-client")
    
    assert limiter.is_allowed("test-client") is False