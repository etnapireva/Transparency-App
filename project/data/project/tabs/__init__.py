# Tabs package - each module exposes a render() function

from .dashboard import render as render_dashboard
from .sentiment import render as render_sentiment
from .topics import render as render_topics
from .style_metrics import render as render_style_metrics
from .speaker_comparison import render as render_speaker_comparison
from .qa import render as render_qa
from .evaluation import render as render_evaluation
from .methodology import render as render_methodology

__all__ = [
    "render_dashboard",
    "render_sentiment",
    "render_topics",
    "render_style_metrics",
    "render_speaker_comparison",
    "render_qa",
    "render_evaluation",
    "render_methodology",
]
