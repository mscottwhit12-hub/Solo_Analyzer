"""Colors and drawing helpers shared by the Solo Analyzer widgets."""

from PyQt6.QtGui import QColor

DEGREE_COLORS = {
    "1": (255, 100, 100),
    "2": (255, 165, 0),
    "3": (100, 180, 255),
    "♭3": (100, 180, 255),
    "4": (255, 105, 180),
    "♭5": (0, 200, 200),
    "5": (180, 100, 255),
    "6": (255, 165, 0),
    "♭6": (255, 165, 0),
    "♭7": (255, 165, 0),
    "7": (255, 215, 0),
    "Out": (80, 80, 90),
}

BACKGROUND = QColor(15, 15, 20)
BORDER = QColor(40, 40, 50)
PLACEHOLDER_TEXT = QColor(100, 100, 100)
UI_FONT = "Segoe UI"


def degree_color(degree, fallback, alpha=255):
    """Color for a scale degree label, or `fallback` if the label has no assigned color."""
    rgb = DEGREE_COLORS.get(degree)
    return QColor(*rgb, alpha) if rgb else fallback


def paint_background(painter, width, height):
    painter.fillRect(0, 0, width, height, BACKGROUND)
    painter.setPen(BORDER)
    painter.drawRect(0, 0, width - 1, height - 1)
