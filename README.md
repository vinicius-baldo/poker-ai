# PokerAI

An intelligent poker assistant that uses AI to analyze poker games, track opponent behavior, and provide strategic recommendations in real-time.

## Project Overview

PokerAI is designed to:

- **Read poker table information** from online poker platforms (PokerStars, etc.)
- **Track and analyze opponent behavior** to build player profiles
- **Provide strategic recommendations** using LLM-powered analysis
- **Consider multiple factors** including hand strength, opponent history, and game context

## Core Features

### 🎯 Real-time Analysis

- Screen capture and OCR for table reading
- Hand strength evaluation
- Pot odds calculation
- Position analysis

### 👥 Opponent Profiling

- Track betting patterns
- Analyze playing style (tight/loose, aggressive/passive)
- Build hand range databases
- Historical performance tracking

### 🤖 AI-Powered Decisions

- LLM integration for strategic recommendations
- Multi-factor decision analysis
- Hand range estimation for opponents
- Action recommendations with reasoning

## Architecture

```
PokerAI/
├── src/
│   ├── core/           # Core poker logic and game state
│   ├── vision/         # Screen capture and OCR
│   ├── analysis/       # Hand analysis and odds calculation
│   ├── profiling/      # Opponent tracking and analysis
│   ├── ai/            # LLM integration and decision making
│   ├── ui/            # User interface and controls
│   └── utils/         # Utilities and helpers
├── data/              # Opponent profiles and game data
├── config/            # Configuration files
├── tests/             # Test suite
└── docs/              # Documentation
```

## Technology Stack

- **Python** - Core application logic
- **OpenCV** - Screen capture and image processing
- **Tesseract** - OCR for reading table information
- **OpenAI API** - LLM for strategic decisions
- **SQLite/PostgreSQL** - Data storage for opponent profiles
- **PyQt/Tkinter** - User interface
- **NumPy/Pandas** - Data analysis and manipulation

## Getting Started

1. Clone the repository
2. Install dependencies (see requirements.txt)
3. Configure API keys and settings
4. Run the application

## License

MIT License
