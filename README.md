# MathTutor: AI-Powered Learning Platform for BBA Students

## Overview

MathTutor is an intelligent tutoring system specifically designed for BBA (Bachelor of Business Administration) students who need guidance in quantitative business subjects like finance, statistics, economics, and accounting. The platform goes beyond simply providing answers - it creates a personalized learning experience that emphasizes explanation, understanding, and progressive skill development.

## Core Educational Principles

1. **Explainability Over Answers**: Rather than just giving solutions, our system provides detailed Chain-of-Thought explanations that break down complex problems into understandable steps.

2. **Personalized Learning Progression**: The system adapts to each student's level of understanding, providing hints of increasing specificity before revealing complete solutions.

3. **Hybrid Guidance Approach**: We've implemented a system that blends user-driven choices with intelligent recommendations - students maintain control while receiving contextually appropriate suggestions.

4. **Verification and Correctness**: Each solution undergoes rigorous verification to ensure mathematical accuracy and pedagogical clarity.

5. **Subject-Specific Context**: The system recognizes the different requirements of various business disciplines, from time-value-of-money calculations to statistical analyses.

## Implementation Versions

### Simple Implementation (`math_tutor_simple.py`)

We've created a streamlined version that demonstrates the core functionality:

- Problem classification (math vs. general question)
- Step-by-step solution generation
- Progressive hint system (3 levels)
- Basic answer assessment
- Command-line demo interface

To try this version:
```
python -m app.demo
```

### Full Implementation (`main.py`)

Our comprehensive implementation includes:

- Advanced state management
- Solution verification and regeneration
- Temporal tracking of student progress
- Web interface with Streamlit
- Specialized mathematical tools

To run the full version:
```
streamlit run app/main.py
```

## Project Structure

```
app/
├── main.py                      # Full implementation entry point
├── math_tutor_simple.py         # Simplified implementation
├── demo.py                      # Command-line demo
├── test_math_tutor.py           # Test script
├── math_services/               # Core tutoring services
│   ├── main.py                  # MathTutor orchestration
│   ├── prompts.py               # Centralized prompt templates
│   ├── state.py                 # State management
│   ├── services/                # Specialized service modules
│   │   ├── verification.py      # Solution verification
│   │   ├── reasoning.py         # Solution generation
│   │   ├── hint_generation.py   # Hint generation
│   │   ├── proximity_assessment.py # Answer evaluation
│   │   └── response_parsing.py  # Variable extraction
│   └── tools/                   # Specialized mathematical tools
│       └── math_tools.py        # Financial, statistical tools
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/mathtutor.git
   cd mathtutor
   ```

2. Create a virtual environment:
   ```
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

5. Run the simple demo:
   ```
   python -m app.demo
   ```

## Usage - Simple Demo

1. Type in a math problem or business-related question
2. The system will categorize it and generate a solution
3. For math problems, you can:
   - Request progressive hints (Level 1-3)
   - View the complete solution
   - Submit your answer for evaluation
4. For general questions, you'll receive a comprehensive explanation

## Development Approach

We're following an incremental development approach:

1. [x] Simple monolithic implementation (`math_tutor_simple.py`)
2. [x] Dual-state model (internal state and user-facing responses)
3. [x] Command-line demo
4. [x] Modularized components and services
5. [x] Mathematical tool integration
6. [x] Web UI with Streamlit

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 