# MathTutor: AI-Powered Learning Platform for BBA Students

## Overview

MathTutor is an intelligent tutoring system specifically designed for BBA (Bachelor of Business Administration) students who need guidance in quantitative business subjects like finance, statistics, economics, and accounting. The platform goes beyond simply providing answers - it creates a personalized learning experience that emphasizes explanation, understanding, and progressive skill development.

## Core Educational Principles

1. **Explainability Over Answers**: Rather than just giving solutions, our system provides detailed Chain-of-Thought explanations that break down complex problems into understandable steps.

2. **Personalized Learning Progression**: The system adapts to each student's level of understanding, providing hints of increasing specificity before revealing complete solutions.

3. **Hybrid Guidance Approach**: We've implemented a system that blends user-driven choices with intelligent recommendations - students maintain control while receiving contextually appropriate suggestions.

4. **Verification and Correctness**: Each solution undergoes rigorous verification to ensure mathematical accuracy and pedagogical clarity.

5. **Subject-Specific Context**: The system recognizes the different requirements of various business disciplines, from time-value-of-money calculations to statistical analyses.

6. **Interactive Visual Learning**: The system enhances understanding through visualizations, interactive calculations, and comparative analyses.

## Key Features

### Core Tutoring Capabilities

- Problem classification (math vs. general question)
- Step-by-step solution generation
- Progressive hint system with multiple levels
- Answer assessment and feedback
- Solution verification and regeneration
- State tracking for personalized learning

### Interactive Tools

- **Visualization Tools**: Plot mathematical functions, create visualizations of financial concepts
- **Calculation Tools**: Specialized calculators for financial, statistical, and accounting problems
- **Practice Problem Generation**: Create custom practice problems at varying difficulty levels
- **Equation Solving**: Step-by-step equation solving with explanations
- **Investment Comparison**: Compare different investment options with visual growth charts

## LangGraph Implementation

MathTutor leverages the LangGraph framework to create a robust, modular architecture with:

- Advanced state management
- Node-based workflow processing
- Conditional graph routing
- State observation and monitoring

To run the application:
```
langgraph dev
```

## Architecture

### Modular Architecture

The codebase has been refactored into a modular architecture:

```
app/
├── main.py                      # Graph definition and main entry point
├── models.py                    # State models
├── prompts.py                   # Prompt templates
├── test_graph.py                # Test script for the graph
├── demo_tools.py                # Demonstrations of custom tools
│
├── math_services/               # Core math tutoring services
│   ├── services/                # Service modules
│   │   ├── node_services.py     # Core node functions for the graph
│   │   ├── verification.py      # Solution verification
│   │   └── proximity_assessmnet.py # Answer evaluation
│   │
│   └── tools/                   # Interactive tools
│       ├── math_tools.py        # Core math calculation functions
│       ├── interactive_tools.py # Visual and interactive tools
│       └── tool_integration.py  # LangChain tools integration
```

### State Flow

1. **Input State**: Problem, requested action, user answer
2. **Internal State**: Complete processing state with tracking
3. **Output State**: User-visible results and feedback

### Node-based Processing

The workflow is built using a graph of processing nodes:

1. **Initial Nodes**:
   - Initialize session
   - Classify workflow

2. **Math Workflow**:
   - Parse variables
   - Define context
   - Generate solution
   - Verify solution
   - Generate hints
   - Process user answer
   - Format response

3. **Conditional Routing**:
   - Math vs. general questions
   - Need for solution regeneration
   - Next suggestions based on user progress

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

5. Run the application:
   ```
   langgraph dev
   ```

6. To test the interactive tools:
   ```
   python -m app.demo_tools
   ```

## Interactive Tool Demonstrations

The `demo_tools.py` script demonstrates various interactive capabilities:

1. **Present Value Calculation**: Step-by-step calculation with visualization
2. **Function Visualization**: Plot mathematical functions and identify key points
3. **Equation Solving**: Solve algebraic equations with detailed explanations
4. **Practice Problem Generation**: Create custom practice problems based on topic and difficulty
5. **Investment Comparison**: Compare multiple investment options with visual growth charts

## Future Enhancements

1. **Student Profile Tracking**: Long-term learning progression and analytics
2. **Expanded Interactive Tools**: More subject-specific tools and visualizations
3. **Multi-step Problems**: Complex, multi-part business case studies
4. **Tool-integrated Learning Sessions**: Interactive tutoring sessions with integrated tools
5. **Adaptive Difficulty**: Dynamically adjust problem difficulty based on student performance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 