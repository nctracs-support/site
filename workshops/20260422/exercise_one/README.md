
## Prompt 1
Ask me one question at a time so we can develop a thorough, step-by-step spec for this idea. Each question should build on my previous answers, and our end goal is to have a detailed specification I can hand off to a developer. Let’s do this iteratively and dig into every relevant detail. Remember, only one question at a time. 

Here’s the idea: 

Create a command-line tool that reads a CSV file of daily temperature readings, detects anomalies (days that differ by more than 2 standard deviations from a 30-day rolling average), and outputs:
- A list of flagged dates
- The magnitude of deviations
- A simple ASCII chart highlighting anomalies

Constraints:
- Must validate the schema
- Must explicitly handle missing values
- Must not silently drop rows
- Must include tests
- Must follow AGENTS.md guidelines

## Prompt 2
Now that we’ve wrapped up the brainstorming process, can you compile our findings into a comprehensive, developer-ready specification? 

Include all relevant requirements, architecture choices, data handling details, error handling strategies, and a testing plan so a developer can immediately begin implementation.

## Prompt 3
You are a senior software architect reviewing a developer-ready specification for an agentic coding system.
Evaluate whether the specification is: Clear, Complete, Consistent, Correct, Testable, and Unambiguous.

Identify:
- Missing requirements or underspecified behaviors
- Ambiguities or subjective language
- Inconsistencies or conflicting constraints
- Hidden assumptions or unstated dependencies
- Edge cases, failure modes, and security risks not addressed
- Gaps in data handling, error handling, and observability
- Testability and verifiability weaknesses

For each issue:
- Quote the problematic section
- Explain the risk
- Propose a precise improvement

Conclude with a readiness assessment:

Is this specification safe to hand off to a reasoning LLM to generate an implementation plan? Why or why not?

`Attach specification.md file`

## Prompt 4
Draft a detailed, step-by-step blueprint for building this project. Then, once you have a solid plan, break it down into small, iterative chunks that build on each other. Look at these chunks and then go another round to break it into small steps. Review the results and ensure the steps are small enough to implement safely with strong testing but large enough to move the project forward. Iterate until you feel the steps are the right size for this project.

From here, you should have the foundation to provide a series of prompts for a code-generation LLM that will implement each step in a test-driven manner. Prioritize best practices, incremental progress, and early testing, ensuring no big jumps in complexity at any stage. Make sure that each prompt builds on the previous prompts, and ends with wiring things together. There should be no hanging or orphaned code that isn't integrated into a previous step. At each prompt, provide ways for the human to validate the implementation.

Make sure to separate each prompt section. Use markdown. Each prompt should be tagged as text using code tags. The goal is to output prompts, but context, etc is important as well.

`Attach specification`

## Now Build
