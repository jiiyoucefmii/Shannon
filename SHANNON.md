# Shannon — the NLP tutor platform

Double-click **Shannon.bat**. A window opens (keep it open; closing it stops Shannon) and your browser shows
http://127.0.0.1:8765 after about a minute. Everything runs on this PC; nothing is reachable from the network.

## For students
- **Chat**: ask about NLP. *Hint* mode guides you with a question, *Explain* gives a full explanation. Math is
  rendered, code is coloured, every Python block is checked (syntax, unsafe calls such as `eval` or `shell=True`,
  secrets in code, undefined names, common style mistakes). Numbers like `[1]` open the course passage behind a
  sentence. Rate answers with the thumbs.
- **Practice**: pick a topic, a type (multiple choice, short answer, coding) and a difficulty. Shannon writes a new
  question from the course material, checks it, and marks your answer straight away with feedback.
- **Assignments**: quizzes and exams your instructor published. Exams allow one attempt and hide the breakdown.
- **My progress**: your marks per topic, first vs latest attempt.

## For instructors (sidebar → Instructor)
- **Question bank**: generate up to 5 questions at a time. Each is verified before you see it:
  multiple-choice keys are re-solved blind, marking points must be supported by the course text, the model
  answer must earn its own marks, coding exercises are run against their tests (wrong tests are dropped) and the
  starter code must fail. You still **approve, edit or reject** every question; only approved ones can be used.
- **Assessments**: tick approved questions, create a quiz or exam, publish it.
- **Submissions**: every answer with its automatic mark, the test results, the marking points, and flags
  (copying from another student, from the reference answer or from the course text; attempts to instruct the
  grader). Set your own mark and a comment; students see them.
- **Insights**: grading consistency (automatic vs your marks), explanation helpfulness (ratings), concept
  retention (same student, same topic, later attempts), topics and hardest questions, and exports
  (the approved problem bank as JSON lines, submissions as CSV).
- To protect this area with a PIN, start Shannon with
  `.venv\Scripts\python scripts\shannon_server.py --open --instructor-pin 1234`.

## How marking works
- Multiple choice: exact match.
- Short answer: one yes/no decision per marking point by the base model (same answer → same mark), then feedback.
- Coding: banned imports/calls are refused before running; then the tests run in a separate, time-limited Python
  process that cannot write files, start programs or use the network: 7 marks for tests, 3 for the explanation.

## Limits (read before using it for real grades)
- The model is small (4B). Generated answer keys can be wrong (seen in testing: a wrong "discount factor"), which
  is why every question needs instructor approval, and automatic marks should be spot-checked.
- The code runner is a guard for course exercises, not a security sandbox for untrusted code.
- One GPU: requests take turns. Other programs using the GPU (browsers, Figma) make Shannon several times slower.
- Data: `data/shannon.db` (questions, submissions, ratings). Chats stay in each browser.
