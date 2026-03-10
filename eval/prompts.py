"""Judge prompt templates for RAG evaluation."""

FAITHFULNESS_JUDGE_PROMPT = """\
You are an impartial judge evaluating whether an AI-generated answer is \
faithful to the provided context. Every claim in the answer must be supported \
by the context.

Context:
{rag_context}

Generated Answer:
{generated_answer}

Score the faithfulness on a scale of 1-5:
1 = Most claims are fabricated or contradicted by context
2 = Several unsupported claims
3 = Some claims supported, some not
4 = Most claims supported by context
5 = All claims are directly supported by context

Respond with ONLY a JSON object:
{{"score": <int 1-5>, "reasoning": "<brief explanation>"}}
"""

CORRECTNESS_JUDGE_PROMPT = """\
You are an impartial judge evaluating whether an AI-generated answer \
captures the key points from a reference answer.

Reference Answer:
{ground_truth}

Generated Answer:
{generated_answer}

Score the correctness on a scale of 1-5:
1 = Completely misses the key points
2 = Captures few key points
3 = Captures some key points but misses important ones
4 = Captures most key points
5 = Captures all key points accurately

Respond with ONLY a JSON object:
{{"score": <int 1-5>, "reasoning": "<brief explanation>"}}
"""
