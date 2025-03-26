"""Shared prompts for RAG implementations"""

SUFFICIENCY_TEMPLATE = """You are a helpful assistant evaluating if the retrieved context is sufficient to answer a question.

Question: {query}

Retrieved Context:
{context}

First, analyze what information is needed to answer the question completely and accurately.
Then, evaluate if the retrieved context contains the necessary information.

If the context is sufficient to provide a complete and accurate answer, respond with "SUFFICIENT".
If the context is missing important information, respond with "INSUFFICIENT" followed by a specific refined search query 
that would help retrieve the missing information.

Your evaluation:"""

ANSWER_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context.
If the context doesn't contain the answer, just say you don't know and keep your answer short.
Do not make up information that is not provided in the context.

Context:
{context}

Question: {query}

Provide a comprehensive, accurate, and well-structured answer to the question based on the provided context."""

PLANNING_TEMPLATE = """You're an AI assistant that helps break down complex questions into simpler sub-questions.

Question: {query}

First, analyze if this question is complex (requires multiple pieces of information or multiple steps to answer completely).

If the question is SIMPLE (can be answered in one step with a single piece of information), respond with:
"SIMPLE"

If the question is COMPLEX, break it down into 2-4 sub-questions that would help answer the main question.
Format your response as:
"COMPLEX
1. [First sub-question]
2. [Second sub-question]
..."

Your analysis:"""

CRITIQUE_TEMPLATE = """You are a critical evaluator reviewing an answer to a user's question.
Question: {query}

Provided Answer: {answer}

Retrieved Context:
{context}

First, analyze if the answer:
1. Directly addresses the user's question
2. Is factually correct according to the context
3. Avoids making up information not in the context
4. Is complete and thorough
5. Is well-organized and clear

Then, identify any issues or ways to improve the answer.
Finally, rewrite the answer to address these issues. The revised answer should be comprehensive, accurate, and well-structured.

Your critique:""" 