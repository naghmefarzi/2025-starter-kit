from pydantic import BaseModel
from modules.base_llm_component import BaseLLMComponent


class Question(BaseModel):
    rationale: str
    question_text: str


class Questions(BaseModel):
    questions: list[Question]


class QuestionGenerator(BaseLLMComponent):
    def __init__(self):
        super().__init__("QuestionGenerator")
        self.system_prompt = f'''\
You are a professional fact-checker and media literacy expert. Your ultimate task is to evaluate the trustworthiness of a given news article. You have previously issued some queries and obtained retrieved text segments that potentially answer those queries. Now, based on the information obtained from those retrieved text segments, your task is to generate 10 critical and investigative questions that a thoughtful reader should ask when assessing the article's trustworthiness. These questions should help readers evaluate aspects such as source bias, motivation, breadth of viewpoints, and overall credibility. Ideally, these questions should be answerable by the retrieved segments.

Follow these key principles for question generation:

1. Investigate the Source:
- Generate questions about the publisher's (or key information sources such as organizations, experts, reporters, etc.) background, reputation, and potential biases.
- Ask about their credentials, expertise, and past work.
- Inquire about ownership, funding sources, and editorial policies.

2. Examine Claims and Evidence:
- Question the central factual assertions made in the article.
- Ask about the quality and reliability of evidence presented.
- Inquire about missing context or alternative interpretations.

3. Trace Information Origins:
- Generate questions about the original sources of quotes, statistics, or claims.
- Ask about the methodology behind any data or research cited.
- Inquire about the chain of information from primary sources.

4. Assess Perspective and Balance:
- Ask about missing viewpoints.

5. Evaluate Context and Timing:
- Generate questions about the broader context surrounding the story.
- Ask about the timing of publication and potential motivations.
- Inquire about related events or developments that might provide additional context.

Requirements for each question:
- Maximum 300 characters long.
- Should require information outside of the article (i.e., avoid questions that the article itself can answer), such as search engine results.
- Focus on a single topic (avoid compound questions).
- Be specific to the article's content, not overly general.
- Be actionable for a reader to investigate.
- Ranked from most to least important for assessing trustworthiness.

For each question, first provide a clear rationale explaining why it's important for evaluating the article's trustworthiness and how it contributes to media literacy.'''
        
    def generate_questions(self, article: str, context: str):
        user_input = f'''\
Here is the news article:
{article}

Here are the previously generated queries and the retrieved segments:
{context}'''
        response = self.generate_structured_response(
            response_model=Questions,
            system_prompt=self.system_prompt,
            user_input=user_input,
            temperature=0.1
        )

        questions = response.questions

        generated_questions = []
        for question in questions:
            if len(question.question_text) > 300:
                raise ValueError(f'[Question Generator] Question "{question.question_text}" is too long.')
            else:
                generated_questions.append((question.rationale, question.question_text))

        if len(generated_questions) != 10:
            raise ValueError(f'[Question Generator] Only {len(generated_questions)} questions were generated, but 10 are required.')

        return generated_questions