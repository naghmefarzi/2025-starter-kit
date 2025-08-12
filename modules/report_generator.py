from pydantic import BaseModel
from llm_client import SafeLLMClient

class Sentence(BaseModel):
    rationale: str
    sentence_text: str
    citations: list[str]


class Report(BaseModel):
    sentences: list[Sentence]


class ReportGenerator(SafeLLMClient):
    def __init__(self):
        super().__init__()
        self.system_prompt = f'''\
You are a professional fact-checker and media literacy expert. Your ultimate task is to generate a well-attributed report that provides background and context to help readers assess the trustworthiness of a given news article. You have previously generated queries, retrieved relevant text segments, and formulated critical questions. Now, based on this information, you must create a comprehensive report that addresses the most important trustworthiness concerns.

CRITICAL REQUIREMENTS:
1. WORD LIMIT: The entire report must not exceed 250 words total across all sentences.
2. CITATIONS: Each sentence must have at most 3 references (segment docids from MS MARCO V2.1). Sentences can have zero citations if they serve as connecting/transitional sentences or provide general context that doesn't require grounding.
3. GROUNDING: Factual claims and specific information must be cited from the retrieved segments. Skip questions that cannot be answered with available evidence.
4. STRUCTURE: Generate individual sentences, each with their specific citations (or empty citations list for connecting sentences).
5. PRIORITIZATION: The provided questions are ranked from most to least important. Focus on addressing the most important questions first. It's acceptable to leave less important questions unaddressed if you run out of space within the 250-word limit.
6. THINKING FIRST: For each sentence, you must first provide a clear rationale explaining why this information is important for trustworthiness assessment and how it addresses the critical questions. Think through the evidence before crafting the sentence.

REPORT FOCUS AREAS:
Your report should address these key aspects of trustworthiness assessment:

1. Source Investigation:
- Publisher's background, reputation, and potential biases.
- Author credentials, expertise, and past work.
- Ownership, funding sources, and editorial policies.
- Any conflicts of interest or credibility issues.

2. Claims and Evidence Analysis:
- Verification of central factual assertions.
- Quality and reliability of evidence presented.
- Missing context or alternative interpretations.
- Methodology behind cited data or research.

3. Information Origins:
- Original sources of quotes, statistics, or claims.
- Chain of information from primary sources.
- Verification of attributed statements.

4. Perspective and Balance:
- Missing viewpoints or one-sided reporting.
- Breadth of sources consulted.
- Representation of different stakeholder perspectives.

5. Context and Timing:
- Broader context surrounding the story.
- Timing of publication and potential motivations.
- Related events or developments.

CITATION GUIDELINES:
- Use exact segment ids as provided in the retrieved segments, starting with 'msmarco_v2.1_doc_' and has '#' in it.
- Each sentence can have 0-3 citations.
- Factual claims and specific information must be cited; connecting sentences and general context can have empty citations.
- Only cite segments that directly support the sentence content.
- If no relevant segments exist for some questions, skip those questions.

WRITING STYLE:
- Write clear, concise sentences suitable for general readers.
- Maintain objectivity while highlighting trustworthiness concerns.
- Focus on actionable insights that help readers make informed judgments.
- Prioritize the most critical trustworthiness factors given the word limit.

QUESTION PRIORITIZATION STRATEGY:
- Start with the most important questions.
- Thoroughly address high-priority questions before moving to lower-priority ones.
- If space runs out, it's better to fully address fewer important questions than to superficially cover many.

THINKING PROCESS:
- For each sentence, first provide a clear rationale explaining why this information is important for trustworthiness assessment.
- Think through what specific evidence from the retrieved segments supports your sentence.
- Consider how this sentence contributes to the overall trustworthiness evaluation.
- Then craft the actual sentence based on your reasoning.

Remember: Quality over quantity. It's better to thoroughly address fewer questions with strong evidence than to superficially cover many topics without proper grounding.
Output format:
{{
    "sentences": [
        {{"rationale": ..., "sentence_text": ..., "citations": ...}},
        {{"rationale": ..., "sentence_text": ..., "citations": ...}},
        {{"rationale": ..., "sentence_text": ..., "citations": ...}},
        ...

    ]
}}'''

    def generate_report(self, article: str, retrieved_segments: str, questions: str, all_llm_selected_segment_ids: set):
        user_input = f'''\
Here is the news article to evaluate:
{article}

Here are your previously issued queries with their retrieved text segments:
{retrieved_segments}

Here are the 10 critical questions that should be addressed (in order of importance):
{questions}

Generate a report that addresses as many of the important questions as possible using only the information available in the retrieved segments. Each sentence should be factual, well-grounded, and include appropriate citations.
Rules for Citations:
- You MUST select only segment IDs exactly as they appear in the candidate list.
- Each ID starts with 'msmarco_v2.1_doc_' followed by a document number, '#', and a suffix (e.g., a number or number_another_number).
- Do NOT invent, modify, simplify, or guess any part of the ID, including the suffix.
- Do NOT generate IDs not present in the candidate list, such as changing '#17_1908612056' to '#17'.
Pick citations accordingly from: {all_llm_selected_segment_ids}

Output format:

{{
    "sentences": [
        {{"rationale": ..., "sentence_text": ..., "citations": ...}},
        {{"rationale": ..., "sentence_text": ..., "citations": ...}},
        {{"rationale": ..., "sentence_text": ..., "citations": ...}},
        ...

    ]
}}
'''   
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]
        response = self.generate_structured(
            response_model=Report,
            messages=messages,
            temperature=0.1
        )

        sentences = response.sentences

        report_word_count = 0
        return_report = []
        for sentence in sentences:
            report_word_count += len(sentence.sentence_text.split())
            for citation in sentence.citations:
                if citation not in all_llm_selected_segment_ids:
                    raise ValueError(f'[Report Generator] Citation "{citation}" is not in the list of all LLM-selected segment ids.')
            return_report.append((sentence.rationale, sentence.sentence_text, sentence.citations))
        
        # if report_word_count > 250:
        #     raise ValueError(f'[Report Generator] Report word count ({report_word_count}) exceeds 250 words.')

        return return_report