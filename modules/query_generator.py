from pydantic import BaseModel
from llm_client import SafeLLMClient
import json

class QueryWithRationale(BaseModel):
    rationale: str
    query: str


class QueryReasoning(BaseModel):
    queries_with_rationale: list[QueryWithRationale]


class QueryGenerator(SafeLLMClient):

    def __init__(self):
        super().__init__()
        self.key_aspects = '''\
1. Investigate the Source:
- Examine the publisher or author's background and reputation.
- Queries should target credibility, potential biases, ownership, or past controversies.

2. Check the Claim:
- Identify the central factual claims or assertions made in the article.
- Queries should verify these claims against reputable external sources, independent coverage, official statements, or expert opinions.

3. Trace the Information:
- Trace specific quotes, statistics, or media back to their original sources.
- Queries should seek the original context of quotes or verify numbers/statistics with official data.

4. Use Multiple Angles (Lateral Reading):
- Check how widely the claim is reported and who else endorses or disputes it.
- Queries should include terms indicating external verification, debunking, support, or differing perspectives from reputable third-party sources.

5. Tailor Queries to the Article's Details:
- Incorporate precise and unique identifiers from the article, such as proper nouns, dates, locations, and specific terminology.
- Queries should be focused and specific, designed to efficiently surface credible verification or refutation of the claims.'''

    def generate_query(self, article: str, context: str = None, feedback: str = None):
        if context:
            system_prompt = f'''\
You are a professional fact-checker. Given a news article, your task is to carefully evaluate its trustworthiness. You will do this by generating detailed, well-thought-out search queries that a skilled fact-checker would issue to a search engine, each preceded by a clear rationale explaining why that query is essential for verifying the article's reliability. Note that search engine is based on the BM25 (with RM3) sparse retrieval algorithm. Frame your queries in a way that works best with it. Follow the framework below closely:

{self.key_aspects}

You have already generated some queries, and some document segments have been retrieved for them. However, the information obtained so far is still insufficient to assess the article's trustworthiness. Based on the previously generated questions and the retrieved segments, generate five additional queries that either address aspects not yet covered or rephrase queries that remain unanswered by the retrieved segments. For each search query you produce, you must provide a rationale clearly articulating why this particular query is important and how it contributes to your fact-checking process. Each rationale must demonstrate thoughtful analysis, critical thinking, and alignment with professional fact-checking practices.
Return a JSON object matching this schema:

{{
    "queries_with_rationale": [
        {{"rationale": ..., "query": ...}},
        {{"rationale": ..., "query": ...}},
        {{"rationale": ..., "query": ...}},
        ...

    ]
}}'''
            user_input = f'''\
Here is the news article:
{article}

Here are the previously generated queries with retrieved segments:
{context}

Feedback on the previously generated queries and retrieved segments: {feedback}'''
        else:
            system_prompt = f'''\
You are a professional fact-checker. Given a news article, your task is to carefully evaluate its trustworthiness. You will do this by generating five detailed, well-thought-out search queries that a skilled fact-checker would issue to a search engine, each preceded by a clear rationale explaining why that query is essential for verifying the article's reliability. Note that search engine is based on the BM25 (with RM3) sparse retrieval algorithm. Frame your queries in a way that works best with it. Follow the framework below closely:

{self.key_aspects}

For each search query you produce, you must provide a rationale clearly articulating why this particular query is important and how it contributes to your fact-checking process. Each rationale must demonstrate thoughtful analysis, critical thinking, and alignment with professional fact-checking practices.
Return a JSON object matching this schema:

{{
    "queries_with_rationale": [
        {{"rationale": ..., "query": ...}},
        {{"rationale": ..., "query": ...}},

        ...
    ]
}}'''
            user_input = f'''\
Here is the news article:
{article}

Return a JSON object matching this schema:
{{
    "queries_with_rationale": [
        {{"rationale": ..., "query": ...}},
        {{"rationale": ..., "query": ...}},

        ...
    ]
}}'''
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        try:
            response = self.generate_structured(
                response_model=QueryReasoning,
                messages=messages,
                temperature=0.3,
                top_p = 1
            )

            print(f"Generated {len(response.queries_with_rationale)} queries successfully")
            
            generated_queries = [
                (query_with_rationale.query, query_with_rationale.rationale)
                for query_with_rationale in response.queries_with_rationale
            ]
            
            if len(generated_queries) < 5:
                raise ValueError(f'[Query Generator] Only {len(generated_queries)} queries were generated, but 5 are required.')
            
            return generated_queries
            
        except Exception as e:
            raise RuntimeError(f"Query generation failed: {e}")