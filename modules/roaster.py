from pydantic import BaseModel, Field
from llm_client import SafeLLMClient


class RoastedArticle(BaseModel):
    article: str = Field(description="A concise critique of the news article exposing credibility gaps")


class Roaster(SafeLLMClient):
    def __init__(self):
        super().__init__()
        self.system_prompt = '''You are a credibility analyst. Test the given article by finding opposing evidence and viewpoints.

ANALYZE FOR:
1. Contradictory evidence from reputable sources
2. Expert opinions that disagree with article claims  
3. Source credibility issues (bias, funding, track record)
4. Methodology flaws in cited studies
5. Unsupported claims or missing sources

OUTPUT:
Write a brief critique exposing credibility gaps. If article is credible, state that.
Return only JSON: {"article": "your critique here"}

Be factual, specific, objective. Focus on evidence-based contradictions, not opinions.'''
        
    def roast(self, article: str):
        user_input = f'''Here is the news article to analyze:

{article}

---

Analyze this article following the framework provided and return your response as a JSON object with this exact structure:

{{
    "article": "your critique text here"
}}

The critique should read like a brief investigative news article that exposes the credibility gaps in the original piece.'''
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]
        response = self.generate_structured(
            response_model=RoastedArticle,
            messages=messages,
            temperature=0.3
        )
        
        roasted_article = response.article

        print(f"ROASTED ARTICLE: {roasted_article}")
        return roasted_article