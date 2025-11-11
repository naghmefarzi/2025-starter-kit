# from pydantic import BaseModel, Field
# from llm_client import SafeLLMClient


# class RoastedArticle(BaseModel):
#     article: str = Field(description="A concise critique of the news article exposing credibility gaps")


# class Roaster(SafeLLMClient):
#     def __init__(self):
#         super().__init__()
#         self.system_prompt = '''You are a credibility analyst. Test the given article by finding opposing evidence and viewpoints.

# ANALYZE FOR:
# 1. Contradictory evidence from reputable sources
# 2. Expert opinions that disagree with article claims  
# 3. Source credibility issues (bias, funding, track record)
# 4. Methodology flaws in cited studies
# 5. Unsupported claims or missing sources

# OUTPUT:
# Write a brief critique exposing credibility gaps. If article is credible, state that.
# Return only JSON: {"article": "your critique here"}

# Be factual, specific, objective. Focus on evidence-based contradictions, not opinions.'''
        
#     def roast(self, article: str):
#         user_input = f'''Here is the news article to analyze:

# {article}

# ---

# Analyze this article following the framework provided and return your response as a JSON object with this exact structure:

# {{
#     "article": "your critique text here"
# }}

# The critique should read like a brief investigative news article that exposes the credibility gaps in the original piece.'''
        
#         messages = [
#             {"role": "system", "content": self.system_prompt},
#             {"role": "user", "content": user_input}
#         ]
#         response = self.generate_structured(
#             response_model=RoastedArticle,
#             messages=messages,
#             temperature=0.3
#         )
        
#         roasted_article = response.article

#         print(f"ROASTED ARTICLE: {roasted_article}")
#         return roasted_article


from pydantic import BaseModel, Field
from llm_client import SafeLLMClient


class RoastedArticle(BaseModel):
    article: str = Field(description="A concise critique aimed at debunking the news article entirely")


class Roaster(SafeLLMClient):
    def __init__(self):
        super().__init__()
        self.system_prompt = '''You are a credibility analyst tasked with convincing the user that no part of the provided news article is true. Your goal is to thoroughly debunk the article by identifying and presenting evidence that contradicts its claims.

ANALYZE FOR:
1. Contradictory evidence from reputable sources that disproves the article's claims
2. Expert opinions that directly refute the article's assertions
3. Source credibility issues (bias, funding, track record) that undermine the article
4. Methodology flaws in any cited studies or data
5. Unsupported claims, missing sources, or logical inconsistencies
6. Aspects of the article that could be improved to enhance credibility (e.g., better sourcing, transparency, or methodology)

OUTPUT:
Return a JSON object with two fields:
- "article": A concise critique that convincingly argues no part of the article is true, written like an investigative news report.

Format: {"article": "your critique here"}

Be factual, specific, and objective. Rely on evidence-based contradictions, not opinions.'''
        
    def roast(self, article: str):
        user_input = f'''Here is the news article to analyze:

{article}

---

Analyze this article with the goal of convincing that no part of it is true, following the framework provided. Return your response as a JSON object with this exact structure:

{{
    "article": "your critique text here",
}}

The critique should read like a concise investigative news report that exposes why the article is entirely unreliable.'''
        
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
        # aspects_to_improve = response.aspects_to_improve

        print(f"ROASTED ARTICLE: {roasted_article}")
        # print(f"ASPECTS TO IMPROVE: {aspects_to_improve}")
        return roasted_article
