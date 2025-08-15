from pydantic import BaseModel
from llm_client import SafeLLMClient


class Evaluation(BaseModel):
    evaluation_reasoning: str
    has_sufficient_information: bool
    

class InformationEvaluator(SafeLLMClient):
    def __init__(self):
        super().__init__()
        self.system_prompt = f'''\
You are a rigorous fact-checker with a skeptical mindset, tasked with evaluating the trustworthiness of news articles. You have previously generated several search queries, and the search engine has retrieved relevant information segments. Your role is to be highly critical and demanding when determining whether you have collected sufficient information to confidently evaluate the trustworthiness of the given news article.

Apply this framework with extreme scrutiny to assess whether you have enough information:

1. Investigate the Source: Thoroughly examine the publisher's or author's background and reputation. Look for any red flags including credibility issues, potential biases, questionable ownership structures, past controversies, or conflicts of interest. Be suspicious of sources with limited track records or unclear funding.

2. Check the Claims: Identify and rigorously challenge every central factual claim or assertion made in the article. Demand verification from multiple reputable external sources, independent coverage, official statements, or expert opinions. Be skeptical of claims that lack proper substantiation.

3. Trace the Information: Meticulously track specific quotes, statistics, or media back to their original sources. Insist on finding the original context of quotes and verify all numbers/statistics with official data. Be wary of second-hand reporting or unverified information chains.

4. Use Multiple Angles (Lateral Reading): Extensively check how widely the claim is reported and critically assess who endorses or disputes it. Actively seek out contradictory evidence, debunking efforts, alternative perspectives, and dissenting opinions from reputable third-party sources.

Maintain a highly skeptical stance—assume information is insufficient unless you have overwhelming evidence to the contrary. Set a high bar for what constitutes "sufficient information." Only consider information adequate if you can thoroughly verify claims from multiple independent, credible sources and have addressed potential counterarguments. First, explain your critical reasoning in detail, then provide your boolean evaluation decision regarding whether you have truly sufficient information.
'''
 


    def clean_retrieval_data(self, query_retrieval_history: str):
        """Remove rationale and deduplicate information to reduce noise"""
        import json
        
        try:
            data = json.loads(query_retrieval_history)
            cleaned_data = {}
            
            seen_segments = set()
            
            for query_key, query_data in data.items():
                # Keep only query and retrieved segments, remove rationale
                cleaned_query = {
                    "query": query_data["query"],
                    "retrieved_segments": []
                }
                
                # Deduplicate segments based on URL and segment_text
                for segment in query_data["retrieved_segments"]:
                    segment_hash = f"{segment['url']}_{segment['segment_text'][:100]}"
                    if segment_hash not in seen_segments:
                        seen_segments.add(segment_hash)
                        # Keep only essential fields
                        clean_segment = {
                            # "url": segment["url"],
                            # "title": segment["title"], 
                            "segment_text": segment["segment_text"]
                        }
                        cleaned_query["retrieved_segments"].append(clean_segment)
                
                if cleaned_query["retrieved_segments"]:  # Only include if has segments
                    cleaned_data[query_key] = cleaned_query
            
            return json.dumps(cleaned_data, indent=2)
            
        except json.JSONDecodeError:
            # If not valid JSON, return simplified version
            return "Error parsing retrieval history - invalid JSON format"
    

    def evaluate(self, article: str, query_retrieval_history: str):
        cleaned_history = self.clean_retrieval_data(query_retrieval_history)

        user_input = f'''
Here is the news article:
{article}

Here are the previously generated queries with retrieved segments:
{cleaned_history}



IMPORTANT: Respond with valid JSON only. No additional text or explanations.
Required format:
{{
    "evaluation_reasoning": "your detailed reasoning here",
    "has_sufficient_information": true/false
}}'''
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]
        response = self.generate_structured(
            response_model=Evaluation,
            messages=messages,
            temperature=0
        )

        evaluation_reasoning = response.evaluation_reasoning
        has_sufficient_information = response.has_sufficient_information

        return evaluation_reasoning, has_sufficient_information
