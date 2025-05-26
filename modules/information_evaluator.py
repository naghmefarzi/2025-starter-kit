import os
from openai import AzureOpenAI
from pydantic import BaseModel


class Evaluation(BaseModel):
    evaluation_reasoning: str
    has_sufficient_information: bool
    

class InformationEvaluator:
    def __init__(self):
        self.information_evaluator = AzureOpenAI(
            api_version=os.getenv('OPENAI_API_VERSION'),
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_key=os.getenv('AZURE_OPENAI_API_KEY')
        )
        self.system_prompt = f'''\
You are a rigorous fact-checker with a skeptical mindset, tasked with evaluating the trustworthiness of news articles. You have previously generated several search queries, and the search engine has retrieved relevant information segments. Your role is to be highly critical and demanding when determining whether you have collected sufficient information to confidently evaluate the trustworthiness of the given news article.

Apply this framework with extreme scrutiny to assess whether you have enough information:

1. Investigate the Source: Thoroughly examine the publisher's or author's background and reputation. Look for any red flags including credibility issues, potential biases, questionable ownership structures, past controversies, or conflicts of interest. Be suspicious of sources with limited track records or unclear funding.

2. Check the Claims: Identify and rigorously challenge every central factual claim or assertion made in the article. Demand verification from multiple reputable external sources, independent coverage, official statements, or expert opinions. Be skeptical of claims that lack proper substantiation.

3. Trace the Information: Meticulously track specific quotes, statistics, or media back to their original sources. Insist on finding the original context of quotes and verify all numbers/statistics with official data. Be wary of second-hand reporting or unverified information chains.

4. Use Multiple Angles (Lateral Reading): Extensively check how widely the claim is reported and critically assess who endorses or disputes it. Actively seek out contradictory evidence, debunking efforts, alternative perspectives, and dissenting opinions from reputable third-party sources.

Maintain a highly skeptical stance—assume information is insufficient unless you have overwhelming evidence to the contrary. Set a high bar for what constitutes "sufficient information." Only consider information adequate if you can thoroughly verify claims from multiple independent, credible sources and have addressed potential counterarguments. First, explain your critical reasoning in detail, then provide your boolean evaluation decision regarding whether you have truly sufficient information.'''
 
    def evaluate(self, article: str, query_retrieval_history: str):
        user_input = f'''
Here is the news article:
{article}

Here are the previously generated queries with retrieved segments:
{query_retrieval_history}'''
        
        completion = self.information_evaluator.beta.chat.completions.parse(
            model='gpt-4.1',
            messages=[
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': user_input}
            ],
            temperature=0,
            presence_penalty=0,
            frequency_penalty=0,
            response_format=Evaluation,
        )

        evaluation_reasoning = completion.choices[0].message.parsed.evaluation_reasoning
        has_sufficient_information = completion.choices[0].message.parsed.has_sufficient_information

        return evaluation_reasoning, has_sufficient_information
