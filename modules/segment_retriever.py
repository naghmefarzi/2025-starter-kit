import os
import json
from openai import AzureOpenAI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
from pyserini.index.lucene import Document
from pyserini.search.lucene import LuceneSearcher


class SelectedSegments(BaseModel):
    segment_ids: list[str]


class SegmentRetriever:
    def __init__(self):
        self.bm25rm3_top_k = 1000
        self.searcher = LuceneSearcher(os.getenv('INDEX_PATH'))
        self.searcher.set_bm25(0.9, 0.4)
        self.searcher.set_rm3(10, 10, 0.5)
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2', cache_folder='../cache')
        self.selector_top_k = 20
        self.selector = AzureOpenAI(
            api_version=os.getenv('OPENAI_API_VERSION'),
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_key=os.getenv('AZURE_OPENAI_API_KEY')
        )

    def search(self, query: str, article: str, exclude_docids: list[str]):
        # Get top hits from BM25+RM3
        hits = self.searcher.search(query, k=self.bm25rm3_top_k)
        results = []
        for i, hit in enumerate(hits):
            if hits[i].docid.split('#')[0] not in exclude_docids:
                segment_json = json.loads(Document(hit.lucene_document).raw())
                results.append({'segment_id': hit.docid,
                                'url': segment_json['url'],
                                'title': segment_json['title'],
                                'headings': segment_json['headings'],
                                'segment': segment_json['segment'],
                                'start_char': segment_json['start_char'],
                                'end_char': segment_json['end_char'],
                                'bm25rm3_score': hit.score, 'bm25rm3_rank': i + 1})

        # Rerank hits
        query_segment_pairs = [(query, f'{result["title"]}\n\n{result["segment"]}') for result in results]
        rerank_scores = self.reranker.predict(query_segment_pairs)
        for i in range(len(results)):
            results[i]['rerank_score'] = float(rerank_scores[i])
        results.sort(key=lambda x: x['rerank_score'], reverse=True)
        results = results[:100]

        # Prepare user input for the LLM selector
        top_segments = []
        top_segment_ids = []
        for result in results[:self.selector_top_k]:
            top_segment_ids.append(result['segment_id'])
            top_segments.append({'segment_id': result['segment_id'], 'title': result['title'], 
                                 'segment_text': result['segment']})

        # Select 3 most relevant segments
        system_prompt = f'''\
You are an expert assistant tasked with selecting the most relevant text segments to answer a given question about a news article, which will serve as the context for the retrieval-augmented generation module to answer that question.

You will be provided with:
1. The news article.
2. A question about the news article.
3. A list of {self.selector_top_k} candidate text segments. These segments have been retrieved as potentially relevant and are pre-ranked (from most relevant to least relevant), but this ranking might not be perfect.

Your goal is to identify and return at most 3 segments from the candidates that are most helpful for answering the question.
The segments you select should be ordered from most relevant to least relevant.
If fewer than 3 segments are relevant, return only those that are. If no segments are relevant, return an empty list.
Focus on the direct relevance of the segment to the question in the context of the provided article. Ideally, the selected segments should come from various sources.'''
        user_input = f'''\
Here is the news article:
{article}

Here is the question:
{query}

Here are the 20 candidate segments, ranked by estimated relevance (highest to lowest):
{json.dumps(top_segments, indent=4)}

Please select the (at most) 3 most relevant segments from the list above to answer the question.'''

        completion = self.selector.beta.chat.completions.parse(
            model='gpt-4.1',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_input}
            ],
            temperature=0,
            presence_penalty=0,
            top_p=1,
            frequency_penalty=0,
            response_format=SelectedSegments,
        )

        llm_selected_segment_ids = completion.choices[0].message.parsed

        llm_selected_results = []
        # Make sure the selected docids are indeed from those provided
        for segment_id in llm_selected_segment_ids.segment_ids:
            if segment_id not in segment_id:
                raise ValueError(f'Selected segment_id: {segment_id} is not in the provided list of candidate segments.')

        for i in range(len(results)):
            results[i]['reranker_rank'] = i + 1
            if results[i]['segment_id'] in llm_selected_segment_ids.segment_ids:
                llm_selected_results.append(results[i])

        return results, llm_selected_results


# Test
# segment_retriever = SegmentRetriever()
# results = segment_retriever.search('What primary evidence exists (e.g., menus, ads, news articles) confirming Sam Panopoulos invented Hawaiian pizza at the Satellite in 1962?', '', ['msmarco_v2.1_doc_04_420132660'])
