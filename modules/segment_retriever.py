import os
import json
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
from pyserini.index.lucene import Document
from pyserini.search.lucene import LuceneSearcher
from llm_client import llm_client 


class SelectedSegments(BaseModel):
    segment_ids: list[str]


class SegmentRetriever:
    def __init__(self):
        self.bm25rm3_top_k = 1000
        self.searcher = LuceneSearcher(os.getenv('INDEX_PATH'))
        self.searcher.set_bm25(0.9, 0.4)
        self.searcher.set_rm3(10, 10, 0.5)
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2', cache_folder='../cache')
        self.selector_top_k = 10

    def search(self, query: str, article: str, exclude_docids: list[str]):
        hits = self.searcher.search(query, k=self.bm25rm3_top_k)
        results = []
        for i, hit in enumerate(hits):
            if hits[i].docid.split('#')[0] not in exclude_docids:
                segment_json = json.loads(Document(hit.lucene_document).raw())
                # print(f"segment_json:{segment_json}")

                results.append({'segment_id': hit.docid,
                                'url': segment_json['url'],
                                'title': segment_json['title'],
                                'headings': segment_json['headings'],
                                'segment': segment_json['contents'],
                                'start_char': segment_json['start_char'],
                                'end_char': segment_json['end_char'],
                                'bm25rm3_score': hit.score, 'bm25rm3_rank': i + 1})
        query_segment_pairs = [(query, f'{result["title"]}\n\n{result["segment"]}') for result in results]
        rerank_scores = self.reranker.predict(query_segment_pairs)
        for i in range(len(results)):
            results[i]['rerank_score'] = float(rerank_scores[i])
        results.sort(key=lambda x: x['rerank_score'], reverse=True)
        results = results[:100]

        top_segments = []
        top_segment_ids = []
        for result in results[:self.selector_top_k]:
            # print(f"result:{result}")

            top_segment_ids.append(result['segment_id'])
            top_segments.append({'segment_id': result['segment_id'], 'title': result['title'], 
                                 'segment_text': result['segment']})

        system_prompt = f'''\
You are an expert assistant tasked with selecting the most relevant text segments to answer a given query about a news article, which will serve as the context for the retrieval-augmented generation module to answer that query.

You will be provided with:
1. The news article.
2. A query about the news article.
3. A list of {self.selector_top_k} candidate text segments. These segments have been retrieved as potentially relevant and are pre-ranked (from most relevant to least relevant), but this ranking might not be perfect.

Your goal is to identify and return at most 3 segments from the candidates that are most helpful for answering the query.
The segments you select should be ordered from most relevant to least relevant.
If fewer than 3 segments are relevant, return only those that are. If no segments are relevant, return an empty list.
Focus on the direct relevance of the segment to the query in the context of the provided article. Ideally, the selected segments should come from various sources.'''
        user_input = f'''\
Here is the news article:
{article}

Here is the query:
{query}

Here are the 20 candidate segments, ranked by estimated relevance (highest to lowest):
{json.dumps(top_segments, indent=4)}

Please select the (at most) 3 most relevant segments from the list above to answer the query.
Output should be ONLY a list of (at most) 3 most relevant segments starting with 'msmarco_v2.1_doc_' and has '#' in it:
[segment_id, segment_id, segment_id]'''

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        completion = llm_client.generate_structured(
            response_model=SelectedSegments,
            messages=messages,
            temperature=0,
            # presence_penalty=0,
            top_p=1,
            # frequency_penalty=0,
        )

        llm_selected_segment_ids = completion

        # Enhanced validation with fallbacks
        provided_ids_set = set(top_segment_ids)
        validated_segment_ids = []

        for segment_id in llm_selected_segment_ids.segment_ids:
            if segment_id in provided_ids_set:
                validated_segment_ids.append(segment_id)
            else:
                print(f"Warning: LLM hallucinated segment_id: {segment_id}")
                # Use fallback: pick the first available ID not already selected
                available_ids = [vid for vid in top_segment_ids if vid not in validated_segment_ids]
                if available_ids:
                    fallback_id = available_ids[0]
                    print(f"Using fallback: {fallback_id}")
                    validated_segment_ids.append(fallback_id)

        # Ensure we have at least one segment
        if not validated_segment_ids:
            print("Warning: No valid segments selected, using top segment by rerank score")
            validated_segment_ids = [top_segment_ids[0]]

        # Build final results using validated IDs
        llm_selected_results = []
        for i, result in enumerate(results):
            results[i]['reranker_rank'] = i + 1
            if results[i]['segment_id'] in validated_segment_ids:
                llm_selected_results.append(results[i])

        return results, llm_selected_results
