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
        provided_ids_set = set(top_segment_ids)

        for result in results[:self.selector_top_k]:
            # print(f"result:{result}")

            top_segment_ids.append(result['segment_id'])
            top_segments.append({'segment_id': result['segment_id'], 'title': result['title'], 
                                 'segment_text': result['segment']})

        system_prompt = f'''\
You are an expert assistant tasked with selecting the most relevant segment IDs from a provided list of candidate text segments to answer a query about a news article. These segment IDs will be used as context for a retrieval-augmented generation module.

You will be provided with:
1. The news article.
2. A query about the news article.
3. A list of {self.selector_top_k} candidate text segments, each identified by a unique segment ID. These are pre-ranked (highest to lowest relevance), but the ranking may not be perfect.

Your task:
- Select at most 3 segment IDs from the provided candidate list that are most relevant to answering the query.
- Order the selected IDs from most relevant to least relevant.
- If fewer than 3 IDs are relevant, return only those. If none are relevant, return an empty list.
- Focus on the direct relevance of the segment content to the query within the article's context.
- Ideally, select IDs from different sources if equally relevant.

Rules:
- You MUST select only segment IDs exactly as they appear in the candidate list.
- Each ID starts with 'msmarco_v2.1_doc_' followed by a document number, '#', and a suffix (e.g., a number or number_another_number).
- Do NOT invent, modify, simplify, or guess any part of the ID, including the suffix.
- Do NOT generate IDs not present in the candidate list, such as changing '#17_1908612056' to '#17'.

'''
        user_input = f'''\
Here is the news article:
{article}

Here is the query:
{query}

Here are the 10 candidate segments with their segment IDs, ranked by estimated relevance (highest to lowest):
{json.dumps(top_segments, indent=4)}

Please select at most 3 segment IDs from the candidate list above that are most relevant to answering the query.
Output ONLY a list of up to 3 segment IDs, copied exactly as they appear in the candidate list (starting with 'msmarco_v2.1_doc_' and containing '#', e.g., 'msmarco_v2.1_doc_28_903296016#17_1908612056').
Do NOT invent, modify, or simplify any IDs.
Do NOT include explanations, text content, or any other output.


Output format:
[segment_id, segment_id, segment_id]

Pick citations accordingly from: {provided_ids_set}
'''

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        completion = llm_client.generate_structured(
            response_model=SelectedSegments,
            messages=messages,
            temperature=0.2,
            top_p=0.1,
        )

        llm_selected_segment_ids = completion

        # Enhanced validation with fallbacks
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
