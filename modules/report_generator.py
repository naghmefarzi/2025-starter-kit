from pydantic import BaseModel
from llm_client import SafeLLMClient
import json
from typing import List, Dict, Any

class Sentence(BaseModel):
    rationale: str
    sentence_text: str
    citations: list[int]  # Changed to int for mapped IDs

class Report(BaseModel):
    sentences: list[Sentence]

class ReportGenerator(SafeLLMClient):
    def __init__(self):
        super().__init__()
        self.chunk_system_prompt = f'''\
You are a professional fact-checker and media literacy expert. Your task is to generate part of a well-attributed report that provides background and context to help readers assess the trustworthiness of a given news article.

CRITICAL REQUIREMENTS:
1. CITATIONS: Each sentence must have at most 3 references (segment IDs from the provided list). Sentences can have zero citations if they serve as connecting/transitional sentences.
2. GROUNDING: Factual claims and specific information must be cited from the retrieved segments. Skip questions that cannot be answered with available evidence.
3. PRIORITIZATION: Focus on addressing the most important questions first from your assigned subset.
4. THINKING FIRST: For each sentence, provide a clear rationale explaining why this information is important for trustworthiness assessment. if you cant find the answer to the questions, dont reflect them in the sentences.

REPORT FOCUS AREAS:
- Source Investigation (publisher background, author credentials, ownership, funding)
- Claims and Evidence Analysis (verification of assertions, quality of evidence)
- Information Origins (original sources, chain of information)
- Perspective and Balance (missing viewpoints, breadth of sources)
- Context and Timing (broader context, timing motivations)

CITATION GUIDELINES:
- Use only the segment IDs provided in the candidate list (simple integers: 1, 2, 3, etc.)
- Each sentence can have 0-3 citations
- Only cite segments that directly support the sentence content
- If no relevant segments exist for some questions, skip those questions

WRITING STYLE:
- Write clear, concise sentences suitable for general readers
- Maintain objectivity while highlighting trustworthiness concerns
- Focus on actionable insights that help readers make informed judgments

Output format:
{{
    "sentences": [
        {{ "sentence_text": ..., "rationale": ..., "citations": [1, 2]}},
        ...
    ]
}}'''

        self.polish_system_prompt = f'''\
You are a professional fact-checker and media literacy expert. Your task is to polish and refine a fact-checking report by combining multiple partial reports into a single, coherent, and concise final report.

CRITICAL REQUIREMENTS:
1. WORD LIMIT: The final report must not exceed 250 words total across all sentences.
2. COHERENCE: Combine information from partial reports into a logical flow
3. DEDUPLICATION: Remove redundant information while preserving important details
4. PRIORITIZATION: Keep the most critical trustworthiness information
5. CITATIONS: Preserve all valid citations from the partial reports (use simple integer IDs)

POLISH TASKS:
- Combine similar sentences and consolidate information
- Remove redundancy while maintaining key insights
- Ensure logical flow from most to least important points
- Verify all citations are preserved correctly
- Cut less important details if needed to meet word limit

WRITING STYLE:
- Maintain clear, concise sentences suitable for general readers
- Preserve objectivity while highlighting key trustworthiness concerns
- Focus on the most actionable insights for readers

Output should be a JSON with the following format:
{{
    "sentences": [
        {{ "sentence_text": ..., "rationale": ..., "citations": [1, 2, 3]}},
        ...
    ]
}}'''

    def _create_segment_mapping(self, all_llm_selected_segment_ids: set) -> tuple[Dict[int, str], List[Dict]]:
        """
        Create mapping from simple IDs (1, 2, 3, ...) to actual segment IDs.
        Returns: (id_mapping, segments_for_llm)
        """
        id_mapping = {}
        segments_for_llm = []
        
        segment_list = list(all_llm_selected_segment_ids)
        
        for i, actual_segment_id in enumerate(segment_list, 1):
            id_mapping[i] = actual_segment_id
            segments_for_llm.append({
                "segment_id": i,
                "actual_id": actual_segment_id  # Keep for reference but LLM will use segment_id
            })
        
        return id_mapping, segments_for_llm

    def _prepare_segments_with_mapping(self, retrieved_segments: str, all_llm_selected_segment_ids: set) -> tuple[str, Dict[int, str]]:
        """
        Prepare segments with simplified ID mapping for LLM consumption.
        """
        # Parse retrieved segments
        segments_dict = json.loads(retrieved_segments) if isinstance(retrieved_segments, str) else retrieved_segments
        
        # Create ID mapping
        id_mapping, _ = self._create_segment_mapping(all_llm_selected_segment_ids)
        
        # Create new segments dict with mapped IDs
        mapped_segments = {}
        reverse_mapping = {v: k for k, v in id_mapping.items()}
        
        for query, segments in segments_dict.items():
            mapped_segments[query] = []
            for segment in segments:
                # Find the actual segment ID (could be in different fields)
                actual_id = None
                if 'segment_id' in segment:
                    actual_id = segment['segment_id']
                elif 'id' in segment:
                    actual_id = segment['id']
                
                if actual_id and actual_id in reverse_mapping:
                    # Create new segment dict with mapped ID
                    mapped_segment = segment.copy()
                    mapped_segment['segment_id'] = reverse_mapping[actual_id]
                    mapped_segments[query].append(mapped_segment)
        
        return json.dumps(mapped_segments, indent=2), id_mapping

    def _validate_and_map_citations(self, citations: List[int], id_mapping: Dict[int, str]) -> List[str]:
        """
        Validate LLM citations and map them back to actual IDs.
        """
        validated_citations = []
        available_ids = list(id_mapping.keys())
        
        for citation_id in citations:
            if citation_id in id_mapping:
                actual_id = id_mapping[citation_id]
                validated_citations.append(actual_id)
            else:
                print(f"Warning: LLM hallucinated citation ID: {citation_id}")
                # Use fallback: pick the first available ID not already selected
                available_fallback = [aid for aid in available_ids if id_mapping[aid] not in validated_citations]
                if available_fallback:
                    fallback_id = available_fallback[0]
                    actual_id = id_mapping[fallback_id]
                    print(f"Using fallback: {fallback_id} -> {actual_id}")
                    validated_citations.append(actual_id)
        
        return validated_citations

    def chunk_input(self, retrieved_segments: str, questions: str, max_chunk_size: int = 5000) -> List[Dict[str, Any]]:
        """
        Split the input into manageable chunks for processing.
        Each chunk contains a subset of questions and relevant segments.
        """
        # Parse questions
        questions_dict = json.loads(questions) if isinstance(questions, str) else questions
        question_items = list(questions_dict.items())
        
        # Parse retrieved segments
        segments_dict = json.loads(retrieved_segments) if isinstance(retrieved_segments, str) else retrieved_segments
        
        chunks = []
        current_chunk_questions = []
        current_chunk_size = 0
        
        # Estimate base size for segments (we'll include all segments in each chunk for now)
        base_segments_size = len(json.dumps(segments_dict, indent=2))
        
        for i, (q_id, q_data) in enumerate(question_items):
            question_size = len(json.dumps({q_id: q_data}, indent=2))
            
            # If adding this question would exceed the limit, create a chunk
            if current_chunk_size + question_size + base_segments_size > max_chunk_size and current_chunk_questions:
                chunks.append({
                    "questions": dict(current_chunk_questions),
                    "segments": segments_dict,
                    "chunk_id": len(chunks) + 1,
                    "total_chunks": "TBD"  # Will be updated later
                })
                current_chunk_questions = []
                current_chunk_size = 0
            
            current_chunk_questions.append((q_id, q_data))
            current_chunk_size += question_size
        
        # Add the last chunk if it has content
        if current_chunk_questions:
            chunks.append({
                "questions": dict(current_chunk_questions),
                "segments": segments_dict,
                "chunk_id": len(chunks) + 1,
                "total_chunks": "TBD"
            })
        
        # Update total_chunks count
        for chunk in chunks:
            chunk["total_chunks"] = len(chunks)
        
        return chunks

    def generate_chunk_report(self, article: str, chunk: Dict[str, Any], all_llm_selected_segment_ids: set) -> List[tuple]:
        """Generate a report for a single chunk of questions and segments."""
        
        # Prepare segments with ID mapping
        mapped_segments_str, id_mapping = self._prepare_segments_with_mapping(
            json.dumps(chunk["segments"], indent=2), 
            all_llm_selected_segment_ids
        )
        
        # Create available IDs list for LLM reference
        available_ids = list(id_mapping.keys())
        
        user_input = f'''\
Here is the news article to evaluate:
{article}

Here are the retrieved text segments for this chunk:
{mapped_segments_str}

Here are the questions to address in this chunk (chunk {chunk["chunk_id"]} of {chunk["total_chunks"]}):
{json.dumps(chunk["questions"], indent=2)}

Generate a report that addresses as many of the questions as possible using only the information available in the retrieved segments. Focus on the most important questions first. Each sentence in the report should be factual, well-grounded and informative standalone.
If you dont find the answer to a question you dont need to mention that in the report. 

Rules for Citations:
- You MUST select only segment IDs from this list: {available_ids}
- Use simple integer IDs (1, 2, 3, etc.) as shown in the segments above
- Do NOT invent or modify any IDs

Output should be a JSON with the following format:
{{
    "sentences": [
        {{"sentence_text": ..., "rationale": ..., "citations": [1, 2]}},
        ...
    ]
}}
'''   
        messages = [
            {"role": "system", "content": self.chunk_system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        response = self.generate_structured(
            response_model=Report,
            messages=messages,
            temperature=0.1
        )
        
        chunk_report = []
        for sentence in response.sentences:
            # Validate and map citations back to actual IDs
            actual_citations = self._validate_and_map_citations(sentence.citations, id_mapping)
            chunk_report.append((sentence.rationale, sentence.sentence_text, actual_citations))
        
        return chunk_report

    def polish_combined_report(self, partial_reports: List[List[tuple]], all_llm_selected_segment_ids: set) -> List[tuple]:
        """Polish and combine multiple partial reports into a final coherent report."""
        
        # Create ID mapping for polishing step
        id_mapping, _ = self._create_segment_mapping(all_llm_selected_segment_ids)
        reverse_mapping = {v: k for k, v in id_mapping.items()}
        available_ids = list(id_mapping.keys())
        
        # Flatten all partial reports and convert citations to mapped IDs
        all_sentences = []
        for report in partial_reports:
            all_sentences.extend(report)
        
        # Create input for polishing with mapped citations
        sentences_for_polish = []
        for rationale, sentence_text, citations in all_sentences:
            # Convert actual citations back to mapped IDs for LLM
            mapped_citations = []
            for citation in citations:
                if citation in reverse_mapping:
                    mapped_citations.append(reverse_mapping[citation])
            
            sentences_for_polish.append({
                "rationale": rationale,
                "sentence_text": sentence_text,
                "citations": mapped_citations
            })
        
        user_input = f'''\
Here are the partial reports to combine and polish:

{json.dumps({"sentences": sentences_for_polish}, indent=2)}

Your task is to:
1. Combine these partial reports into a single coherent report
2. Remove redundancy and consolidate similar information
3. Ensure the final report does not exceed 250 words
4. Maintain the most important trustworthiness insights
5. Preserve all valid citations

Rules for Citations:
- Only use citation IDs from this list: {available_ids}
- Use simple integer IDs (1, 2, 3, etc.)

Output format:
{{
    "sentences": [
        {{"sentence_text": ..., "rationale": ..., "citations": [1, 2]}},
        ...
    ]
}}
'''
        
        messages = [
            {"role": "system", "content": self.polish_system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        response = self.generate_structured(
            response_model=Report,
            messages=messages,
            temperature=0.1
        )
        
        final_report = []
        total_words = 0
        
        for sentence in response.sentences:
            words_in_sentence = len(sentence.sentence_text.split())
            total_words += words_in_sentence
            
            # Validate and map citations back to actual IDs
            actual_citations = self._validate_and_map_citations(sentence.citations, id_mapping)
            final_report.append((sentence.rationale, sentence.sentence_text, actual_citations))
        
        if total_words > 250:
            print(f"Warning: Final report has {total_words} words, exceeding 250 word limit")
        
        return final_report

    def generate_report(self, article: str, retrieved_segments: str, questions: str, all_llm_selected_segment_ids: set) -> List[tuple]:
        """
        Main method that generates a report using chunked processing and polishing.
        """
        try:
            # Try to generate the report in one go first
            return self._generate_single_report(article, retrieved_segments, questions, all_llm_selected_segment_ids)
        
        except Exception as e:
            print(f"Single report generation failed: {e}")
            print("Falling back to chunked generation...")
            
            # Chunk the input
            chunks = self.chunk_input(retrieved_segments, questions)
            print(f"Created {len(chunks)} chunks for processing")
            
            # Generate partial reports for each chunk
            partial_reports = []
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)}")
                try:
                    chunk_report = self.generate_chunk_report(article, chunk, all_llm_selected_segment_ids)
                    partial_reports.append(chunk_report)
                    print(f"Chunk {i+1} generated {len(chunk_report)} sentences")
                except Exception as chunk_error:
                    print(f"Error processing chunk {i+1}: {chunk_error}")
                    # Continue with other chunks
                    continue
            
            if not partial_reports:
                raise ValueError("All chunks failed to generate reports")
            
            # Polish and combine the partial reports
            print("Polishing and combining partial reports...")
            final_report = self.polish_combined_report(partial_reports, all_llm_selected_segment_ids)
            print(f"Final report has {len(final_report)} sentences")
            
            return final_report

    def _generate_single_report(self, article: str, retrieved_segments: str, questions: str, all_llm_selected_segment_ids: set) -> List[tuple]:
        """Original single-pass report generation (fallback when chunking isn't needed)."""
        
        # Prepare segments with ID mapping
        mapped_segments_str, id_mapping = self._prepare_segments_with_mapping(retrieved_segments, all_llm_selected_segment_ids)
        available_ids = list(id_mapping.keys())
        
        system_prompt = f'''\
You are a professional fact-checker and media literacy expert. Your ultimate task is to generate a well-attributed report that provides background and context to help readers assess the trustworthiness of a given news article. You have previously generated queries, retrieved relevant text segments, and formulated critical questions. Now, based on this information, you must create a comprehensive report that addresses the most important trustworthiness concerns.

CRITICAL REQUIREMENTS:
1. WORD LIMIT: The entire report must not exceed 250 words total across all sentences.
2. CITATIONS: Each sentence must have at most 3 references (segment IDs). Sentences can have zero citations if they serve as connecting/transitional sentences or provide general context that doesn't require grounding.
3. GROUNDING: Factual claims and specific information must be cited from the retrieved segments. Skip questions that cannot be answered with available evidence.
4. STRUCTURE: Generate individual sentences, each with their specific citations (or empty citations list for connecting sentences).
5. PRIORITIZATION: The provided questions are ranked from most to least important. Focus on addressing the most important questions first. It's acceptable to leave less important questions unaddressed if you run out of space within the 250-word limit.
6. THINKING FIRST: For each sentence, you must first provide a clear rationale explaining why this information is important for trustworthiness assessment and how it addresses the critical questions. Think through the evidence before crafting the sentence.

Remember: Quality over quantity. It's better to thoroughly address fewer questions with strong evidence than to superficially cover many topics without proper grounding.
'''

        user_input = f'''\
Here is the news article to evaluate:
{article}

Here are your previously issued queries with their retrieved text segments:
{mapped_segments_str}

Here are the 10 critical questions that should be addressed (in order of importance):
{questions}

Generate a report by addressing as many of the important questions as possible using only the information available in the retrieved segments. Each sentence in the report should be factual, well-grounded and informative standalone.
If you dont find the answer to a question you dont need to mention that in the report.

Rules for Citations:
- You MUST select only segment IDs from this list: {available_ids}
- Use simple integer IDs (1, 2, 3, etc.) as shown in the segments above
- Do NOT invent or modify any IDs

Output format a JSON object with sentences as key and a list of JSON entries as value:
{{
    "sentences": [
        {{"sentence_text": ..., "rationale": ..., "citations": [1, 2]}},
        {{"sentence_text": ..., "rationale": ..., "citations": [3]}},
        {{"sentence_text": ..., "rationale": ..., "citations": [4]}},
        ...
    ]
}}
'''   
        messages = [
            {"role": "system", "content": system_prompt},
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
            
            # Validate and map citations back to actual IDs
            actual_citations = self._validate_and_map_citations(sentence.citations, id_mapping)
            return_report.append((sentence.rationale, sentence.sentence_text, actual_citations))
        
        return return_report