from pydantic import BaseModel
from llm_client import SafeLLMClient
import json
from typing import List, Dict, Any

class Sentence(BaseModel):
    rationale: str
    sentence_text: str
    citations: list[str]

class Report(BaseModel):
    sentences: list[Sentence]

class ReportGenerator(SafeLLMClient):
    def __init__(self):
        super().__init__()
        self.chunk_system_prompt = f'''\
You are a professional fact-checker and media literacy expert. Your task is to generate part of a well-attributed report that provides background and context to help readers assess the trustworthiness of a given news article.

CRITICAL REQUIREMENTS:
1. CITATIONS: Each sentence must have at most 3 references (segment docids from MS MARCO V2.1). Sentences can have zero citations if they serve as connecting/transitional sentences.
2. GROUNDING: Factual claims and specific information must be cited from the retrieved segments. Skip questions that cannot be answered with available evidence.
3. PRIORITIZATION: Focus on addressing the most important questions first from your assigned subset.
4. THINKING FIRST: For each sentence, provide a clear rationale explaining why this information is important for trustworthiness assessment.

REPORT FOCUS AREAS:
- Source Investigation (publisher background, author credentials, ownership, funding)
- Claims and Evidence Analysis (verification of assertions, quality of evidence)
- Information Origins (original sources, chain of information)
- Perspective and Balance (missing viewpoints, breadth of sources)
- Context and Timing (broader context, timing motivations)

CITATION GUIDELINES:
- Use exact segment ids as provided, starting with 'msmarco_v2.1_doc_' and containing '#'
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
        {{ "sentence_text": ..., "rationale": ..., "citations": ...}},
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
5. CITATIONS: Preserve all valid citations from the partial reports

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
    
        {{ "sentence_text": ..., "rationale": ..., "citations": ...}},
        ...
    ]

}}'''

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
        
        user_input = f'''\
Here is the news article to evaluate:
{article}

Here are the retrieved text segments for this chunk:
{json.dumps(chunk["segments"], indent=2)}

Here are the questions to address in this chunk (chunk {chunk["chunk_id"]} of {chunk["total_chunks"]}):
{json.dumps(chunk["questions"], indent=2)}

Generate a report that addresses as many of the questions as possible using only the information available in the retrieved segments. Focus on the most important questions first.

Rules for Citations:
- You MUST select only segment IDs exactly as they appear in the candidate list.
- Each ID starts with 'msmarco_v2.1_doc_' followed by a document number, '#', and a suffix.
- Do NOT invent, modify, simplify, or guess any part of the ID.
Pick citations accordingly from: {all_llm_selected_segment_ids}

Output should be a JSON with the following format:
{{

    "sentences": [
        {{"sentence_text": ..., "rationale": ..., "citations": ...}},
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
            for citation in sentence.citations:
                if citation not in all_llm_selected_segment_ids:
                    raise ValueError(f'[Chunk Report Generator] Citation "{citation}" is not in the list of all LLM-selected segment ids.')
            chunk_report.append((sentence.rationale, sentence.sentence_text, sentence.citations))
        
        return chunk_report

    def polish_combined_report(self, partial_reports: List[List[tuple]], all_llm_selected_segment_ids: set) -> List[tuple]:
        """Polish and combine multiple partial reports into a final coherent report."""
        
        # Flatten all partial reports
        all_sentences = []
        for report in partial_reports:
            all_sentences.extend(report)
        
        # Create input for polishing
        sentences_for_polish = []
        for rationale, sentence_text, citations in all_sentences:
            sentences_for_polish.append({
                "rationale": rationale,
                "sentence_text": sentence_text,
                "citations": citations
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
- Preserve existing citations exactly as they appear
- Only use citations from: {all_llm_selected_segment_ids}

Output format:
{{
    "sentences": [
        {{ "sentence_text": ..., "rationale": ..., "citations": ...}},
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
            
            for citation in sentence.citations:
                if citation not in all_llm_selected_segment_ids:
                    raise ValueError(f'[Polish Report Generator] Citation "{citation}" is not in the list of all LLM-selected segment ids.')
            
            final_report.append((sentence.rationale, sentence.sentence_text, sentence.citations))
        
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
        system_prompt = f'''\
You are a professional fact-checker and media literacy expert. Your ultimate task is to generate a well-attributed report that provides background and context to help readers assess the trustworthiness of a given news article. You have previously generated queries, retrieved relevant text segments, and formulated critical questions. Now, based on this information, you must create a comprehensive report that addresses the most important trustworthiness concerns.

CRITICAL REQUIREMENTS:
1. WORD LIMIT: The entire report must not exceed 250 words total across all sentences.
2. CITATIONS: Each sentence must have at most 3 references (segment docids from MS MARCO V2.1). Sentences can have zero citations if they serve as connecting/transitional sentences or provide general context that doesn't require grounding.
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

Output format a JSON object with sentences as key and a list of JSON entries as value:
{{
    "sentences": [

        {{"sentence_text": ..., "rationale": ..., "citations": ...}},
        {{"sentence_text": ..., "rationale": ..., "citations": ...}},
        {{"sentence_text": ..., "rationale": ..., "citations": ...}},
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
            for citation in sentence.citations:
                if citation not in all_llm_selected_segment_ids:
                    raise ValueError(f'[Report Generator] Citation "{citation}" is not in the list of all LLM-selected segment ids.')
            return_report.append((sentence.rationale, sentence.sentence_text, sentence.citations))
        
        return return_report