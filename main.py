import json
from tqdm import tqdm
from dotenv import load_dotenv
from modules.query_generator import QueryGenerator
from modules.segment_retriever import SegmentRetriever
from modules.information_evaluator import InformationEvaluator
from modules.question_generator import QuestionGenerator
from modules.report_generator import ReportGenerator


load_dotenv()


def main():
    max_query_iterations = 5

    # Read topics
    articles = []
    with open('./data/trec-2025-dragun-topics.jsonl', 'r', encoding='utf-8') as f_in:
        for line in f_in:
            articles.append(json.loads(line.strip()))
    
    # Initialize modules
    query_generator = QueryGenerator()
    segment_retriever = SegmentRetriever()
    information_evaluator = InformationEvaluator()
    question_generator = QuestionGenerator()
    report_generator = ReportGenerator()

    # Process topics
    tracking_data = {}
    for article in tqdm(articles[4:], desc="Processing articles"):
        article_id = article.get('docid')
        article_json_str = json.dumps({k: v for k, v in article.items() if k != 'docid'}, indent=4)
        
        per_article_tracking_data = {}
        iteration_counter = 1
        query_retrieval_history = {}
        information_evaluation_reasoning = ''

        while iteration_counter < max_query_iterations + 1:
            iteration_data = {}  # One iteration of query generation, segment retrieval, and information evaluation
            # 1. Generate search queries
            if iteration_counter == 1:
                queries = query_generator.generate_query(article_json_str)
            else:
                queries = query_generator.generate_query(article_json_str,  json.dumps(query_retrieval_history, indent=4), 
                                                         information_evaluation_reasoning)

            # 2. Retrieve segments
            for i, (query, rationale) in enumerate(queries):
                segments, llm_selected_segments = segment_retriever.search(query, article_json_str, [article_id])  # Exclude the target article itself from the search
                iteration_data[f'query_{i+1}'] = {
                    'query': query,
                    'rationale': rationale,
                    'reranked_segments': segments,
                    'llm_selected_segments': llm_selected_segments
                }
                simplified_llm_selected_segments = [{'segment_id': segment['segment_id'], 'url': segment['url'], 'title': segment['title'],
                                                     'segment_text': segment['segment']} for segment in llm_selected_segments]
                query_retrieval_history[f'query_{1+len(query_retrieval_history)}'] = {
                    'query': query,
                    'rationale': rationale,
                    'retrieved_segments': simplified_llm_selected_segments
                }

            # 3. Evaluate if sufficient information is retrieved
            evaluation_reasoning, has_sufficient_information = information_evaluator.evaluate(article_json_str, json.dumps(query_retrieval_history, indent=4))
            iteration_data['evaluation'] = {'has_sufficient_information': has_sufficient_information, 'evaluation_reasoning': evaluation_reasoning}
            information_evaluation_reasoning = evaluation_reasoning

            # 4. If so, finish
            per_article_tracking_data[f'iteration_{iteration_counter}'] = iteration_data
            iteration_counter += 1
            if has_sufficient_information:
                break
        
        # Prepare input for question generation and report generation
        reorganized_query_retrieval_data = {}
        all_llm_selected_segment_ids = set()  # To verify if citations come from selected segments
        query_counter = 1
        for iteration_num in range(1, iteration_counter):
            iteration_key = f'iteration_{iteration_num}'
            if iteration_key in per_article_tracking_data:
                iteration_data = per_article_tracking_data[iteration_key]
                query_keys = [k for k in iteration_data.keys() if k.startswith('query_')]
                for query_key in sorted(query_keys, key=lambda x: int(x.split('_')[1])):
                    query_data = iteration_data[query_key]
                    segments = []
                    for segment in query_data['llm_selected_segments']:
                        segments.append({'segment_id': segment['segment_id'], 'url': segment['url'], 
                                         'title': segment['title'], 'segment_text': segment['segment']})
                        all_llm_selected_segment_ids.add(segment['segment_id'])

                    reorganized_query_retrieval_data[f'query_{query_counter}'] = {
                        'query': query_data['query'],
                        'rationale': query_data['rationale'],
                        'llm_selected_segments': segments
                    }
                    query_counter += 1
        
        # 5. Generate questions for Task 1
        generated_questions = question_generator.generate_questions(article_json_str, json.dumps(reorganized_query_retrieval_data, indent=4))
        questions_json = {}
        for i, question in enumerate(generated_questions):
            questions_json[f'question_{i+1}'] = {'question': question[1], 'rationale': question[0]}
        per_article_tracking_data['question_generation'] = questions_json

        # 6. Generate reports for Task 2
        generated_report = report_generator.generate_report(article_json_str, json.dumps(reorganized_query_retrieval_data, indent=4),
                                                            json.dumps(questions_json, indent=4), all_llm_selected_segment_ids)
        report_json = {}
        for i, sentence in enumerate(generated_report):
            report_json[f'sentence_{i+1}'] = {'sentence': sentence[1], 'rationale': sentence[0], 'citations': sentence[2]}
        per_article_tracking_data['report_generation'] = report_json

        tracking_data[article_id] = per_article_tracking_data
        with open('tracking_data.json', 'w', encoding='utf-8') as f:
            json.dump(tracking_data, f, indent=4, ensure_ascii=False)
        

if __name__ == "__main__":
    main() 
    