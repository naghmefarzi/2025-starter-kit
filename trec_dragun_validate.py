#!/usr/bin/env python3
import sys
import json


all_50_doc_ids = {
    "msmarco_v2.1_doc_04_420132660", "msmarco_v2.1_doc_06_1440134319", "msmarco_v2.1_doc_08_300872161",
    "msmarco_v2.1_doc_15_116067546", "msmarco_v2.1_doc_21_861891150", "msmarco_v2.1_doc_22_1648697797",
    "msmarco_v2.1_doc_25_481628070", "msmarco_v2.1_doc_25_501708725", "msmarco_v2.1_doc_25_502424913",
    "msmarco_v2.1_doc_34_2529216", "msmarco_v2.1_doc_34_7751734", "msmarco_v2.1_doc_35_441326441",
    "msmarco_v2.1_doc_35_1300032609", "msmarco_v2.1_doc_38_227081897", "msmarco_v2.1_doc_39_1014551192",
    "msmarco_v2.1_doc_39_1165221603", "msmarco_v2.1_doc_41_1960853470", "msmarco_v2.1_doc_42_654618974",
    "msmarco_v2.1_doc_47_1430382251", "msmarco_v2.1_doc_48_273997000", "msmarco_v2.1_doc_48_515083157",
    "msmarco_v2.1_doc_48_515287844", "msmarco_v2.1_doc_48_730773621", "msmarco_v2.1_doc_48_1289995263",
    "msmarco_v2.1_doc_52_1666095905", "msmarco_v2.1_doc_54_304836636", "msmarco_v2.1_doc_54_1547893878",
    "msmarco_v2.1_doc_55_248024230", "msmarco_v2.1_doc_57_933363597", "msmarco_v2.1_doc_58_748655897"
}

def validate_task1(file_path):
    errors = []
    line_num = 0
    team_ids = set()
    run_ids = set()
    topics_ranks = {}  # track ranks seen for each topic
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for raw_line in f:
                line_num += 1
                # Strip newline at the end (and any extra whitespace)
                line = raw_line.rstrip('\n')
                if line == '':
                    continue  # skip empty line (if any)
                # Split fields by tab
                fields = line.split('\t')
                if len(fields) != 5:
                    errors.append(f"Line {line_num}: Expected 5 tab-separated fields, but got {len(fields)} fields.")
                    continue
                topic_id, team_id, run_id, rank_str, question = fields
                # Track team_id and run_id for consistency checks
                team_ids.add(team_id)
                run_ids.add(run_id)
                # Validate rank field is an integer 1-10
                if rank_str.isdigit():
                    rank = int(rank_str)
                    if rank < 1 or rank > 10:
                        errors.append(f"Line {line_num}: Rank {rank} is out of the valid range 1-10.")
                else:
                    errors.append(f"Line {line_num}: Rank '{rank_str}' is not a valid integer.")
                # Validate question field
                if '\t' in question:
                    errors.append(f"Line {line_num}: Question contains a tab character, which is not allowed.")
                if '\n' in question:
                    errors.append(f"Line {line_num}: Question contains a newline character, which is not allowed.")
                if len(question) > 300:
                    errors.append(f"Line {line_num}: Question is {len(question)} characters long, exceeding the 300-character limit.")
                # Record the rank for this topic (to check completeness of 1-10)
                if topic_id not in topics_ranks:
                    topics_ranks[topic_id] = set()
                if rank_str.isdigit():
                    topics_ranks[topic_id].add(int(rank_str))
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return False
    except Exception as e:
        print(f"Error: Could not read file: {e}")
        return False

    # Check that all lines had consistent team_id and run_id (one run per file)
    if len(team_ids) > 1:
        errors.append(f"Inconsistent team_id: multiple team IDs found ({', '.join(team_ids)}). All lines in one submission must have the same team_id.")
    if len(run_ids) > 1:
        errors.append(f"Inconsistent run_id: multiple run IDs found ({', '.join(run_ids)}). All lines in one submission must have the same run_id.")
    # Check that each topic has ranks 1 through 10 exactly
    for topic_id, ranks in topics_ranks.items():
        # Missing or extra ranks?
        expected_ranks = set(range(1, 11))
        if ranks != expected_ranks:
            missing = sorted(expected_ranks - ranks)
            extra = sorted(ranks - expected_ranks)
            if missing:
                errors.append(f"Topic {topic_id}: Missing rank(s) {missing}. Each topic should have ranks 1-10.")
            if extra:
                errors.append(f"Topic {topic_id}: Found invalid rank(s) {extra} (ranks must be 1-10).")
        # Check if exactly 10 questions for the topic
        if len(ranks) != 10:
            errors.append(f"Topic {topic_id}: Expected 10 questions (ranks 1-10), but found {len(ranks)} unique rank(s).")
    # Check if all topic_ids are included
    if set(topics_ranks.keys()) != all_50_doc_ids:
        errors.append(f"The following topics are missing: {all_50_doc_ids.difference(set(topics_ranks.keys()))}.")

    # Output results
    if errors:
        print("Submission is INVALID. Found the following issues:")
        for err in errors:
            print("- " + err)
        return False
    else:
        print("Submission is VALID: All checks passed.")
        return True

def validate_task2(file_path):
    errors = []
    line_num = 0
    # Variables to enforce consistency across lines
    team_id_constant = None
    run_id_constant = None
    type_constant = None
    starter_kit_constant = None
    seen_topics = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for raw_line in f:
                line_num += 1
                line = raw_line.strip()
                if line == '':
                    continue  # skip empty lines
                # Parse JSON for each line
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: Invalid JSON – {e}")
                    continue
                # Top-level keys check
                if not isinstance(obj, dict) or "metadata" not in obj or "responses" not in obj:
                    errors.append(f"Line {line_num}: Each line must be a JSON object with 'metadata' and 'responses'.")
                    continue
                metadata = obj["metadata"]
                responses = obj["responses"]
                # Metadata object checks
                if not isinstance(metadata, dict):
                    errors.append(f"Line {line_num}: 'metadata' should be a JSON object.")
                    continue
                # Required metadata fields
                required_meta = {"team_id", "run_id", "topic_id", "type", "use_starter_kit"}
                meta_keys = set(metadata.keys())
                missing_meta = required_meta - meta_keys
                extra_meta = meta_keys - required_meta
                if missing_meta:
                    errors.append(f"Line {line_num}: Missing metadata field(s): {', '.join(missing_meta)}.")
                if extra_meta:
                    errors.append(f"Line {line_num}: Unexpected metadata field(s): {', '.join(extra_meta)}.")
                if missing_meta:
                    # skip further checks for this line if critical fields are missing
                    continue
                # Extract metadata fields
                team_id = metadata["team_id"]
                run_id = metadata["run_id"]
                topic_id = metadata["topic_id"]
                run_type = metadata["type"]
                starter_flag = metadata["use_starter_kit"]
                # Check data types and allowed values for metadata fields
                if not isinstance(team_id, str):
                    errors.append(f"Line {line_num}: team_id must be a string.")
                if not isinstance(run_id, str):
                    errors.append(f"Line {line_num}: run_id must be a string.")
                if not isinstance(topic_id, str):
                    errors.append(f"Line {line_num}: topic_id must be a string.")
                if not isinstance(run_type, str):
                    errors.append(f"Line {line_num}: type must be a string ('automatic' or 'manual').")
                elif run_type not in {"automatic", "manual"}:
                    errors.append(f"Line {line_num}: type has invalid value '{run_type}' (must be 'automatic' or 'manual').")
                if not isinstance(starter_flag, int):
                    errors.append(f"Line {line_num}: use_starter_kit must be an integer (0 or 1).")
                elif starter_flag not in {0, 1}:
                    errors.append(f"Line {line_num}: use_starter_kit has invalid value {starter_flag} (must be 0 or 1).")
                # Consistency across lines (team_id, run_id, type, use_starter_kit should remain constant)
                if team_id_constant is None:
                    team_id_constant = team_id
                elif team_id_constant != team_id:
                    errors.append(f"Line {line_num}: Inconsistent team_id '{team_id}' (expected '{team_id_constant}' from previous lines).")
                if run_id_constant is None:
                    run_id_constant = run_id
                elif run_id_constant != run_id:
                    errors.append(f"Line {line_num}: Inconsistent run_id '{run_id}' (expected '{run_id_constant}').")
                if type_constant is None:
                    type_constant = run_type
                elif type_constant != run_type:
                    errors.append(f"Line {line_num}: Inconsistent type '{run_type}' (expected '{type_constant}').")
                if starter_kit_constant is None:
                    starter_kit_constant = starter_flag
                elif starter_kit_constant != starter_flag:
                    errors.append(f"Line {line_num}: Inconsistent use_starter_kit value {starter_flag} (expected {starter_kit_constant}).")
                # Topic uniqueness
                if topic_id in seen_topics:
                    errors.append(f"Line {line_num}: Duplicate topic_id '{topic_id}' – each topic should appear only once in the file.")
                else:
                    seen_topics.add(topic_id)
                # Responses field checks
                if not isinstance(responses, list):
                    errors.append(f"Line {line_num}: 'responses' should be a list of response objects.")
                    continue
                total_word_count = 0
                for i, resp in enumerate(responses, start=1):
                    if not isinstance(resp, dict):
                        errors.append(f"Line {line_num}: Response #{i} is not a JSON object.")
                        continue
                    if "text" not in resp or "citations" not in resp:
                        errors.append(f"Line {line_num}: Response #{i} is missing 'text' or 'citations' field.")
                        continue
                    text = resp["text"]
                    citations = resp["citations"]
                    # Check types of text and citations
                    if not isinstance(text, str):
                        errors.append(f"Line {line_num}: 'text' in response #{i} should be a string.")
                    if not isinstance(citations, list):
                        errors.append(f"Line {line_num}: 'citations' in response #{i} should be a list.")
                        # if citations is not a list, skip further checks for this response
                        continue
                    if len(citations) > 3:
                        errors.append(f"Line {line_num}: Response #{i} has {len(citations)} citations (maximum 3 allowed).")
                    for cit in citations:
                        if not isinstance(cit, str):
                            errors.append(f"Line {line_num}: Citation in response #{i} is not a string (value: {cit}).")
                    # Count words in text for the 250-word limit
                    if isinstance(text, str):
                        word_count = len(text.split())
                        total_word_count += word_count
                # After iterating responses, check the total word count for this topic
                if total_word_count > 250:
                    errors.append(f"Line {line_num}: Total word count {total_word_count} exceeds 250-word limit for topic '{topic_id}'.")
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return False
    except Exception as e:
        print(f"Error: Could not read file: {e}")
        return False

    if seen_topics != all_50_doc_ids:
        errors.append(f"The following topics are missing: {all_50_doc_ids.difference(seen_topics)}.")
    # Output results
    if errors:
        print("Submission is INVALID. Found the following issues:")
        for err in errors:
            print("- " + err)
        return False
    else:
        print("Submission is VALID: All checks passed.")
        return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python trec_dragun_validate.py <submission_file> <task1|task2>")
        sys.exit(1)
    file_path = sys.argv[1]
    task = sys.argv[2].lower()
    if task == "task1":
        valid = validate_task1(file_path)
    elif task == "task2":
        valid = validate_task2(file_path)
    else:
        print("Error: Unknown task identifier. Use 'task1' or 'task2'.")
        sys.exit(1)
    # Exit with code 0 if valid, or 1 if invalid
    if not valid:
        sys.exit(1)
