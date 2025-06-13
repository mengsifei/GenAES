import re
import pandas as pd
from typing import Dict, List, Union, Tuple
from cefrpy import CEFRSpaCyAnalyzer, CEFRLevel
import spacy

def extract_feedback_with_clean_quotes(feedback_str: str) -> Dict[str, Union[str, List[str]]]:
    section_map = {
        "Task Response feedback": "TR_feedback",
        "Coherence and Cohesion feedback": "CC_feedback",
        "Lexical Resource feedback": "LR_feedback",
        "Grammatical Range and Accuracy feedback": "GRA_feedback",
        "Is off topic": "is_off_topic",
        "Word limit satisfied": "word_limit",
        "Corrected essay": "Corrected_essay"
    }
    
    result = {v: None for v in section_map.values()}
    quote_results = {f"{v}_quotes": [] for v in section_map.values() if v.endswith('_feedback')}

    section_pattern = r'"(?P<header>(?:[^"]|\\")+)"\s*:\s*"(?P<content>(?:[^"]|\\")*)"'
    
    for match in re.finditer(section_pattern, feedback_str):
        header = match.group('header')
        content = match.group('content').replace('\\"', '"')
        
        if header in section_map:
            key = section_map[header]
            result[key] = content
            
            # Extract and clean quoted phrases for feedback sections
            if key.endswith('_feedback'):
                quotes = re.findall(r"'(.*?)'", content)
                clean_quotes = []
                for quote in quotes:
                    # Remove trailing punctuation
                    cleaned = re.sub(r'[.,;:!?]+$', '', quote.strip())
                    if cleaned:  # Only keep non-empty strings
                        clean_quotes.append(cleaned)
                quote_results[f"{key}_quotes"] = clean_quotes
    
    # Handle special cases
    for orig, new in [("Is off topic", "is_off_topic"), 
                     ("Word limit satisfied", "word_limit")]:
        if result[new] is None:
            match = re.search(rf'{orig}\s*:\s*"([^"]+)"', feedback_str)
            if match:
                result[new] = match.group(1)
    
    # Handle corrected essay (multi-line)
    if result["Corrected_essay"] is None:
        essay_match = re.search(
            r'"Corrected essay"\s*:\s*"(.*?)"(?=\s*[,\]}]|$)',
            feedback_str, 
            re.DOTALL
        )
        if essay_match:
            result["Corrected_essay"] = essay_match.group(1).replace('\\"', '"')
    
    return pd.Series({**result, **quote_results})


def extract_feedback_keys_values(feedback_str):
    try:
        # Map the feedback sections to standardized column names
        section_map = {
            '"Task Response feedback"': 'TR_feedback',
            '"Coherence and Cohesion feedback"': 'CC_feedback',
            '"Lexical Resource feedback"': 'LR_feedback',
            '"Grammatical Range and Accuracy feedback"': 'GRA_feedback',
            '"Corrected essay"': 'Corrected_essay'
        }
        result = {v: None for v in section_map.values()}  # Initialize with None
        for original_section, new_key in section_map.items():
            # Find the start of the section
            start = feedback_str.find(original_section)
            if start == -1:
                continue  
            # Find the end of this section (either next section or end of string)
            end = len(feedback_str)
            for other_section in section_map:
                if other_section != original_section:
                    other_start = feedback_str.find(other_section, start + 1)
                    if other_start != -1 and other_start < end:
                        end = other_start
            section_content = feedback_str[start:end].strip()
            key_end = section_content.find(':')
            if key_end == -1:
                continue
            value = section_content[key_end+1:].strip().strip(' ,')
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            result[new_key] = value
        return pd.Series(result)  # Return as Series for DataFrame expansion
    except Exception as e:
        print(f"Error processing feedback: {e}")
        return pd.Series({k: None for k in section_map.values()})
    

def create_train_input(row):
    feedback_parts = [
        f"Task Response Feedback: {row['TR_feedback']}",
        f"Coherence and Cohesion Feedback: {row['CC_feedback']}",
        f"Lexical Resource Feedback: {row['LR_feedback']}",
        f"Grammatical Range and Accuracy Feedback: {row['GRA_feedback']}", 
        f"The essay has {row['word_count']} words and {row['paragraph_count']} paragraphs.",
        f"The CEFR statistics of this essay: {row['cefr_stat']}"
    ]
    feedback_str = "\n".join(feedback_parts)
    
    return (
        "{{TOPIC}}\n" + row['topic'] + 
        "\n{{ESSAY}}\n" + row['essay'] + 
        "\n{{CORRECTED_ESSAY}}\n" + row['Corrected_essay'] + 
        "\n{{FEEDBACK}}\n" + feedback_str
    )

column_mapping = {
    'Task Response': 'TR_score',
    'Coherence and Cohesion': 'CC_score',
    'Lexical Resource': 'LR_score',
    'Grammatical Range and Accuracy': 'GRA_score'
}


nlp = spacy.load("en_core_web_sm")

def get_cefr_stats(text):
    if not isinstance(text, str) or not text.strip():
        return {f'{level}_words': 0 for level in ['A1','A2','B1','B2','C1','C2','unknown']} | {'total_words': 0}
    
    ABBREVIATION_MAPPING = {
        "'m": "am",
        "'s": "is",
        "'re": "are",
        "'ve": "have",
        "'d": "had",
        "n't": "not",
        "'ll": "will"
    }

    ENTITY_TYPES_TO_SKIP_CEFR = {
        'QUANTITY', 'MONEY', 'LANGUAGE', 'LAW',
        'WORK_OF_ART', 'PRODUCT', 'GPE',
        'ORG', 'FAC', 'PERSON'
    }

    def get_word_level_count_statistic(level_tokens: List[Tuple[str, str, bool, float, int, int]]) -> dict:
        """Safe counting of CEFR levels with error handling"""
        difficulty_levels_count = [0] * 6
        unknown_count = 0
        result = {}
        
        for token in level_tokens:
            try:
                level = token[3]
                if level is None:
                    unknown_count += 1
                    continue
                    
                # Safely handle level conversion
                try:
                    level_round = round(float(level))
                    if 1 <= level_round <= 6:
                        difficulty_levels_count[level_round - 1] += 1
                    else:
                        unknown_count += 1
                except (ValueError, TypeError):
                    unknown_count += 1
                    
            except Exception as e:
                print(f"Error processing token: {e}")
                unknown_count += 1
        
        # Convert to CEFR level names
        for i in range(1, 7):
            result[f'{CEFRLevel(i)}_words'] = difficulty_levels_count[i - 1]
        result['unknown_words'] = unknown_count
        result['total_words'] = sum(difficulty_levels_count) + unknown_count
        
        # Calculate percentages
        if result['total_words'] > 0:
            for i in range(1, 7):
                result[f'{CEFRLevel(i)}_pct'] = (difficulty_levels_count[i - 1] / result['total_words']) * 100
            result['unknown_pct'] = (unknown_count / result['total_words']) * 100
        else:
            for i in range(1, 7):
                result[f'{CEFRLevel(i)}_pct'] = 0.0
            result['unknown_pct'] = 0.0
            
        return result

    try:
        # Handle encoding errors by cleaning the text first
        clean_text = text.encode('ascii', errors='ignore').decode('ascii')
        doc = nlp(clean_text)
        text_analyzer = CEFRSpaCyAnalyzer(
            entity_types_to_skip=ENTITY_TYPES_TO_SKIP_CEFR,
            abbreviation_mapping=ABBREVIATION_MAPPING
        )
        tokens = text_analyzer.analize_doc(doc)
        ans = str(get_word_level_count_statistic(tokens))
        return ans
        
    except Exception as e:
        print(f"Error analyzing text: {e}")
        return str({f'{level}_words': 0 for level in ['A1','A2','B1','B2','C1','C2','unknown']} | {'total_words': 0})
    
    

def replace_single_newlines(text):
    # Replace \n not preceded by \n or not followed by \n
    return re.sub(r'(?<!\n)\n(?!\n)', '\\\\n\\\\n', text)
# feedback_data = extract_feedback_with_clean_quotes(feedback_text)
# print(feedback_data["LR_feedback_quotes"])  