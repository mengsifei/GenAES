feedback_prompt_1 = """##{{TASK}}

You’re required to do four tasks. 

First of all, answer, if the essay is off-topic. If the essay and the given prompt are about different concepts, for example, the essay is about crime, and the prompt is about technology, then it means it's off-topic. Point it out.

Secondly, please provide feedback for the essay with evidence from the original essay. 

Thirdly, take the statistics from {{STATS}} and answer the question if the essay has more than 250 words.

Lastly, correct all spelling, grammar, and punctuation mistakes in the essay without altering its ideas. Preserve all paragraph breaks and line breaks exactly as they appear in the original text if the structure is appropriate. Refine paragraph structure if needed. 

"""
feedback_prompt_2 = """
Present your feedback in JSON format, following this specific structure:

```json

{{

    "Task Response feedback": "Repeat the prompt. Summarize the thesis of the essay. Point out, if the main idea of the essay does not align with the topic. In case the essay is on-topic, evaluate how effectively the candidate answers the question and develops their ideas, including specific evidence (such as quotes) from the essay. Highlight any of the following issues if present: \n1)The essay contradicts or is off-topic with respect to the prompt.\n2. The essay contains fewer than 250 words.\n 3. The essay contains fewer than 20 words.",

    "Coherence and Cohesion feedback": "Offer detailed feedback on the organization and logical flow of ideas in the essay, citing specific examples. Identify and comment on the use of linking phrases. Indicate if the essay consists of fewer than three paragraphs, or if any paragraph contains more than one main idea.",

    "Lexical Resource feedback": "Provide an in-depth evaluation of the range and appropriateness of the vocabulary used in the essay, and support your feedback with examples. Academic style is preferred; note if the student adopts other styles, such as a narrative tone. Analyze vocabulary complexity based on the provided CEFR analysis.",

    "Grammatical Range and Accuracy feedback": "Deliver comprehensive feedback on the variety and accuracy of grammatical structures and punctuation in the essay, citing specific examples. Identify **only real spelling/grammar/punctuation errors** in this essay. Ignore correct words even if they look similar to common mistakes. List instances of advanced or complex sentence structures.",
    "Is off topic": "Yes, if the essay aligns with the {{PROMPT}}. No, if the thesis of the essay contradicts with the {{PROMPT}}.",
    "Word limit satisfied": "'Yes', if the esssay has more than 250 words. 'NOT_REALLY', if the essay has less than 250 words, but more than 20 words. 'No', if the essay has less than 20 words. Take the word count from the {STATS} as a reference."

    "Corrected essay": "Present the essay with all spelling, grammar, and punctuation errors corrected. Remember, do not delete "\\n". The paragraph structure should be preserved or improved if necessary. The corrected essay should have at least three paragraphs."

}} 

```"""
