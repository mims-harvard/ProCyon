import torch
import random
import numpy as np

def aaseq_type_to_prompt(aaseq_type):
    if isinstance(aaseq_type, str):
        aaseq_type = aaseq_type.lower()

    if aaseq_type == 'protein':
        return "Protein"
    elif aaseq_type == 'domain':
        return "Domain"
    elif aaseq_type == "peptide":
        return "Peptide"
    else: # Default for now, can change later if we need peptide
        return "Amino acid sequence"

def compose_qa_examples(examples, pos_neg, num_examples=None, is_ppi=False, sample_examples=False, aaseq_type=None):
    header = "Positive example" if pos_neg == "positive" else "Negative example"
    output = "yes" if pos_neg == "positive" else "no"
    num_examples = len(examples) if num_examples is None else num_examples # Number of examples to include
    aaseq = aaseq_type_to_prompt(aaseq_type)
    if is_ppi:
        if sample_examples:
            example_text = [
                f"""{header} {i+1}:\n{aaseq} 1: <|protein|>\n{aaseq} 2: <|protein|>\nOutput: [ANSWER] {output}"""
                for i, example in enumerate(examples)
                if i < num_examples
            ]
        else:
            example_text = "\n".join([
                f"""{header} {i+1}:\n{aaseq} 1: <|protein|>\n{aaseq} 2: <|protein|>\nOutput: [ANSWER] {output}"""
                for i, example in enumerate(examples)
                if i < num_examples
            ])
        text_ids = []
        aaseq_ids = sum([[example['aaseq_1'], example['aaseq_2']] for i, example in enumerate(examples) if i < num_examples], start=[])
    else:
        if sample_examples:
            example_text = [
                f"{header} {i+1}:\nDescription: [EXT]\n{aaseq}: <|protein|>\n[CONTEXT]Output: [ANSWER] {output}"
                for i, example in enumerate(examples)
                if i < num_examples
            ]
        else:
            example_text = "\n".join([
                f"{header} {i+1}:\nDescription: [EXT]\n{aaseq}: <|protein|>\n[CONTEXT]Output: [ANSWER] {output}"
                for i, example in enumerate(examples)
                if i < num_examples
            ])
        text_ids = [example['text'] for i, example in enumerate(examples) if i < num_examples]
        aaseq_ids = [example['aaseq'] for i, example in enumerate(examples) if i < num_examples]
    return example_text, text_ids, aaseq_ids

def compose_retrieval_examples(examples, pos_neg, num_examples=None, is_ppi=False, sample_examples = False, aaseq_type=None):
    header = "Positive example" if pos_neg == "positive" else "Negative example"
    num_examples = len(examples) if num_examples is None else num_examples 
    aaseq = aaseq_type_to_prompt(aaseq_type)
    if is_ppi:
        if sample_examples:
            example_text = [
                f"{header} {i+1}:\n{aaseq}: <|protein|>\n{aaseq}: <|protein|>"
                for i, example in enumerate(examples)
                if i < num_examples
            ] # Can cut this off to reduce number of examples
        else:
            example_text = "\n".join([
                f"{header} {i+1}:\n{aaseq} 1: <|protein|>\n{aaseq} 2: <|protein|>"
                for i, example in enumerate(examples)
                if i < num_examples
            ]) # Can cut this off to reduce number of examples
        text_ids = []
        aaseq_ids = sum([[example['aaseq_1'], example['aaseq_2']] for i, example in enumerate(examples) if i < num_examples], start=[])
    else:
        if sample_examples: # This statement takes out the join of the next statement to avoid constructing a string at this point
            example_text = [
                f"{header} {i+1}:\n[CONTEXT]Description: [EXT]\n{aaseq}: <|protein|>"
                for i, example in enumerate(examples)
                if i < num_examples
            ] # Can cut this off to reduce number of examples
        else:
            example_text = "\n".join([
                f"{header} {i+1}:\n[CONTEXT]Description: [EXT]\n{aaseq}: <|protein|>"
                for i, example in enumerate(examples)
                if i < num_examples
            ]) # Can cut this off to reduce number of examples
        text_ids = [example['text'] for i, example in enumerate(examples) if i < num_examples]
        aaseq_ids = [example['aaseq'] for i, example in enumerate(examples) if i < num_examples]
    return example_text, text_ids, aaseq_ids

def compose_caption_examples(examples, pos_neg, num_examples=None, is_ppi=False, sample_examples=False, aaseq_type=None):
    header = "Positive example" if pos_neg == "positive" else "Negative example"
    num_examples = len(examples) if num_examples is None else num_examples 
    aaseq = aaseq_type_to_prompt(aaseq_type)
    if sample_examples:
        example_text = [
            f"{header} {i+1}:\n[CONTEXT]{aaseq}: <|protein|>\nOutput: [ANSWER] [EXT]" \
            for i, example in enumerate(examples)
            if i < num_examples
        ]
    else:
        example_text = "\n".join([
            f"{header} {i+1}:\n[CONTEXT]{aaseq}: <|protein|>\nOutput: [ANSWER] [EXT]" \
            for i, example in enumerate(examples)
            if i < num_examples
        ])
    text_ids = [example['text'] for i, example in enumerate(examples) if i < num_examples]
    aaseq_ids = [example['aaseq'] for i, example in enumerate(examples) if i < num_examples]
    return example_text, text_ids, aaseq_ids

def get_prompt(task, num_examples=None, is_special_definition=False, is_ppi=False, sample_examples=False, aaseq_type=None):
    aaseq = aaseq_type_to_prompt(aaseq_type)
    if is_special_definition:
        definition = task["Definition"]
    else:
        definition = (
            task['Definition'].
            replace("{Relationship Summary}", task['Relationship Summary']).
            replace("{Biological Summary}", task['Biological Summary']).
            replace("{Task-Specific Relationship}", task['Task-Specific Relationship'])
        )
    
    if task['CATEGORY'] == 'qa':
        positive_examples, positive_text_ids, positive_aaseq_ids = compose_qa_examples(task['Positive Examples'], 'positive', num_examples = num_examples, is_ppi = is_ppi, sample_examples = sample_examples, aaseq_type = aaseq_type)
        negative_examples, negative_text_ids, negative_aaseq_ids = compose_qa_examples(task['Negative Examples'], 'negative', num_examples = num_examples, is_ppi = is_ppi, sample_examples = sample_examples, aaseq_type = aaseq_type)
        if is_ppi:
            if sample_examples:
                prompt = f"""Now, complete the following instance:
{aaseq} 1: <|protein|>
{aaseq} 2: <|protein|>
Output: [ANSWER] """
                prompt = "{positive_examples}{negative_examples}\n" + prompt
                prompt += "{answer}"
                prompt = f"Definition: {definition}" + prompt
            else:
                prompt = f"""Definition: {definition}
{positive_examples}
{negative_examples}
Now, complete the following instance:
{aaseq} 1: <|protein|>
{aaseq} 2: <|protein|>
Output: [ANSWER] """
                prompt += "{answer}"
            text_ids = []
            aaseq_ids = positive_aaseq_ids + negative_aaseq_ids
        else:
            if sample_examples:
                prompt = f"""Now, complete the following instance:
Description: [EXT]
{aaseq}: <|protein|>
[CONTEXT]Output: [ANSWER] """
                prompt = "{positive_examples}{negative_examples}\n" + prompt
                prompt += "{answer}"
                prompt = f"Definition: {definition}" + prompt
            else:
                prompt = f"""Definition: {definition}
{positive_examples}
{negative_examples}
Now, complete the following instance:
Description: [EXT]
{aaseq}: <|protein|>
[CONTEXT]Output: [ANSWER] """
                prompt += "{answer}"
            text_ids = positive_text_ids + negative_text_ids
            aaseq_ids = positive_aaseq_ids + negative_aaseq_ids
    elif task['CATEGORY'] == 'retrieval':
        positive_examples, positive_text_ids, positive_aaseq_ids = compose_retrieval_examples(task['Positive Examples'], 'positive', num_examples = num_examples, is_ppi = is_ppi, sample_examples = sample_examples, aaseq_type = aaseq_type)
        negative_examples = None
        #negative_examples, negative_text_ids, negative_aaseq_ids = compose_retrieval_examples(task['Negative Examples'], 'negative', num_examples = num_examples, is_ppi = is_ppi, sample_examples = sample_examples)
        if is_ppi:
            if sample_examples:
                prompt = f"""Now, complete the following instance:
{aaseq} 1: <|protein|> 
{aaseq} 2: [PROT]"""
                prompt = "{positive_examples}\n" + prompt
                prompt = f"Definition: {definition}" + prompt
            else:
                prompt = f"""Definition: {definition}
{positive_examples}
Now, complete the following instance:
{aaseq} 1: <|protein|> 
{aaseq} 2: [PROT]"""
            text_ids = []
            aaseq_ids = positive_aaseq_ids #+ negative_aaseq_ids
        else:
            if sample_examples:
                prompt = f"""Now, complete the following instance:
[CONTEXT]Description: [EXT]
{aaseq}: [PROT]"""
                prompt = "{positive_examples}\n" + prompt
                prompt = f"Definition: {definition}" + prompt
            else:
                prompt = f"""Definition: {definition}
{positive_examples}
Now, complete the following instance:
[CONTEXT]Description: [EXT]
{aaseq}: [PROT]"""
            text_ids = positive_text_ids #+ negative_text_ids
            aaseq_ids = positive_aaseq_ids #+ negative_aaseq_ids
    elif task['CATEGORY'] == 'caption':
        assert is_ppi == False, "Cannot use PPI with caption task"
        positive_examples, positive_text_ids, positive_aaseq_ids = compose_caption_examples(task['Positive Examples'], 'positive', num_examples = num_examples, sample_examples = sample_examples, aaseq_type = aaseq_type)
        negative_examples = None
        #negative_examples, negative_text_ids, negative_aaseq_ids = compose_caption_examples(task['Negative Examples'], 'negative', num_examples = num_examples, sample_examples = sample_examples)
        if sample_examples:
            prompt = f"""
Now, complete the following instance:
[CONTEXT]{aaseq}: <|protein|>
Output: [ANSWER] [EXT]"""
            prompt = "{positive_examples}\n" + prompt
            prompt = f"Definition: {definition}" + prompt
        else:
            prompt = f"""Definition: {definition}
{positive_examples}
Now, complete the following instance:
[CONTEXT]{aaseq}: <|protein|>
Output: [ANSWER] [EXT]"""
        text_ids = positive_text_ids #+ negative_text_ids
        aaseq_ids = positive_aaseq_ids #+ negative_aaseq_ids

    # Replace context:
    # prompt = prompt.replace("[CONTEXT]", "{context}")
    # if sample_examples:
    #     for i in range(len(positive_examples)):
    #         positive_examples[i] = positive_examples[i].replace("[CONTEXT]", "{context}")
    # else:
    #     positive_examples = positive_examples.replace("[CONTEXT]", "{context}")
        
    # if sample_examples:
    #     for i in range(len(negative_examples)):
    #         negative_examples[i] = negative_examples[i].replace("[CONTEXT]", "{context}")
    # elif negative_examples is not None:
    #     negative_examples = negative_examples.replace("[CONTEXT]", "{context}")

    return prompt, positive_examples, negative_examples, text_ids, aaseq_ids

def get_prompt_open_def(task, num_examples=None, is_special_definition=False, is_ppi=False, sample_examples=False, aaseq_type=None):
    aaseq = aaseq_type_to_prompt(aaseq_type)
    if is_special_definition:
        definition_true = task["Definition"]
    else:
        definition_true = (
            task['Definition'].
            replace("{Relationship Summary}", task['Relationship Summary']).
            replace("{Biological Summary}", task['Biological Summary']).
            replace("{Task-Specific Relationship}", task['Task-Specific Relationship'])
        )

    definition = "{definition}" # Placeholder

    assert not sample_examples, "Not supported"
    
    if task['CATEGORY'] == 'qa':
        positive_examples, positive_text_ids, positive_aaseq_ids = compose_qa_examples(task['Positive Examples'], 'positive', num_examples = num_examples, is_ppi = is_ppi, sample_examples = sample_examples, aaseq_type = aaseq_type)
        negative_examples, negative_text_ids, negative_aaseq_ids = compose_qa_examples(task['Negative Examples'], 'negative', num_examples = num_examples, is_ppi = is_ppi, sample_examples = sample_examples, aaseq_type = aaseq_type)
        if is_ppi:
            if sample_examples:
                prompt = f"""Now, complete the following instance:
{aaseq} 1: <|protein|>
{aaseq} 2: <|protein|>
Output: [ANSWER] """
                prompt = "{positive_examples}{negative_examples}\n" + prompt
                prompt += "{answer}"
                prompt = f"Definition: {definition}" + prompt
            else:
                prompt = f"""Definition: {definition}
{positive_examples}
{negative_examples}
Now, complete the following instance:
{aaseq} 1: <|protein|>
{aaseq} 2: <|protein|>
Output: [ANSWER] """
                prompt += "{answer}"
            text_ids = []
            aaseq_ids = positive_aaseq_ids + negative_aaseq_ids
        else:
            if sample_examples:
                prompt = f"""Now, complete the following instance:
Description: [EXT]
{aaseq}: <|protein|>
[CONTEXT]Output: [ANSWER] """
                prompt = "{positive_examples}{negative_examples}\n" + prompt
                prompt += "{answer}"
                prompt = f"Definition: {definition}" + prompt
            else:
                prompt = f"""Definition: {definition}
{positive_examples}
{negative_examples}
Now, complete the following instance:
Description: [EXT]
{aaseq}: <|protein|>
[CONTEXT]Output: [ANSWER] """
                prompt += "{answer}"
            text_ids = positive_text_ids + negative_text_ids
            aaseq_ids = positive_aaseq_ids + negative_aaseq_ids
    elif task['CATEGORY'] == 'retrieval':
        positive_examples, positive_text_ids, positive_aaseq_ids = compose_retrieval_examples(task['Positive Examples'], 'positive', num_examples = num_examples, is_ppi = is_ppi, sample_examples = sample_examples, aaseq_type = aaseq_type)
        negative_examples = None
        #negative_examples, negative_text_ids, negative_aaseq_ids = compose_retrieval_examples(task['Negative Examples'], 'negative', num_examples = num_examples, is_ppi = is_ppi, sample_examples = sample_examples)
        if is_ppi:
            if sample_examples:
                prompt = f"""Now, complete the following instance:
{aaseq} 1: <|protein|> 
{aaseq} 2: [PROT]"""
                prompt = "{positive_examples}\n" + prompt
                prompt = f"Definition: {definition}" + prompt
            else:
                prompt = f"""Definition: {definition}
{positive_examples}
Now, complete the following instance:
{aaseq} 1: <|protein|> 
{aaseq} 2: [PROT]"""
            text_ids = []
            aaseq_ids = positive_aaseq_ids #+ negative_aaseq_ids
        else:
            if sample_examples:
                prompt = f"""Now, complete the following instance:
[CONTEXT]Description: [EXT]
{aaseq}: [PROT]"""
                prompt = "{positive_examples}\n" + prompt
                prompt = f"Definition: {definition}" + prompt
            else:
                prompt = f"""Definition: {definition}
{positive_examples}
Now, complete the following instance:
[CONTEXT]Description: [EXT]
{aaseq}: [PROT]"""
            text_ids = positive_text_ids #+ negative_text_ids
            aaseq_ids = positive_aaseq_ids #+ negative_aaseq_ids
    elif task['CATEGORY'] == 'caption':
        assert is_ppi == False, "Cannot use PPI with caption task"
        positive_examples, positive_text_ids, positive_aaseq_ids = compose_caption_examples(task['Positive Examples'], 'positive', num_examples = num_examples, sample_examples = sample_examples, aaseq_type = aaseq_type)
        negative_examples = None
        #negative_examples, negative_text_ids, negative_aaseq_ids = compose_caption_examples(task['Negative Examples'], 'negative', num_examples = num_examples, sample_examples = sample_examples)
        if sample_examples:
            prompt = f"""
Now, complete the following instance:
[CONTEXT]{aaseq}: <|protein|>
Output: [ANSWER] [EXT]"""
            prompt = "{positive_examples}\n" + prompt
            prompt = f"Definition: {definition}" + prompt
        else:
            prompt = f"""Definition: {definition}
{positive_examples}
Now, complete the following instance:
[CONTEXT]{aaseq}: <|protein|>
Output: [ANSWER] [EXT]"""
        text_ids = positive_text_ids #+ negative_text_ids
        aaseq_ids = positive_aaseq_ids #+ negative_aaseq_ids

    # Replace context:
    # prompt = prompt.replace("[CONTEXT]", "{context}")
    # if sample_examples:
    #     for i in range(len(positive_examples)):
    #         positive_examples[i] = positive_examples[i].replace("[CONTEXT]", "{context}")
    # else:
    #     positive_examples = positive_examples.replace("[CONTEXT]", "{context}")
        
    # if sample_examples:
    #     for i in range(len(negative_examples)):
    #         negative_examples[i] = negative_examples[i].replace("[CONTEXT]", "{context}")
    # elif negative_examples is not None:
    #     negative_examples = negative_examples.replace("[CONTEXT]", "{context}")

    return prompt, definition_true, positive_examples, negative_examples, text_ids, aaseq_ids


def sample_demonstrations_for_prompts(
        template, 
        positive_examples, 
        negative_examples, 
        example_text_ids, 
        example_aaseq_ids,
        is_ppi = False,
    ):
    '''
    Performs sampling of the number of examples included in a given instruction prompt
    '''

    assert negative_examples is not None, "FIX THE NEGATIVE EXAMPLES PROBLEM BEFORE USING THIS FUNCTION"

    if is_ppi:
        all_lens = [len(positive_examples) + len(negative_examples), len(example_aaseq_ids) // 2]
        # example_aaseq_ids should be even number, so floor works
    else:
        all_lens = [len(positive_examples) + len(negative_examples), len(example_text_ids), len(example_aaseq_ids)]

    common_len = np.unique(all_lens)
    if len(common_len) > 1:
        print('common len', common_len)
    assert len(common_len) == 1, "Not all given examples are of same length"
    L = common_len[0]
        
    # Step 1: sample indices:
    num_ex = np.random.randint(0, (L // 2) + 1)

    if num_ex == 0:
        if "{answer}" in template:
            return template.format(positive_examples='', negative_examples='', answer = "{answer}"), [], []
        else:
            return template.format(positive_examples='', negative_examples=''), [], []

    # Step 2: index examples and join via \n
    pos_ex_all = "\n" + "\n".join(positive_examples[:num_ex])
    neg_ex_all = "\n" + "\n".join(negative_examples[:num_ex])

    # Step 3: choose out text ids and aaseq_ids based on sampling num_ex b/w the desired number
    if is_ppi:
        tup_pairs = [(i, i + 1) for i in np.arange(0, (len(example_aaseq_ids) - 1), 2)]
        pos_ex_inds = np.random.choice(np.arange(L // 2), size = (num_ex,), replace = False)
        neg_ex_inds = np.random.choice(np.arange(L // 2), size = (num_ex,), replace = False) + (L // 2)

        pos_exs, neg_exs = [], []
        for i in range(num_ex):
            pi1, pi2 = tup_pairs[pos_ex_inds[i]]
            pos_exs.append(pi1); pos_exs.append(pi2)

            ni1, ni2 = tup_pairs[neg_ex_inds[i]]
            neg_exs.append(ni1); neg_exs.append(ni2)

        example_aaseq_ids_final = pos_exs + neg_exs
        example_text_ids_final = None

    else:
        pos_ex_inds = np.random.choice(np.arange(L // 2), size = (num_ex,), replace = False)
        neg_ex_inds = np.random.choice(np.arange(L // 2), size = (num_ex,), replace = False) + (L // 2)
        # Add L//2 bc neg's are stacked at the end

        example_text_ids_final = [example_text_ids[i] for i in pos_ex_inds] + [example_text_ids[i] for i in neg_ex_inds]
        example_aaseq_ids_final = [example_aaseq_ids[i] for i in pos_ex_inds] + [example_aaseq_ids[i] for i in neg_ex_inds]

    # Step 4: Replace in positive example and negative example strings to the template
    if "{answer}" in template:
        new_template = template.format(positive_examples = pos_ex_all, negative_examples = neg_ex_all, answer = "{answer}")
    else:
        new_template = template.format(positive_examples = pos_ex_all, negative_examples = neg_ex_all)

    return new_template, example_text_ids_final, example_aaseq_ids_final