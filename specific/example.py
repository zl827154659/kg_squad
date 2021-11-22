# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from utils.feature import Feature
import pdb

label_dict = {'A': 0, 'B': 1, 'X': 2}



"""
    example:
        id, text1, text2, text3, text4, text5, label
        text = ' question_stem  choice_text [SEP] qc_meaning [SEP] ac_meaning [SEP] triples
"""
class ConceptNetExample:
    def __init__(self, idx, choice1, choice2, label = -1):
        self.idx = idx
        self.text1 = choice1
        self.text2 = choice2
        self.label = int(label)
    
    def __str__(self):
        return f"{self.idx} | {self.text1} | {self.text2} | {self.label}"
        
    def fl(self, tokenizer, max_seq_length):
        fs = self.f(tokenizer, max_seq_length)
        return (*fs, self.label)
        
    def f(self, tokenizer, max_seq_length):
        tokens1 = tokenizer.tokenize(self.text1)
        tokens2 = tokenizer.tokenize(self.text2)


        feature1 = Feature.make_single(self.idx, tokens1, tokenizer, max_seq_length)
        feature2 = Feature.make_single(self.idx, tokens2, tokenizer, max_seq_length)
        return (feature1, feature2)
        
        
    @classmethod
    def load_from_json(cls, json_obj, append_answer_text=False, append_descr=0, append_triple=True, append_context=False):
        choices = json_obj['question']['choices']
        question_concept = json_obj['question']['question_concept']
        context = json_obj['context']
        def mkinput(question_concept, choice):
            if choice['triple'] and append_triple:
                triples = ' [SEP] '.join([' '.join(trip) for trip in choice['triple']])
                first_triple = ' '.join(choice['triple'][0])
                following_triple = ' [SEP] '.join([' '.join(trip) for trip in choice['triple'][1:]]) if len(choice['triple']) > 1 else None
                triples_temp = triples
            else:
                triples_temp = question_concept + ' [SEP] ' + choice['text']
                following_triple = None
            if append_answer_text:
                question_text = '{} {}'.format(json_obj['question']['stem'], choice['text'])
            else:
                question_text = json_obj['question']['stem']
            if append_descr == 1:
                triples_temp = '{} [SEP] {} [SEP] {}'.format(json_obj['question']['qc_meaning'], choice['ac_meaning'], triples_temp)
            elif append_descr == 2:
                triples_temp = '{} [SEP] {} [SEP] {}'.format(triples_temp, json_obj['question']['qc_meaning'], choice['ac_meaning']) if following_triple is None else \
                    '{} [SEP] {} [SEP] {} [SEP] {}'.format(first_triple, json_obj['question']['qc_meaning'], choice['ac_meaning'], following_triple)
            
            text = ' {} [SEP] {} '.format(question_text, triples_temp)
            if append_context:
                text = text + f'[SEP] {context}'
            return text

        texts = []
        for i, choice in enumerate(choices):
            texts.append(mkinput(question_concept, choice))
        # text1 = mkinput(question_concept, choices[0])
        # text2 = mkinput(question_concept, choices[1])
        # text3 = mkinput(question_concept, choices[2])
        # text4 = mkinput(question_concept, choices[3])
        # text5 = mkinput(question_concept, choices[4])
        try:
            label =  label_dict[json_obj['answerKey']]
        except:
            label = -1
        while len(texts) < 2:
            texts.append(mkinput(question_concept, {
                'label': 'X',
                'text': '',
                'triple': ['', '', ''],
                'surface': '',
                'weight': 0,
                'ac_meaning': ''
            }))
        # return cls(
        #     json_obj['initial_id'],
        #     text1,
        #     text2,
        #     text3,
        #     text4,
        #     text5,
        #     label,
        # )
        return cls(
            json_obj['initial_id'],
            texts[0],
            texts[1],
            label,
        )

    def to_json(self):
        return {
            'ID': self.idx,
            'Text1': self.text1,
            'Text2': self.text2,
            'Label': self.label
        }