import collections
import json
import re
import string

import pandas as pd

label_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'X': 7}


def main(prediction_file:str, data_file:str, output:str):
    pred_data = []
    df = pd.read_csv(prediction_file, header=None, names=['idx', 'result1', 'result2', 'result3', 'result4', 'result5', 'label', 'pred_label'])
    for row in df.iterrows():
        pred_data.append({
            'id': row[1]['idx'],
            'label': row[1]['label'],
            'pred_label': row[1]['pred_label']
        })
    with open(data_file, 'r') as f:
        dataset = json.load(f)
    real_answer = {}
    for data in dataset:
        id = data['initial_id']
        choices = [choice['text'] for choice in data['question']['choices']]
        real_answer[id] = choices
    for data in pred_data:
        id = data['id']
        label = data['label']
        pred = data['pred_label']
        data['real_answer'] = real_answer[id][int(label)]
        data['prediction'] = real_answer[id][int(pred)]

    with open(output, 'w') as f:
        json.dump(pred_data, f)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_raw_scores(dataset):
    exact_scores = {}
    f1_scores = {}

    for data in dataset:
        qid = data['id']
        real_answer = data['real_answer']
        prediction = data['prediction']
        exact_scores[qid] = compute_exact(real_answer, prediction)
        f1_scores[qid] = compute_f1(real_answer, prediction)
    return exact_scores, f1_scores

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    new_scores = {}
    for qid, s in scores.items():
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores


def make_qid_to_has_ans(dataset):
    qid_to_has_ans = {}
    for data in dataset:
        qid_to_has_ans[data['id']] = bool(data['real_answer'])
    return qid_to_has_ans

def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores.values()) / total),
            ('f1', 100.0 * sum(f1_scores.values()) / total),
            ('total', total),
        ])
    else:
        total = len(qid_list)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total),
            ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
            ('total', total),
        ])

def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval['%s_%s' % (prefix, k)] = new_eval[k]

def evaluate_squad(data_file, verbose=False):
    with open(data_file, 'r') as f:
        dataset = json.load(f)
    na_probs = {k['id']: 0.0 for k in dataset}
    exact_raw, f1_raw = get_raw_scores(dataset)
    qid_to_has_ans = make_qid_to_has_ans(dataset)  # maps qid to True/False
    exact_thresh = apply_no_ans_threshold(exact_raw, na_probs, qid_to_has_ans,
                                          1.0)
    f1_thresh = apply_no_ans_threshold(f1_raw, na_probs, qid_to_has_ans,
                                       1.0)
    out_eval = make_eval_dict(exact_thresh, f1_thresh)
    return out_eval


if __name__ == '__main__':
    main('/home/ray/project/DEKCOR-CommonsenseQA/eval/kg_squad_only_context_output/dev/kg_squad_pred.csv',
         '/home/ray/dataset/kg_squad/dev_data.json',
         '/home/ray/project/DEKCOR-CommonsenseQA/eval/kg_squad_only_context_output/dev/answer_prediction.json')
    print(evaluate_squad('/home/ray/project/DEKCOR-CommonsenseQA/eval/kg_squad_only_context_output/dev/answer_prediction.json'))
    # exact低很多是因为这个是句子级的匹配，要答案完全一致才算正确，而F1值是按字符级运算准确率而得的
