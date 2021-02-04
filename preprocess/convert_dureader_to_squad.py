import json
import random
import re

from tqdm import tqdm

max_seq_len = 512
special_token = u'▲'


def convert(fname):
    examples = []
    fails = 0
    neg_num = 500
    neg_samples = []
    for line in open(fname):
        item = json.loads(line)
        neg_samples.append(item)
        if len(neg_samples) >= neg_num:
            break

    for i, line in enumerate(tqdm(open(fname))):
        source = json.loads(line)
        if (len(source['answer_spans']) == 0):
            continue
        if source['answers'] == []:
            continue
        if (source['match_scores'][0] < 0.8):
            continue
        if (source['answer_spans'][0][1] > max_seq_len):
            continue

        docs_index = source['answer_docs'][0]
        start_id = source['answer_spans'][0][0]
        end_id = source['answer_spans'][0][1] + 1  # !!!!!
        question_type = source['question_type']
        fake_answer = source['fake_answers'][0]
        fake_answer = re.sub('\s+', ' ', fake_answer)
        if len(fake_answer) > 450:
            continue

        question = source['question'].strip()
        answers = {
            "text": '',
            "answer_start": ''
        }

        paragraph = {
            "id": str(source['question_id']),
            "context": [],
            "qas": [
                    {
                        "question": question,
                        "ans_doc": '',
                        "id": str(source['question_id']) + '_0',
                        "answers":[answers]
                    }
            ]
        }

        success = True

        for doc_id, doc in enumerate(source['documents']):

            try:
                answer_passage_idx = doc['most_related_para']
            except:
                success = False
                fails += 1
                break

            assert answer_passage_idx == 0

            doc_tokens = doc['segmented_paragraphs'][answer_passage_idx]
            # 将换行符去掉
            doc_tokens = ['' if t ==
                          special_token else t for t in doc_tokens]

            ques_len = len(doc['segmented_title']) + 1
            doc_tokens = doc_tokens[ques_len:]

            if doc_id == docs_index:
                start_id, end_id = start_id - ques_len, end_id - ques_len

                if start_id >= end_id or end_id > len(doc_tokens) or start_id >= len(doc_tokens):
                    fails += 1
                    success = False
                    break

                new_doc_tokens = ""

                for idx, token in enumerate(doc_tokens):
                    if idx == start_id:
                        new_start_id = len(new_doc_tokens)
                        break
                    new_doc_tokens = re.sub('\s+', ' ', new_doc_tokens + token)

                new_doc_tokens = "".join(doc_tokens)
                new_doc_tokens = re.sub('\s+', ' ', new_doc_tokens)
                new_end_id = new_start_id + len(fake_answer)

                if fake_answer != "".join(new_doc_tokens[new_start_id:new_end_id]):
                    start = new_doc_tokens.find(fake_answer)
                    if start != -1:
                        new_start_id = start
                        new_end_id = new_start_id + len(fake_answer)
                    else:
                        fails += 1
                        success = False
                        break

                # 部分数据以句号开头
                if fake_answer[0] == '。':
                    fake_answer = fake_answer[1:]
                    new_start_id += 1
                assert fake_answer == "".join(
                    new_doc_tokens[new_start_id:new_end_id])
                answers['text'] = fake_answer
                answers['answer_start'] = new_start_id
                # 真实包含answer的文档
                paragraph['context'].append((new_doc_tokens, 1))
            else:
                new_doc_tokens = "".join(doc_tokens)
                new_doc_tokens = re.sub('\s+', ' ', new_doc_tokens)
                # 没有包含answer的文档
                paragraph['context'].append((new_doc_tokens, 0))

        if not success:
            continue

        while len(paragraph['context']) < 5:
            item = random.choice(neg_samples)
            if item['question'].strip() == question:
                continue
            doc = random.choice(item['documents'])
            doc_tokens = doc['segmented_paragraphs'][0]
            doc_tokens = [special_token if t ==
                          '[unused1]' else t for t in doc_tokens]
            ques_len = len(doc['segmented_title']) + 1
            doc_tokens = doc_tokens[ques_len:]
            new_doc_tokens = "".join(doc_tokens)
            new_doc_tokens = re.sub('\s+', ' ', new_doc_tokens)
            # 凑数的假文档
            paragraph['context'].append((new_doc_tokens, 2))

        if i >= neg_num:
            idx = random.randint(0, i)
            if idx < neg_num:
                neg_samples[idx] = source

        random.shuffle(paragraph['context'])
        ans_doc = -1
        contexts = []
        fake_docs = []
        for i, x in enumerate(paragraph['context']):
            if x[1] == 1:
                ans_doc = i
            if x[1] == 2:
                fake_docs.append(i)
            contexts.append(x[0])

        paragraph['context'] = contexts
        paragraph['qas'][0]['ans_doc'] = ans_doc
        paragraph['qas'][0]['fake_docs'] = fake_docs

        example = {
            "paragraphs": [paragraph],
            "id": str(source['question_id']),
            "title": question
        }

        examples.append(example)
    return examples


if __name__ == "__main__":
  fname = '/nfs/users/xueyou/data/corpus/dureader_v2/extracted/devset/search.dev.json'
  examples = convert(fname)
  data = {"version":"v1.0","data":examples}
  with open('/nfs/users/xueyou/data/corpus/dureader_v2/multi_passage/dev_v2.json','w') as f:
      f.write(json.dumps(data,ensure_ascii=False,indent=2))